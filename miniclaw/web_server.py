from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import MiniClawAgent
from .history import clear_history, load_history
from .llm import ConfigError, DeepSeekClient
from .tools import create_default_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
MODELS = ("deepseek-chat", "deepseek-reasoner")

app = FastAPI(title="MiniClaw Web", version="0.1.0")
if WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

_AGENT_LOCK = threading.Lock()
_AGENT: MiniClawAgent | None = None
_CLIENT: DeepSeekClient | None = None
_BOOT_ERROR: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class ModelRequest(BaseModel):
    model: str


def _get_or_create_agent() -> MiniClawAgent:
    global _AGENT, _CLIENT, _BOOT_ERROR

    if _BOOT_ERROR:
        raise RuntimeError(_BOOT_ERROR)

    if _AGENT is not None:
        return _AGENT

    try:
        client = DeepSeekClient.from_env()
    except ConfigError as exc:
        _BOOT_ERROR = str(exc)
        raise RuntimeError(_BOOT_ERROR) from exc

    history = load_history()
    if history:
        agent = MiniClawAgent(
            client=client,
            tools=create_default_registry(client=client),
            messages=history,
        )
    else:
        agent = MiniClawAgent(client=client, tools=create_default_registry(client=client))

    _CLIENT = client
    _AGENT = agent
    return agent


def _get_client() -> DeepSeekClient:
    _get_or_create_agent()
    if _CLIENT is None:
        raise RuntimeError("Web 客户端初始化失败。")
    return _CLIENT


def _to_sse_message(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _map_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    phase = str(event.get("type") or "unknown")
    mapped: dict[str, Any] = {
        "event": "tao",
        "phase": phase,
        "step": int(event.get("step") or 0),
    }

    if "tool_index" in event:
        try:
            mapped["tool_index"] = int(event.get("tool_index") or 0)
        except (TypeError, ValueError):
            mapped["tool_index"] = 0

    if phase in {"action", "action_delta"}:
        mapped["tool"] = str(event.get("tool") or "")
        mapped["arguments"] = event.get("arguments") or {}
    elif phase in {
        "thought",
        "thought_delta",
        "observation_start",
        "observation_delta",
        "observation",
    }:
        mapped["content"] = str(event.get("content") or "")
        if "tool" in event:
            mapped["tool"] = str(event.get("tool") or "")
        if "data" in event:
            mapped["data"] = event.get("data") or {}

    return mapped


@app.get("/", response_model=None)
def root() -> Any:
    index_path = WEB_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "error": "未找到 web/index.html，请先创建前端页面。",
        },
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "MiniClaw Web"}


@app.get("/api/models")
def get_models() -> JSONResponse:
    try:
        with _AGENT_LOCK:
            client = _get_client()
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "current": client.model,
            "models": list(MODELS),
        },
    )


@app.post("/api/model")
def set_model(request: ModelRequest) -> JSONResponse:
    if request.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型：{request.model}。可选：{', '.join(MODELS)}",
        )

    try:
        with _AGENT_LOCK:
            client = _get_client()
            client.model = request.model
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    return JSONResponse(status_code=200, content={"ok": True, "model": request.model})


@app.post("/api/clear")
def clear_chat() -> JSONResponse:
    try:
        with _AGENT_LOCK:
            agent = _get_or_create_agent()
            agent.reset_messages()
            clear_history()
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    return JSONResponse(status_code=200, content={"ok": True})


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空。")

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(payload: dict[str, Any]) -> None:
        event_queue.put(payload)

    def worker() -> None:
        try:
            with _AGENT_LOCK:
                agent = _get_or_create_agent()
                answer = agent.run_turn(
                    message,
                    on_content_delta=lambda delta: emit(
                        {
                            "event": "content_delta",
                            "delta": delta,
                        }
                    ),
                    on_reasoning_delta=lambda delta: emit(
                        {
                            "event": "reasoning_delta",
                            "delta": delta,
                        }
                    ),
                    on_trace=lambda trace_event: emit(_map_trace_event(trace_event)),
                )

            if answer:
                emit(
                    {
                        "event": "final_answer",
                        "content": answer,
                    }
                )
        except Exception as exc:  # pragma: no cover - runtime guard
            emit({"event": "error", "message": str(exc)})
        finally:
            emit({"event": "done"})
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Generator[str, None, None]:
        while True:
            try:
                payload = event_queue.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

            if payload is None:
                break
            yield _to_sse_message(payload)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("miniclaw.web_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
