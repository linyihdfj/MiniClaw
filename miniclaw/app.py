from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from .runtime import RuntimeError, runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"


class ChatRequest(BaseModel):
    # Web 端单次提问的输入模型。
    message: str = Field(min_length=1, max_length=20000)


class ModelRequest(BaseModel):
    # 模型切换接口的输入模型。
    model: str


def create_app() -> FastAPI:
    # Web 和 CLI 共用同一份 runtime；这里仅负责 HTTP/SSE 封装。
    app = FastAPI(title="MiniClaw Web", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if WEB_DIR.is_dir():
        app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

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
            model_info = runtime.get_model_info()
        except RuntimeError as exc:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

        return JSONResponse(status_code=200, content={"ok": True, **model_info})

    @app.post("/api/model")
    def set_model(request: ModelRequest) -> JSONResponse:
        try:
            model = runtime.set_model(request.model)
        except RuntimeError as exc:
            message = str(exc)
            status_code = 400 if message.startswith("不支持的模型") else 500
            return JSONResponse(status_code=status_code, content={"ok": False, "error": message})

        return JSONResponse(status_code=200, content={"ok": True, "model": model})

    @app.post("/api/clear")
    def clear_chat() -> JSONResponse:
        try:
            runtime.reset()
        except RuntimeError as exc:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

        return JSONResponse(status_code=200, content={"ok": True})

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> EventSourceResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message 不能为空。")

        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def emit(payload: dict[str, Any]) -> None:
            event_queue.put(payload)

        def worker() -> None:
            try:
                # 共享 turn_lock，避免 Web 和 CLI 同时修改同一段对话历史。
                with runtime.turn_lock:
                    runtime.get_agent().run_turn(
                        message,
                        on_trace=lambda trace_event: emit(_map_trace_event(trace_event)),
                        on_stream_event=emit,
                    )
            except Exception as exc:  # pragma: no cover - runtime guard
                emit({"event": "error", "message": str(exc)})
            finally:
                emit({"event": "done"})
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        async def stream() -> Any:
            # Agent 运行在线程里，SSE 响应在异步协程里消费队列并持续推送。
            while True:
                payload = await asyncio.to_thread(event_queue.get)
                if payload is None:
                    break
                yield JSONServerSentEvent(data=payload)

        return EventSourceResponse(
            stream(),
            ping=15,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _map_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    # 前端不直接依赖 pydantic-ai 的原始事件格式，这里统一转成稳定的 UI 事件。
    phase = str(event.get("type") or "unknown")
    mapped: dict[str, Any] = {
        "event": "tao",
        "phase": phase,
        "step": int(event.get("step") or 0),
        "agent_role": str(event.get("agent_role") or "main"),
        "agent_name": str(event.get("agent_name") or "MiniClaw"),
        "stream_id": str(event.get("stream_id") or "main"),
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
        "delegation_start",
        "delegation_progress",
        "delegation_result",
    }:
        mapped["content"] = str(event.get("content") or "")
        if "tool" in event:
            mapped["tool"] = str(event.get("tool") or "")
        if "data" in event:
            mapped["data"] = event.get("data") or {}

    return mapped


# Uvicorn 默认加载的就是这个模块级 app。
app = create_app()
