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
    message: str = Field(min_length=1, max_length=20000)


class ModelRequest(BaseModel):
    model: str


def create_app() -> FastAPI:
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
                with runtime.turn_lock:
                    answer = runtime.get_state().agent.run_turn(
                        message,
                        on_content_delta=lambda delta: emit(
                            {"event": "content_delta", "delta": delta}
                        ),
                        on_reasoning_delta=lambda delta: emit(
                            {"event": "reasoning_delta", "delta": delta}
                        ),
                        on_trace=lambda trace_event: emit(_map_trace_event(trace_event)),
                    )

                if answer:
                    emit({"event": "final_answer", "content": answer})
            except Exception as exc:  # pragma: no cover - runtime guard
                emit({"event": "error", "message": str(exc)})
            finally:
                emit({"event": "done"})
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        async def stream() -> Any:
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


app = create_app()
