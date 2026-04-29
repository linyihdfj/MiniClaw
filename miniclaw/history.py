from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "workspace" / "history.json"


def load_history() -> list[ModelMessage]:
    if not HISTORY_PATH.is_file():
        return []

    raw_text = HISTORY_PATH.read_text(encoding="utf-8")
    try:
        return list(ModelMessagesTypeAdapter.validate_json(raw_text))
    except Exception:
        legacy_messages = _load_legacy_messages(raw_text)
        save_history(legacy_messages)
        return legacy_messages


def save_history(messages: list[ModelMessage]) -> None:
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.write_bytes(ModelMessagesTypeAdapter.dump_json(messages, indent=2))


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def _load_legacy_messages(raw_text: str) -> list[ModelMessage]:
    import json

    raw_messages = json.loads(raw_text)
    if not isinstance(raw_messages, list):
        return []

    messages: list[ModelMessage] = []
    tool_name_by_id: dict[str, str] = {}

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue

        role = str(raw_message.get("role") or "")
        content = raw_message.get("content")

        if role == "system":
            messages.append(ModelRequest(parts=[SystemPromptPart(content=str(content or ""))]))
            continue

        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=str(content or ""))]))
            continue

        if role == "assistant":
            parts: list[Any] = []
            if content:
                parts.append(TextPart(content=str(content)))

            for raw_tool_call in raw_message.get("tool_calls") or []:
                if not isinstance(raw_tool_call, dict):
                    continue
                function_payload = raw_tool_call.get("function") or {}
                if not isinstance(function_payload, dict):
                    function_payload = {}

                tool_name = str(function_payload.get("name") or "unknown_tool")
                tool_call_id = str(raw_tool_call.get("id") or "")
                tool_args = str(function_payload.get("arguments") or "{}")
                part = ToolCallPart(
                    tool_name=tool_name,
                    args=tool_args,
                    tool_call_id=tool_call_id,
                )
                parts.append(part)
                if tool_call_id:
                    tool_name_by_id[tool_call_id] = tool_name

            if parts:
                messages.append(ModelResponse(parts=parts, model_name="legacy-history"))
            continue

        if role == "tool":
            tool_call_id = str(raw_message.get("tool_call_id") or "")
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=tool_name_by_id.get(tool_call_id, "unknown_tool"),
                            content=str(content or ""),
                            tool_call_id=tool_call_id,
                        )
                    ]
                )
            )

    return messages
