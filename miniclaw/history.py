from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "workspace" / "history.json"


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    raw_messages = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    clean_messages = [_clean_message(message) for message in raw_messages]

    if clean_messages != raw_messages:
        HISTORY_PATH.write_text(
            json.dumps(clean_messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return clean_messages


def save_history(messages: list[dict[str, Any]]) -> None:
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    clean_messages = [_clean_message(message) for message in messages]
    HISTORY_PATH.write_text(
        json.dumps(clean_messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(message)
    clean.pop("reasoning_content", None)

    if clean.get("role") == "assistant":
        tool_calls = clean.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                normalized_call = deepcopy(tool_call)
                normalized_call.setdefault("type", "function")

                function_payload = normalized_call.get("function")
                if not isinstance(function_payload, dict):
                    function_payload = {}

                normalized_call["function"] = {
                    "name": str(function_payload.get("name") or ""),
                    "arguments": str(function_payload.get("arguments") or "{}"),
                }
                normalized_tool_calls.append(normalized_call)

            clean["tool_calls"] = normalized_tool_calls

    return clean
