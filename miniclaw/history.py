from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "workspace" / "history.json"


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


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
    clean = dict(message)
    clean.pop("reasoning_content", None)
    return clean
