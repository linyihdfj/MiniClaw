from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


_RUNTIME_ENV_LOADED = False


def _load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _ensure_runtime_env_loaded() -> None:
    global _RUNTIME_ENV_LOADED
    if _RUNTIME_ENV_LOADED:
        return

    _RUNTIME_ENV_LOADED = True
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    seen: set[Path] = set()

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue

        seen.add(resolved)
        _load_dotenv_file(resolved)


class ConfigError(Exception):
    """Raised when required runtime configuration is missing."""


class LLMError(Exception):
    """Raised when the LLM API request fails or returns invalid data."""


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        _ensure_runtime_env_loaded()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ConfigError(
                "请先设置 DEEPSEEK_API_KEY。你可以在项目根目录 .env 中写入："
                "DEEPSEEK_API_KEY=你的 key"
            )

        return cls(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/beta"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        on_content_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}

        for chunk in response:
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_parts.append(reasoning_content)
                if on_reasoning_delta:
                    on_reasoning_delta(reasoning_content)

            if delta.content:
                content_parts.append(delta.content)
                if on_content_delta:
                    on_content_delta(delta.content)

            for tool_call_delta in delta.tool_calls or []:
                index = tool_call_delta.index
                tool_call = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )

                if tool_call_delta.id:
                    tool_call["id"] = tool_call_delta.id
                if tool_call_delta.function.name:
                    tool_call["function"]["name"] = tool_call_delta.function.name
                if tool_call_delta.function.arguments:
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                if on_tool_call_delta:
                    on_tool_call_delta(
                        {
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    )

        content = "".join(content_parts)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)

        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

        return message
