from __future__ import annotations

import json
from typing import Any

from .llm import DeepSeekClient, LLMError
from .tools import create_read_only_registry


ANALYSIS_SYSTEM_PROMPT = """你是 MiniClaw 的文件分析子 Agent。
你只负责分析 workspace 内文件。需要读取文件时只能调用只读工具。
分析完成后，用简洁中文返回结论。"""


class AnalysisSubAgent:
    def __init__(self, client: DeepSeekClient, max_steps: int = 4):
        self.client = client
        self.tools = create_read_only_registry()
        self.max_steps = max_steps

    def analyze(self, relative_path: str, task: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"请分析文件 {relative_path}。任务：{task}",
            },
        ]

        for _ in range(self.max_steps):
            try:
                assistant_message = self.client.chat(
                    messages=messages,
                    tools=self.tools.schemas(),
                    tool_choice="auto",
                )
            except LLMError as exc:
                return f"分析子 Agent 调用失败：{exc}"

            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []

            if not tool_calls:
                return str(assistant_message.get("content") or "").strip()

            for call in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": self._execute_tool_call(call),
                    }
                )

        return f"分析子 Agent 已达到最大轮数 {self.max_steps}。"

    def _execute_tool_call(self, call: dict[str, Any]) -> str:
        name = str(call.get("function", {}).get("name") or "unknown_tool")
        raw_arguments = call.get("function", {}).get("arguments") or "{}"
        arguments = json.loads(raw_arguments)
        return self.tools.run(name, arguments)
