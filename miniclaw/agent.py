from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .history import save_history
from .llm import DeepSeekClient, LLMError
from .tools import ToolRegistry


SYSTEM_PROMPT = """你是 MiniClaw，一个极简 OpenClaw 风格智能体。
你可以直接回答用户，也可以在需要读写本地 workspace 文件时调用工具。
调用 list_files 列出 workspace 根目录时，relative_dir 使用 "."。
需要执行 shell 时，只能调用 run_shell_command，且只用于只读命令。
需要分析 workspace 文件时，可以调用 delegate_file_analysis 委托分析子 Agent。
当你调用工具后，要根据工具返回的 Observation 继续思考并给出最终回复。
不要尝试访问 workspace 之外的路径；如果用户要求越界操作，请说明安全限制。
"""


@dataclass
class MiniClawAgent:
    client: DeepSeekClient
    tools: ToolRegistry
    max_steps: int = 6
    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}]
    )

    def run_turn(
        self,
        user_input: str,
        on_content_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        self._clear_reasoning_content()
        self.messages.append({"role": "user", "content": user_input})

        try:
            for step in range(1, self.max_steps + 1):
                try:
                    assistant_message = self.client.chat(
                        messages=self.messages,
                        tools=self.tools.schemas(),
                        tool_choice="auto",
                        on_content_delta=on_content_delta,
                        on_reasoning_delta=on_reasoning_delta,
                    )
                except LLMError as exc:
                    return f"模型调用失败：{exc}"

                self.messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls") or []

                if not tool_calls:
                    content = assistant_message.get("content")
                    if content:
                        if on_content_delta:
                            return ""
                        return content.strip()
                    return "模型没有返回可显示的内容，请稍后重试。"

                if on_trace:
                    on_trace(
                        {
                            "step": step,
                            "type": "thought",
                            "content": assistant_message.get("reasoning_content")
                            or "模型准备调用工具。",
                        }
                    )

                observations = []
                for call in tool_calls:
                    name = self._tool_name(call)
                    try:
                        arguments = self._tool_arguments(call)
                        observation = self.tools.run(name, arguments)
                    except ValueError as exc:
                        arguments = {"error": str(exc)}
                        observation = json.dumps(
                            {
                                "ok": False,
                                "tool": name,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )

                    if on_trace:
                        on_trace(
                            {
                                "step": step,
                                "type": "action",
                                "tool": name,
                                "arguments": arguments,
                            }
                        )

                    observations.append(observation)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": observation,
                        }
                    )
                    if on_trace:
                        on_trace(
                            {
                                "step": step,
                                "type": "observation",
                                "content": observation,
                            }
                        )

                if not on_trace:
                    print(f"\n[Step {step} Observation]")
                    for observation in observations:
                        print(observation)
                    print()

            return f"已达到最大推理轮数 {self.max_steps}，请把任务拆小一些再试。"
        finally:
            save_history(self.messages)

    def _tool_arguments(self, call: dict[str, Any]) -> dict[str, Any]:
        raw_arguments = call.get("function", {}).get("arguments") or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{exc}") from exc

        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON object。")

        return arguments

    @staticmethod
    def _tool_name(call: dict[str, Any]) -> str:
        return str(call.get("function", {}).get("name") or "unknown_tool")

    def _clear_reasoning_content(self) -> None:
        for message in self.messages:
            message.pop("reasoning_content", None)

    def reset_messages(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
