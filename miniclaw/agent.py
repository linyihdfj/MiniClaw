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
需要了解 workspace 结构时，可以调用 list_directory_tree；需要查找文件内容时，可以调用 search_files。
需要写文件时，可以调用 write_text_file；需要追加或局部替换时，可以调用 append_text_file 或 replace_text_in_file。
需要联网搜索公开信息时，可以调用 search_web，它会综合 Bing、Baidu、Google；需要打开网页正文核验时，可以调用 fetch_web_page。
需要当前时间时，可以调用 get_current_time；需要精确数学计算时，可以调用 calculate_expression。
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
                streamed_content_parts: list[str] = []

                if on_trace:
                    on_trace(
                        {
                            "step": step,
                            "type": "step_start",
                        }
                    )

                def _on_content_delta(delta: str) -> None:
                    streamed_content_parts.append(delta)
                    if on_content_delta:
                        on_content_delta(delta)

                def _on_reasoning_delta(delta: str) -> None:
                    if on_reasoning_delta:
                        on_reasoning_delta(delta)
                    if on_trace:
                        on_trace(
                            {
                                "step": step,
                                "type": "thought_delta",
                                "content": delta,
                            }
                        )

                def _on_tool_call_delta(tool_call: dict[str, Any]) -> None:
                    if not on_trace:
                        return

                    raw_index = tool_call.get("index", 0)
                    try:
                        tool_index = int(raw_index)
                    except (TypeError, ValueError):
                        tool_index = 0

                    function = tool_call.get("function") or {}
                    on_trace(
                        {
                            "step": step,
                            "type": "action_delta",
                            "tool_index": tool_index,
                            "tool": str(function.get("name") or "unknown_tool"),
                            "arguments": self._trace_tool_arguments(
                                str(function.get("arguments") or "")
                            ),
                        }
                    )

                try:
                    assistant_message = self.client.chat(
                        messages=self.messages,
                        tools=self.tools.schemas(),
                        tool_choice="auto",
                        on_content_delta=_on_content_delta if on_content_delta else None,
                        on_reasoning_delta=_on_reasoning_delta
                        if on_reasoning_delta or on_trace
                        else None,
                        on_tool_call_delta=_on_tool_call_delta if on_trace else None,
                    )
                except LLMError as exc:
                    return f"模型调用失败：{exc}"

                self.messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls") or []

                if not tool_calls:
                    content = assistant_message.get("content")
                    if content:
                        if on_content_delta:
                            streamed_content = "".join(streamed_content_parts)
                            if streamed_content == content:
                                return ""
                            if streamed_content and content.startswith(streamed_content):
                                return content[len(streamed_content) :]
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
                for call_index, call in enumerate(tool_calls):
                    name = self._tool_name(call)

                    def _on_tool_event(event: dict[str, Any]) -> None:
                        if not on_trace:
                            return

                        on_trace(
                            {
                                "step": step,
                                "type": str(event.get("type") or "observation_delta"),
                                "tool_index": call_index,
                                "tool": name,
                                "content": str(event.get("content") or ""),
                                "data": event.get("data") or {},
                            }
                        )

                    try:
                        arguments = self._tool_arguments(call)
                        if on_trace:
                            on_trace(
                                {
                                    "step": step,
                                    "type": "action",
                                    "tool_index": call_index,
                                    "tool": name,
                                    "arguments": arguments,
                                }
                            )
                            on_trace(
                                {
                                    "step": step,
                                    "type": "observation_start",
                                    "tool_index": call_index,
                                    "tool": name,
                                    "content": f"开始执行工具：{name}",
                                }
                            )
                        observation = self.tools.run(
                            name,
                            arguments,
                            on_event=_on_tool_event if on_trace else None,
                        )
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
                                "tool_index": call_index,
                                "tool": name,
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

    @staticmethod
    def _trace_tool_arguments(raw_arguments: str) -> dict[str, Any]:
        if not raw_arguments:
            return {}

        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"_partial": raw_arguments}

        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}

    def _clear_reasoning_content(self) -> None:
        for message in self.messages:
            message.pop("reasoning_content", None)

    def reset_messages(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
