from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.usage import UsageLimits

from .history import save_history
from .llm import resolve_model
from .tools import ToolRegistry


SYSTEM_PROMPT = """你是 MiniClaw，一个极简 OpenClaw 风格智能体。
你可以直接回答用户，也可以在需要读写本地 workspace 文件时调用工具。
调用 list_files 列出 workspace 根目录时，relative_dir 使用 "."。
需要了解 workspace 结构时，可以调用 list_directory_tree；需要查找文件内容时，可以调用 search_files。
需要写文件时，可以调用 write_text_file；需要追加或局部替换时，可以调用 append_text_file 或 replace_text_in_file。
需要联网搜索公开信息时，可以调用 search_web，它会通过 SerpAPI 执行 Google 搜索；需要打开网页正文核验时，可以调用 fetch_web_page。
需要当前时间时，可以调用 get_current_time；需要精确数学计算时，可以调用 calculate_expression。
需要执行 shell 时，只能调用 run_shell_command，且只用于只读命令。
需要分析 workspace 文件时，可以调用 delegate_file_analysis 委托分析子 Agent。
当用户要求分析、总结、解释、审查、概括 workspace 内某个文件的内容、结构、作用、逻辑、页面组成或代码行为时，你应优先调用 delegate_file_analysis，而不是自己直接分析。
尤其是 HTML、Python、JavaScript、Markdown、TXT、配置文件等文件分析任务，默认先委托子 Agent，再基于它的结果给出最终回答。
当你调用工具后，要根据工具返回的 Observation 继续思考并给出最终回复。
不要尝试访问 workspace 之外的路径；如果用户要求越界操作，请说明安全限制。
"""


@dataclass
class MiniClawAgent:
    # 持有当前模型、工具注册表和消息历史，是主 Agent 的最小运行单元。
    model: str
    tools: ToolRegistry
    max_steps: int = 6
    messages: list[ModelMessage] | None = None

    def __post_init__(self) -> None:
        self.messages = list(self.messages or [])

    def run_turn(
        self,
        user_input: str,
        on_content_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_trace: Callable[[dict[str, Any]], None] | None = None,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        # 每一轮都临时创建事件桥接器，把模型/工具事件转换成 CLI 和 Web 能显示的格式。
        event_bridge = AgentEventBridge(
            on_content_delta=on_content_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_trace=on_trace,
            on_stream_event=on_stream_event,
            agent_role="main",
            agent_name="MiniClaw",
            stream_id="main",
        )
        agent = self._create_agent(event_bridge=event_bridge)

        try:
            result = agent.run_sync(
                user_input,
                message_history=self.messages or None,
                usage_limits=UsageLimits(request_limit=self.max_steps),
                event_stream_handler=event_bridge.handle_event_stream,
            )
            # result.all_messages() 由 pydantic-ai 维护，包含本轮新增的对话和工具消息。
            self.messages = list(result.all_messages())
            output = str(result.output or "").strip()
            final_output = output or "".join(event_bridge.streamed_content_parts).strip()
            if not final_output:
                final_output = "模型没有返回可显示的内容，请稍后重试。"
            event_bridge.emit_final_answer(final_output)
            if event_bridge.streamed_content_parts:
                return ""
            return final_output
        except Exception as exc:
            failure = f"模型调用失败：{exc}"
            event_bridge.emit_final_answer(failure)
            return failure
        finally:
            save_history(self.messages or [])

    def reset_messages(self) -> None:
        self.messages = []

    def _create_agent(self, event_bridge: "AgentEventBridge") -> PydanticAgent[None, str]:
        model_name, model_settings = resolve_model(self.model)
        # pydantic-ai 负责真正的 function calling 循环，我们这里只装配模型、提示词和工具。
        return PydanticAgent(
            model_name,
            output_type=str,
            system_prompt=SYSTEM_PROMPT,
            name="MiniClaw",
            model_settings=model_settings,
            tools=self.tools.as_pydantic_tools(on_trace=event_bridge.handle_tool_event),
            end_strategy="early",
        )


@dataclass
class AgentEventBridge:
    # 把底层事件流整理成“Thought / Action / Observation / Delegation”轨迹。
    on_content_delta: Callable[[str], None] | None
    on_reasoning_delta: Callable[[str], None] | None
    on_trace: Callable[[dict[str, Any]], None] | None
    on_stream_event: Callable[[dict[str, Any]], None] | None = None
    agent_role: str = "main"
    agent_name: str = "MiniClaw"
    stream_id: str = "main"
    step: int = 0
    _pending_step_start: bool = True
    _tool_index: int = 0
    _tool_index_by_id: dict[str, int] | None = None
    _current_thinking: list[str] | None = None
    _thought_emitted_for_step: bool = False
    streamed_content_parts: list[str] | None = None

    def __post_init__(self) -> None:
        self._tool_index_by_id = {}
        self._current_thinking = []
        self.streamed_content_parts = []

    async def handle_event_stream(self, _ctx: Any, events: Any) -> None:
        async for event in events:
            self._handle_event(event)

    def _handle_event(self, event: Any) -> None:
        # 只要模型开始新一轮思考、增量输出或产出最终结果，就为这一轮分配 step 编号。
        if isinstance(event, PartStartEvent | PartDeltaEvent | FinalResultEvent):
            self._start_step_if_needed()

        if isinstance(event, PartStartEvent):
            self._handle_part_start(event)
        elif isinstance(event, PartDeltaEvent):
            self._handle_part_delta(event)
        elif isinstance(event, FunctionToolCallEvent):
            self._emit_thought_if_needed()
            tool_index = self._index_for_tool_call(event.part.tool_call_id)
            arguments = self._tool_args_dict(event.part)
            self._trace(
                {
                    "step": self.step,
                    "type": "action",
                    "tool_index": tool_index,
                    "tool": event.part.tool_name,
                    "arguments": arguments,
                }
            )
            self._trace(
                {
                    "step": self.step,
                    "type": "observation_start",
                    "tool_index": tool_index,
                    "tool": event.part.tool_name,
                    "content": f"开始执行工具：{event.part.tool_name}",
                }
            )
        elif isinstance(event, FunctionToolResultEvent):
            tool_index = self._index_for_tool_call(event.tool_call_id)
            self._trace(
                {
                    "step": self.step,
                    "type": "observation",
                    "tool_index": tool_index,
                    "tool": getattr(event.result, "tool_name", "unknown_tool"),
                    "content": self._serialize_content(event.result.content),
                }
            )
            # 工具执行完成后，下一次模型再输出内容时应进入新的 step。
            self._pending_step_start = True

    def _handle_part_start(self, event: PartStartEvent) -> None:
        part = event.part
        if isinstance(part, ThinkingPart) and part.content:
            self._append_thinking(part.content)
        elif isinstance(part, TextPart) and part.content:
            self._emit_content_delta(part.content)
        elif isinstance(part, ToolCallPart):
            tool_index = self._index_for_tool_call(part.tool_call_id)
            self._trace(
                {
                    "step": self.step,
                    "type": "action_delta",
                    "tool_index": tool_index,
                    "tool": part.tool_name,
                    "arguments": self._tool_args_dict(part),
                }
            )

    def _handle_part_delta(self, event: PartDeltaEvent) -> None:
        delta = event.delta
        if isinstance(delta, ThinkingPartDelta) and delta.content_delta:
            self._append_thinking(delta.content_delta)
        elif isinstance(delta, TextPartDelta) and delta.content_delta:
            self._emit_content_delta(delta.content_delta)
        elif isinstance(delta, ToolCallPartDelta):
            tool_call_id = delta.tool_call_id or ""
            tool_index = self._index_for_tool_call(tool_call_id)
            self._trace(
                {
                    "step": self.step,
                    "type": "action_delta",
                    "tool_index": tool_index,
                    "tool": delta.tool_name_delta or "",
                    "arguments": self._partial_arguments(delta.args_delta),
                }
            )

    def _start_step_if_needed(self) -> None:
        if not self._pending_step_start:
            return
        self.step += 1
        self._pending_step_start = False
        self._tool_index = 0
        self._tool_index_by_id = {}
        self._current_thinking = []
        self._thought_emitted_for_step = False
        self._trace({"step": self.step, "type": "step_start"})

    def _index_for_tool_call(self, tool_call_id: str) -> int:
        if tool_call_id not in self._tool_index_by_id:
            self._tool_index_by_id[tool_call_id] = self._tool_index
            self._tool_index += 1
        return self._tool_index_by_id[tool_call_id]

    def _append_thinking(self, delta: str) -> None:
        if self.on_reasoning_delta:
            self.on_reasoning_delta(delta)
        self._current_thinking.append(delta)
        self._trace({"step": self.step, "type": "thought_delta", "content": delta})

    def _emit_content_delta(self, delta: str) -> None:
        self.streamed_content_parts.append(delta)
        if self.on_content_delta:
            self.on_content_delta(delta)
        if self.on_stream_event:
            self.on_stream_event(
                {
                    "event": "content_delta",
                    "delta": delta,
                    "agent_role": self.agent_role,
                    "agent_name": self.agent_name,
                    "stream_id": self.stream_id,
                }
            )

    def emit_final_answer(self, content: str) -> None:
        if not content or not self.on_stream_event:
            return
        self.on_stream_event(
            {
                "event": "final_answer",
                "content": content,
                "agent_role": self.agent_role,
                "agent_name": self.agent_name,
                "stream_id": self.stream_id,
            }
        )

    def _emit_thought_if_needed(self) -> None:
        if self._thought_emitted_for_step:
            return
        # 有些模型不会显式输出 reasoning，这里至少给 UI 一个可显示的 Thought 节点。
        content = "".join(self._current_thinking).strip() or "模型准备调用工具。"
        self._trace({"step": self.step, "type": "thought", "content": content})
        self._thought_emitted_for_step = True

    def _trace(self, payload: dict[str, Any]) -> None:
        if self.on_trace:
            enriched = {
                "agent_role": self.agent_role,
                "agent_name": self.agent_name,
                "stream_id": self.stream_id,
                **payload,
            }
            self.on_trace(enriched)

    def handle_tool_event(self, ctx: Any, tool_name: str, event: dict[str, Any]) -> None:
        tool_call_id = str(getattr(ctx, "tool_call_id", "") or "")
        fallback_tool_index = self._index_for_tool_call(tool_call_id)
        event_type = str(event.get("type") or "observation_delta")
        data = event.get("data") or {}
        agent_role = str(event.get("agent_role") or data.get("agent_role") or self.agent_role)
        agent_name = str(event.get("agent_name") or data.get("agent_name") or self.agent_name)
        stream_id = str(event.get("stream_id") or data.get("stream_id") or self.stream_id)
        step = self._coerce_int(event.get("step"), default=self.step)
        tool_index = self._coerce_int(event.get("tool_index"), default=fallback_tool_index)
        content = str(event.get("content") or "")
        if event_type == "content_delta" and self.on_stream_event:
            self.on_stream_event(
                {
                    "event": "content_delta",
                    "delta": content,
                    "agent_role": agent_role,
                    "agent_name": agent_name,
                    "stream_id": stream_id,
                }
            )
            return
        if event_type == "final_answer" and self.on_stream_event:
            self.on_stream_event(
                {
                    "event": "final_answer",
                    "content": content,
                    "agent_role": agent_role,
                    "agent_name": agent_name,
                    "stream_id": stream_id,
                }
            )
            return
        # 工具层可以额外上报 delegation 等高层事件，这里统一并入主时间线。
        payload: dict[str, Any] = {
            "step": step,
            "type": event_type,
            "tool_index": tool_index,
            "tool": str(event.get("tool") or tool_name),
            "content": content,
            "data": data,
            "agent_role": agent_role,
            "agent_name": agent_name,
            "stream_id": stream_id,
        }
        if "arguments" in event:
            payload["arguments"] = event.get("arguments") or {}
        self._trace(payload)

    @staticmethod
    def _partial_arguments(raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments in (None, ""):
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(str(raw_arguments))
        except json.JSONDecodeError:
            return {"_partial": str(raw_arguments)}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}

    @staticmethod
    def _tool_args_dict(part: ToolCallPart) -> dict[str, Any]:
        try:
            return part.args_as_dict()
        except Exception:
            # 增量工具参数在流式阶段可能还不是完整 JSON，这里尽量保留原始片段。
            return AgentEventBridge._partial_arguments(part.args_as_json_str())

    @staticmethod
    def _serialize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _coerce_int(raw: Any, default: int) -> int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default
