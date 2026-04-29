from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.usage import UsageLimits

from .agent import AgentEventBridge
from .llm import resolve_model
from .tools import create_read_only_registry


ANALYSIS_SYSTEM_PROMPT = """你是 MiniClaw 的文件分析子 Agent。
你只负责分析 workspace 内文件。需要读取文件时只能调用只读工具。
分析完成后，用简洁中文返回结论。"""


@dataclass
class AnalysisSubAgent:
    model: str
    max_steps: int = 4

    def __post_init__(self) -> None:
        # 子 Agent 只拿只读工具，避免被委托的分析任务产生副作用。
        self.tools = create_read_only_registry()

    def analyze(
        self,
        relative_path: str,
        task: str,
        stream_id: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        event_bridge = AgentEventBridge(
            on_content_delta=None,
            on_reasoning_delta=None,
            on_trace=on_event,
            on_stream_event=on_event,
            agent_role="sub",
            agent_name="AnalysisSubAgent",
            stream_id=stream_id,
        )
        model_name, model_settings = resolve_model(self.model)
        agent = PydanticAgent(
            model_name,
            output_type=str,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            model_settings=model_settings,
            tools=self.tools.as_pydantic_tools(on_trace=event_bridge.handle_tool_event),
            end_strategy="early",
        )
        try:
            result = agent.run_sync(
                f"请分析文件 {relative_path}。任务：{task}",
                usage_limits=UsageLimits(request_limit=self.max_steps),
                event_stream_handler=event_bridge.handle_event_stream,
            )
        except Exception as exc:
            failure = f"分析子 Agent 调用失败：{exc}"
            event_bridge.emit_final_answer(failure)
            return failure
        analysis = str(result.output or "").strip()
        final_output = analysis or "".join(event_bridge.streamed_content_parts).strip()
        if not final_output:
            final_output = "分析子 Agent 没有返回可显示的内容。"
        event_bridge.emit_final_answer(final_output)
        return final_output
