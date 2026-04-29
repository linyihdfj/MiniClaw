from __future__ import annotations

from typing import Any, Callable

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.usage import UsageLimits

from .llm import DeepSeekClient
from .tools import create_read_only_registry


ANALYSIS_SYSTEM_PROMPT = """你是 MiniClaw 的文件分析子 Agent。
你只负责分析 workspace 内文件。需要读取文件时只能调用只读工具。
分析完成后，用简洁中文返回结论。"""


class AnalysisSubAgent:
    def __init__(self, client: DeepSeekClient, max_steps: int = 4):
        self.client = client
        # 子 Agent 只拿只读工具，避免被委托的分析任务产生副作用。
        self.tools = create_read_only_registry()
        self.max_steps = max_steps

    def analyze(
        self,
        relative_path: str,
        task: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        # 子 Agent 不展开完整内部轨迹，只向外上报“开始 / 结束 / 失败”三个节点。
        if on_event:
            on_event(
                {
                    "type": "delegation_progress",
                    "content": f"子 Agent 已启动，开始分析文件：{relative_path}",
                    "data": {
                        "relative_path": relative_path,
                        "task": task,
                        "status": "subagent_started",
                        "agent_role": "sub",
                        "agent_name": "AnalysisSubAgent",
                    },
                }
            )
        agent = PydanticAgent(
            self.client.create_model(),
            output_type=str,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            tools=self.tools.as_pydantic_tools(),
            end_strategy="early",
        )
        try:
            result = agent.run_sync(
                f"请分析文件 {relative_path}。任务：{task}",
                usage_limits=UsageLimits(request_limit=self.max_steps),
            )
        except Exception as exc:
            failure = f"分析子 Agent 调用失败：{exc}"
            if on_event:
                on_event(
                    {
                        "type": "delegation_result",
                        "content": failure,
                        "data": {
                            "relative_path": relative_path,
                            "task": task,
                            "status": "subagent_failed",
                            "agent_role": "sub",
                            "agent_name": "AnalysisSubAgent",
                        },
                    }
                )
            return failure
        analysis = str(result.output or "").strip()
        if on_event:
            on_event(
                {
                    "type": "delegation_result",
                    "content": "子 Agent 已完成分析。",
                    "data": {
                        "relative_path": relative_path,
                        "task": task,
                        "status": "subagent_completed",
                        "agent_role": "sub",
                        "agent_name": "AnalysisSubAgent",
                        "analysis": analysis,
                    },
                }
            )
        return analysis
