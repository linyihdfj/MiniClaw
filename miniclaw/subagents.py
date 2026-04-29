from __future__ import annotations

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
        self.tools = create_read_only_registry()
        self.max_steps = max_steps

    def analyze(self, relative_path: str, task: str) -> str:
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
            return f"分析子 Agent 调用失败：{exc}"
        return str(result.output or "").strip()
