from __future__ import annotations

import threading

from .agent import MiniClawAgent
from .history import clear_history, load_history
from .llm import ConfigError, MODELS, load_default_model, normalize_model
from .tools import create_default_registry


class RuntimeError(Exception):
    """Raised when the shared MiniClaw runtime cannot be initialized or used."""


class RuntimeManager:
    def __init__(self) -> None:
        self._agent: MiniClawAgent | None = None
        self._state_lock = threading.Lock()
        # turn_lock 用来串行化每一轮对话，避免并发修改 messages/history。
        self.turn_lock = threading.Lock()

    def get_agent(self) -> MiniClawAgent:
        with self._state_lock:
            return self._ensure_agent_locked()

    def get_model_info(self) -> dict[str, object]:
        agent = self.get_agent()
        return {
            "current": agent.model,
            "models": list(MODELS),
        }

    def set_model(self, model: str) -> str:
        try:
            model = normalize_model(model)
        except ConfigError as exc:
            raise RuntimeError(str(exc)) from exc

        with self.turn_lock:
            agent = self.get_agent()
            agent.model = model
            return agent.model

    def reset(self) -> None:
        with self.turn_lock:
            self.get_agent().reset_messages()
            clear_history()

    def _ensure_agent_locked(self) -> MiniClawAgent:
        if self._agent is not None:
            return self._agent

        try:
            model = load_default_model()
        except ConfigError as exc:
            raise RuntimeError(str(exc)) from exc

        history = load_history()
        agent = MiniClawAgent(
            model=model,
            tools=create_default_registry(get_model=lambda: agent.model),
            messages=history or None,
        )
        self._agent = agent
        return self._agent


# 模块级单例，整个进程只维护一份共享 runtime。
runtime = RuntimeManager()
