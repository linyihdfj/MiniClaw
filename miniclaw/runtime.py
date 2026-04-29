from __future__ import annotations

import threading
from dataclasses import dataclass

from .agent import MiniClawAgent
from .history import clear_history, load_history
from .llm import ConfigError, DeepSeekClient
from .tools import create_default_registry


MODELS = ("deepseek-chat", "deepseek-reasoner")


class RuntimeError(Exception):
    """Raised when the shared MiniClaw runtime cannot be initialized or used."""


@dataclass
class RuntimeState:
    client: DeepSeekClient
    agent: MiniClawAgent


class RuntimeManager:
    def __init__(self) -> None:
        self._state: RuntimeState | None = None
        self._boot_error: str | None = None
        self._state_lock = threading.Lock()
        self.turn_lock = threading.Lock()

    def get_state(self) -> RuntimeState:
        with self._state_lock:
            return self._ensure_state_locked()

    def get_model_info(self) -> dict[str, object]:
        state = self.get_state()
        return {
            "current": state.client.model,
            "models": list(MODELS),
        }

    def set_model(self, model: str) -> str:
        if model not in MODELS:
            raise RuntimeError(f"不支持的模型：{model}。可选：{', '.join(MODELS)}")

        with self.turn_lock:
            state = self.get_state()
            state.client.model = model
            return state.client.model

    def reset(self) -> None:
        with self.turn_lock:
            state = self.get_state()
            state.agent.reset_messages()
            clear_history()

    def _ensure_state_locked(self) -> RuntimeState:
        if self._boot_error:
            raise RuntimeError(self._boot_error)

        if self._state is not None:
            return self._state

        try:
            client = DeepSeekClient.from_env()
        except ConfigError as exc:
            self._boot_error = str(exc)
            raise RuntimeError(self._boot_error) from exc

        history = load_history()
        agent = MiniClawAgent(
            client=client,
            tools=create_default_registry(client=client),
            messages=history or None,
        )
        self._state = RuntimeState(client=client, agent=agent)
        return self._state


runtime = RuntimeManager()
