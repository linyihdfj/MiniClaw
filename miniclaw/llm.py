from __future__ import annotations

import httpx
from pydantic import ValidationError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .settings import DeepSeekSettings


class ConfigError(Exception):
    """Raised when required runtime configuration is missing."""


class LLMError(Exception):
    """Raised when the LLM API request fails or returns invalid data."""


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        # provider 与 http_client 会被主 Agent 和子 Agent 复用。
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        )
        self.provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            http_client=self.http_client,
        )

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        try:
            settings = DeepSeekSettings()
        except ValidationError as exc:
            raise ConfigError(
                "请先设置 DEEPSEEK_API_KEY。你可以在项目根目录 .env 中写入："
                "DEEPSEEK_API_KEY=你的 key"
            ) from exc

        return cls(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
        )

    def create_model(self) -> OpenAIChatModel:
        # pydantic-ai 通过 OpenAI-compatible provider 调用 DeepSeek。
        return OpenAIChatModel(self.model, provider=self.provider)
