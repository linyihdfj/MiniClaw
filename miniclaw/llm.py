from __future__ import annotations

import os
from copy import deepcopy

from pydantic import ValidationError

from .settings import DeepSeekSettings


class ConfigError(Exception):
    """Raised when required runtime configuration is missing or invalid."""


MODEL_SETTINGS = {
    "deepseek-v4-flash": {
        "extra_body": {
            "thinking": {
                "type": "disabled",
            }
        }
    },
    "deepseek-v4-pro": {
        "extra_body": {
            "thinking": {
                "type": "enabled",
            }
        },
        "openai_reasoning_effort": "high",
    },
}

MODELS = tuple(MODEL_SETTINGS)
MODEL_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-pro",
}


def normalize_model(model: str) -> str:
    model = MODEL_ALIASES.get(model, model)
    if model not in MODEL_SETTINGS:
        raise ConfigError(f"不支持的模型：{model}。可选：{', '.join(MODELS)}")
    return model


def load_default_model() -> str:
    try:
        settings = DeepSeekSettings()
    except ValidationError as exc:
        raise ConfigError(
            "请先设置 DEEPSEEK_API_KEY。你可以在项目根目录 .env 中写入："
            "DEEPSEEK_API_KEY=你的 key"
        ) from exc

    os.environ["DEEPSEEK_API_KEY"] = settings.api_key
    return normalize_model(settings.model)


def resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    model = normalize_model(model)
    return f"deepseek:{model}", deepcopy(MODEL_SETTINGS[model])
