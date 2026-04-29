from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepSeekSettings(BaseSettings):
    api_key: str = Field(alias="DEEPSEEK_API_KEY")
    base_url: str = Field(
        default="https://api.deepseek.com/beta",
        alias="DEEPSEEK_BASE_URL",
    )
    model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SerpApiSettings(BaseSettings):
    api_key: str = Field(alias="SERPAPI_KEY")
    location: str = Field(
        default="Austin, Texas, United States",
        alias="SERPAPI_LOCATION",
    )
    hl: str = Field(default="en", alias="SERPAPI_HL")
    gl: str = Field(default="us", alias="SERPAPI_GL")
    google_domain: str = Field(default="google.com", alias="SERPAPI_GOOGLE_DOMAIN")
    timeout: int = Field(default=20, alias="SERPAPI_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
