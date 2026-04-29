from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepSeekSettings(BaseSettings):
    # BaseSettings 会自动从 .env 和环境变量加载字段。
    api_key: str = Field(alias="DEEPSEEK_API_KEY")
    model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SerpApiSettings(BaseSettings):
    # 联网搜索相关配置单独拆开，避免没有 SERPAPI_KEY 时影响主模型启动。
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
