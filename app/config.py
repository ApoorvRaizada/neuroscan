"""Settings come from environment variables (or a local .env file)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str  # required, no default on purpose
    model_path: str = "weights/resnet18.pt"
    max_upload_bytes: int = 5 * 1024 * 1024
    min_dim: int = 64
    max_dim: int = 4096


@lru_cache
def get_settings() -> Settings:
    return Settings()