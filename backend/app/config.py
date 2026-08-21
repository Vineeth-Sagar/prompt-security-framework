"""Application settings, loaded from environment variables (.env in dev).

Only the fields needed by the current phase are defined here; later phases
add settings for Redis, Postgres, JWT, the LLM adapter, etc. as those
modules land, instead of speculatively declaring them all now.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration.

    Values are read from environment variables (case-insensitive) or a
    `.env` file in the backend/ directory. See `.env.example` for the full
    list of variables a deployment may need to set.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "prompt-security-framework"
    environment: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-parsing env on every call)."""
    return Settings()
