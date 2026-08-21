"""Application settings, loaded from environment variables (.env in dev).

Fields below cover Phase 0 (app metadata, CORS) plus the connection/secret/
tuning values the near-term phases (context buffer, SWCSA, LLM gateway,
auth) will need, so later phases only have to *use* a setting instead of
adding it. Fields specific to a not-yet-built layer stay unused until that
phase lands — see `.env.example` for the authoritative list and defaults.
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

    # --- app metadata ---
    app_name: str = "prompt-security-framework"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    # --- CORS ---
    frontend_origin: str = "http://localhost:3000"

    # --- data layer (context_buffer / logging phases) ---
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://psf:psf@localhost:5432/psf"

    # --- auth (auth/RBAC phase) ---
    jwt_secret: str = "dev-secret-change-me"

    # --- target LLM (llm_gateway phase) ---
    anthropic_api_key: str = ""

    # --- SWCSA tuning (swcsa phase) ---
    drift_threshold: float = 0.8
    window_size: int = 5
    swcsa_weight_semantic: float = 0.4
    swcsa_weight_role_escalation: float = 0.4
    swcsa_weight_topic_entropy: float = 0.2

    # --- context buffer (context_buffer phase) ---
    session_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-parsing env on every call)."""
    return Settings()
