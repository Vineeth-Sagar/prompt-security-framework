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
    target_llm_provider: str = "gemini"  # "anthropic" or "gemini" — see llm_gateway/factory.py
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    # --- SWCSA tuning (swcsa phase) ---
    drift_threshold: float = 0.8
    window_size: int = 5
    # Tuned via backend/eval/tune_swcsa_weights.py's *floored* search
    # (every weight >= 0.05) against the labeled dataset — deliberately
    # not the raw unconstrained-best combination that script also
    # reports, which drove weight onto 1-2 signals and zeroed the rest
    # (overfits the small hand-written dataset and defeats the point of
    # having independent signals). See that script's module docstring.
    swcsa_weight_semantic: float = 0.05
    swcsa_weight_role_escalation: float = 0.35
    swcsa_weight_topic_entropy: float = 0.5
    swcsa_weight_window_escalation: float = 0.05
    swcsa_weight_drift_trend: float = 0.05

    # --- context buffer (context_buffer phase) ---
    session_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-parsing env on every call)."""
    return Settings()
