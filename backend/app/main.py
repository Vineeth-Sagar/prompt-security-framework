"""FastAPI entrypoint.

Route modules are mounted here as later phases add them
(backend/app/api/routes/). For now this exposes only a liveness endpoint.
"""

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Adaptive Context-Aware Multi-Layer Prompt Security & Output "
        "Governance Framework — API gateway."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness/readiness probe.

    Returns 200 with a static payload as long as the process is up. Does
    not (yet) check downstream dependencies (Redis/Postgres) — that check
    is added once those clients exist, so this stays a cheap process
    liveness probe rather than a full readiness check.
    """
    return {"status": "ok", "service": settings.app_name}
