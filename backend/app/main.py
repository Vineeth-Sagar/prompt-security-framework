"""FastAPI entrypoint.

Route modules are mounted here as later phases add them
(backend/app/api/routes/). For now this exposes only a liveness endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.input import router as input_router
from app.api.routes.logs import router as logs_router
from app.api.routes.ws import router as ws_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Adaptive Context-Aware Multi-Layer Prompt Security & Output "
        "Governance Framework — API gateway."
    ),
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(input_router)
app.include_router(auth_router)
app.include_router(logs_router)
app.include_router(ws_router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness/readiness probe.

    Returns 200 with a static payload as long as the process is up. Does
    not (yet) check downstream dependencies (Redis/Postgres) — that check
    is added once those clients exist, so this stays a cheap process
    liveness probe rather than a full readiness check.
    """
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
