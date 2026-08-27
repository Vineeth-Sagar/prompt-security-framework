"""FastAPI entrypoint.

Route modules are mounted here as later phases add them
(backend/app/api/routes/).

Middleware order matters here and is deliberate — see
`catch_unhandled_errors` below for why an error-catching middleware has
to sit inside the CORS layer rather than being registered as a normal
exception handler.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.auth import router as auth_router
from app.api.routes.input import router as input_router
from app.api.routes.logs import router as logs_router
from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.users import router as users_router
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

logger = logging.getLogger(__name__)


# Registered BEFORE CORSMiddleware, which means it ends up *inside* it:
# Starlette builds the stack so that the most recently added middleware
# is outermost, so CORS wraps this, and this wraps the router.
#
# That ordering is the entire point. Starlette's own
# ServerErrorMiddleware — the thing that turns an unhandled exception
# into a 500 — sits outside every user middleware, CORS included, so a
# 500 it produces has no `access-control-allow-origin` header. A browser
# will not expose such a response to JS at all: `fetch()` rejects with a
# TypeError that is indistinguishable from the server being down, and
# that is exactly how a real server-side error got reported on the
# Playground as "Could not reach the server." while /health was
# simultaneously returning 200 from the same page.
#
# Note a handler registered via `app.add_exception_handler(Exception,...)`
# does NOT work here: Starlette deliberately routes Exception/500
# handlers to ServerErrorMiddleware, i.e. back outside CORS. Catching in
# a middleware inside the CORS layer is what makes the response
# CORS-visible.
@app.middleware("http")
async def catch_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        # Logged with the full traceback server-side; the client gets a
        # generic message, since exception text can carry internals
        # (connection strings, file paths) that don't belong in an API
        # response.
        logger.exception("Unhandled error handling %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "The server hit an unexpected error handling this request. "
                    "It has been logged. Please try again."
                )
            },
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
app.include_router(pipeline_router)
app.include_router(users_router)
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
