"""POST /api/v1/pipeline/run — the actual security pipeline, wired to HTTP.

`app/pipeline.py`'s `run_pipeline()` runs the full input -> preprocessing
-> SWCSA -> IFS-R -> policy -> target LLM -> output governance ->
decision-logging chain, but until this route existed nothing over HTTP
could reach it — `POST /api/v1/input` (api/routes/input.py) only runs
the first two stages. This route is the Playground's entry point: any
authenticated user (any role — a viewer testing prompts doesn't need
log-reading privileges) submits one modality's worth of input and gets
back the full `PipelineResult`, the same shape a caller of
`run_pipeline()` in Python would get.

Target-LLM failures (rate limit, timeout) are mapped to 503/504 with
an actionable message rather than being allowed to propagate as a bare
500 — see the `except` clauses in `run()` for why that distinction
matters to a Playground user.

`session_id` is optional — a one-off Playground submission with no
session doesn't need multi-turn drift context, so an ephemeral id is
generated when the caller omits one. Passing a real session_id (e.g.
carried across submissions client-side) lets SWCSA/IFS-R see the
conversation's actual history, same as api/routes/input.py's session_id
handling.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User
from app.auth.security import get_current_user
from app.context_buffer.redis_buffer import ContextBuffer, get_context_buffer
from app.db import get_session
from app.input_layer.router import resolve_modality
from app.llm_gateway.base import (
    BaseLLMAdapter,
    LLMRateLimitExceededError,
    LLMTimeoutError,
)
from app.llm_gateway.factory import get_llm_adapter
from app.pipeline import PipelineResult, run_pipeline

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineResult)
async def run(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI's DI idiom, not a real default-call bug
    modality: str | None = Form(None),  # noqa: B008
    session_id: str | None = Form(None),  # noqa: B008
    buffer: ContextBuffer = Depends(get_context_buffer),  # noqa: B008
    llm_adapter: BaseLLMAdapter = Depends(get_llm_adapter),  # noqa: B008
    db_session: AsyncSession = Depends(get_session),  # noqa: B008
    _user: User = Depends(get_current_user),  # noqa: B008
) -> PipelineResult:
    """Run one upload through the full pipeline and return the PipelineResult.

    Requires any authenticated user — no role restriction, unlike
    `GET /api/v1/logs` (a viewer can use the Playground; they just can't
    read the audit trail of everyone else's decisions).
    """
    raw = await file.read()

    try:
        resolved_modality = resolve_modality(modality, file.content_type, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if resolved_modality == "text":
        try:
            content: str | bytes = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Text input must be UTF-8 encoded."
            ) from exc
    else:
        content = raw

    effective_session_id = session_id or f"anon-{uuid.uuid4()}"

    try:
        result = await run_pipeline(
            effective_session_id,
            content,
            modality=resolved_modality,
            content_type=file.content_type,
            filename=file.filename,
            buffer=buffer,
            llm_adapter=llm_adapter,
            db_session=db_session,
        )
    except ValueError as exc:
        # Handlers raise ValueError for application-level validation
        # failures (empty/oversized text, ...) — same mapping
        # api/routes/input.py uses for the same handlers. Image/audio
        # handlers may instead raise HTTPException directly for
        # undecodable bytes — that propagates through FastAPI unchanged.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMRateLimitExceededError as exc:
        # The target LLM is a third-party dependency, so its transient
        # failures are *upstream* failures, not bugs in this service —
        # they deserve a 5xx that says which, plus a message a
        # Playground user can act on. Previously these propagated
        # uncaught and FastAPI returned a bare 500 "Internal Server
        # Error", which reads to a user as "the app is broken" rather
        # than "the provider throttled us, wait a moment".
        raise HTTPException(
            status_code=503,
            detail=(
                "The target LLM provider rate limit was exceeded (all retries "
                "exhausted). The prompt was analysed but not answered — wait a "
                "few seconds and submit again."
            ),
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "The target LLM timed out before responding. The prompt was "
                "analysed but not answered — please try again."
            ),
        ) from exc

    return result
