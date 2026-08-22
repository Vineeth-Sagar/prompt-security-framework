"""Persist every pipeline decision to Postgres, and broadcast a compact
event over Redis pub/sub for the live WebSocket feed (ws.py).

Runs regardless of outcome — BLOCK, SAFE_REWRITE, and PASS are all
logged the same way. "Why was this blocked" is exactly what the
explainability dashboard needs most for the BLOCK case, so there's no
special-casing here that would skip logging on any particular action.
"""

import json
from typing import TYPE_CHECKING

import redis.asyncio as redis
from sqlmodel.ext.asyncio.session import AsyncSession

from app.context_buffer.redis_buffer import get_redis_client
from app.logging.models import DecisionLog

if TYPE_CHECKING:
    # Deferred to type-checking only: app.pipeline imports log_decision
    # from this module, so importing PipelineResult back at runtime
    # would be circular. The string annotation below doesn't need this
    # import to actually resolve at call time.
    from app.pipeline import PipelineResult

LIVE_DECISIONS_CHANNEL = "live-decisions"


async def log_decision(
    result: "PipelineResult",
    session: AsyncSession,
    redis_client: redis.Redis | None = None,
) -> DecisionLog:
    """Write one DecisionLog row for `result` and publish a compact live-feed event.

    Args:
        redis_client: Injected client for the publish step, defaulting
            to the process-wide singleton — overridable (e.g.
            pipeline.py passes the same fakeredis instance the session's
            ContextBuffer already uses) so tests/eval scripts don't need
            a real Redis server.
    """
    log = DecisionLog(
        session_id=result.session_id,
        input_modality=result.input_result.modality,
        drift_breakdown=result.drift.model_dump(),
        ifsr_result=result.ifsr.model_dump(),
        policy_action=result.policy.action,
        matched_rule=result.policy.matched_rule,
        pii_found=[m.model_dump() for m in result.pii_scan.found] if result.pii_scan else [],
        latency_ms_per_stage={t.stage: t.duration_ms for t in result.stage_timings},
    )

    session.add(log)
    await session.commit()
    await session.refresh(log)

    await _publish_live_event(log, redis_client if redis_client is not None else get_redis_client())

    return log


async def _publish_live_event(log: DecisionLog, redis_client: redis.Redis) -> None:
    """Publish a compact JSON event for `log` — not the full row (drift
    breakdown/IFS-R result can be large; the live feed just needs enough
    to show an entry, a client fetches GET /api/v1/logs/{id} for detail)."""
    event = {
        "id": log.id,
        "session_id": log.session_id,
        "input_modality": log.input_modality,
        "policy_action": log.policy_action,
        "matched_rule": log.matched_rule,
        "created_at": log.created_at.isoformat(),
    }
    await redis_client.publish(LIVE_DECISIONS_CHANNEL, json.dumps(event))
