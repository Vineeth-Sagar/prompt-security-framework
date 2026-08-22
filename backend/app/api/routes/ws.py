"""WebSocket /ws/live-decisions — the live feed for the admin dashboard.

Every DecisionLog write publishes a compact event to a Redis channel
(decision_logger.py); this endpoint subscribes to that same channel and
forwards each message to connected sockets. Using Redis pub/sub rather
than an in-process list of open sockets means this works correctly with
multiple backend worker processes — a decision logged by worker A
reaches a socket connected to worker B.

Browsers can't set a custom Authorization header on a WebSocket
handshake, so auth comes via a `?token=` query param instead — the
standard workaround for browser WS auth. Only admin/analyst tokens are
accepted, same roles as the logs query endpoints.
"""

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User, UserRole
from app.auth.security import decode_token
from app.context_buffer.redis_buffer import get_redis_client
from app.db import get_session
from app.logging.decision_logger import LIVE_DECISIONS_CHANNEL

router = APIRouter()

_ALLOWED_ROLES = (UserRole.admin, UserRole.analyst)


async def _authenticate(websocket: WebSocket, session: AsyncSession) -> User | None:
    """Resolve the `token` query param to a User, or None if it's
    missing/invalid/wrong-role — never raises, since a WebSocket route
    closes the connection on rejection rather than returning an HTTP
    error response."""
    token = websocket.query_params.get("token")
    if not token:
        return None

    try:
        payload = decode_token(token)
    except HTTPException:
        return None

    if payload.get("type") != "access":
        return None

    email = payload.get("sub")
    if not email:
        return None

    result = await session.exec(select(User).where(User.email == email))
    user = result.first()
    if user is None or user.role not in _ALLOWED_ROLES:
        return None

    return user


@router.websocket("/ws/live-decisions")
async def live_decisions_ws(
    websocket: WebSocket,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    redis_client: redis.Redis = Depends(get_redis_client),  # noqa: B008
) -> None:
    user = await _authenticate(websocket, session)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(LIVE_DECISIONS_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                await websocket.send_text(message["data"])
            except Exception:
                # Client disconnected (or the socket otherwise broke) —
                # stop forwarding rather than let the exception propagate
                # out of the pubsub loop uncaught.
                break
    finally:
        await pubsub.unsubscribe(LIVE_DECISIONS_CHANNEL)
        await pubsub.aclose()
