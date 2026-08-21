"""Per-session sliding-window conversation history, backed by Redis.

Key schema (per session):
    session:{session_id}:turns   Redis LIST, newest turn at the head
                                  (LPUSH), trimmed to WINDOW_SIZE entries.
    session:{session_id}:meta    Redis HASH: created_at (set once),
                                  last_seen (updated on every write).

Both keys get their TTL refreshed (EXPIRE) on every write, so an idle
session's history evicts itself automatically instead of accumulating
forever — SWCSA (Phase 4) only ever needs the last few turns, not a full
history.
"""

import redis.asyncio as redis

from app.config import get_settings


def _turns_key(session_id: str) -> str:
    return f"session:{session_id}:turns"


def _meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"


_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """Return a process-wide Redis client (lazy singleton).

    `redis.from_url` doesn't connect eagerly — building the client here
    doesn't require Redis to be reachable at import time, only when a
    command is actually issued.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client
