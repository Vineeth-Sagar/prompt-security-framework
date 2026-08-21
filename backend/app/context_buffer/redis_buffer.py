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

from datetime import UTC, datetime

import redis.asyncio as redis
from pydantic import BaseModel, Field

from app.config import get_settings


def _turns_key(session_id: str) -> str:
    return f"session:{session_id}:turns"


def _meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"


class TurnRecord(BaseModel):
    """One conversational turn, as stored in a session's sliding window."""

    text: str
    role: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    drift_score: float | None = None


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


class ContextBuffer:
    """CRUD over a session's sliding window of turns."""

    def __init__(
        self,
        client: redis.Redis,
        window_size: int | None = None,
        ttl_seconds: int | None = None,
    ):
        settings = get_settings()
        self._client = client
        self._window_size = window_size if window_size is not None else settings.window_size
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.session_ttl_seconds

    @property
    def client(self) -> redis.Redis:
        """The underlying Redis client — exposed for callers (e.g. tests)
        that need to inspect state ContextBuffer's own API doesn't cover."""
        return self._client

    async def add_turn(self, session_id: str, turn: TurnRecord) -> None:
        """Push `turn` onto the session's window, trim it to size, refresh TTLs.

        All of push + trim + TTL-refresh + meta update run inside one
        Redis transaction (MULTI/EXEC), so a set of concurrent
        `add_turn` calls against the same session can't interleave into
        a half-applied state — each call's writes land atomically.
        """
        turns_key = _turns_key(session_id)
        meta_key = _meta_key(session_id)
        now = datetime.now(UTC).isoformat()

        async with self._client.pipeline(transaction=True) as pipe:
            pipe.lpush(turns_key, turn.model_dump_json())
            pipe.ltrim(turns_key, 0, self._window_size - 1)
            pipe.expire(turns_key, self._ttl_seconds)
            pipe.hsetnx(meta_key, "created_at", now)
            pipe.hset(meta_key, "last_seen", now)
            pipe.expire(meta_key, self._ttl_seconds)
            await pipe.execute()

    async def get_window(self, session_id: str) -> list[TurnRecord]:
        """Return the session's turns, oldest first.

        Returns an empty list for a session that doesn't exist (or has
        expired) rather than raising — an empty window is a normal,
        expected state (a brand-new session), not an error.
        """
        raw = await self._client.lrange(_turns_key(session_id), 0, -1)
        # LPUSH puts the newest turn at index 0; reverse to chronological order.
        return [TurnRecord.model_validate_json(item) for item in reversed(raw)]

    async def clear_session(self, session_id: str) -> None:
        """Delete both of a session's keys."""
        await self._client.delete(_turns_key(session_id), _meta_key(session_id))


_buffer: ContextBuffer | None = None


def get_context_buffer() -> ContextBuffer:
    """Return a process-wide ContextBuffer (lazy singleton).

    A plain function rather than `@lru_cache` so it stays a valid,
    overridable FastAPI dependency (`app.dependency_overrides`) — tests
    swap in a ContextBuffer backed by a fake Redis client this way,
    without touching the route code.
    """
    global _buffer
    if _buffer is None:
        _buffer = ContextBuffer(get_redis_client())
    return _buffer
