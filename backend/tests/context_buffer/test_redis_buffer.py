import asyncio

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.context_buffer.redis_buffer import ContextBuffer, TurnRecord, _meta_key, _turns_key

WINDOW_SIZE = 3
TTL_SECONDS = 100


@pytest_asyncio.fixture
async def redis_client():
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def buffer(redis_client) -> ContextBuffer:
    return ContextBuffer(redis_client, window_size=WINDOW_SIZE, ttl_seconds=TTL_SECONDS)


@pytest.mark.asyncio
async def test_window_never_exceeds_window_size(buffer: ContextBuffer):
    for i in range(WINDOW_SIZE + 4):
        await buffer.add_turn("s1", TurnRecord(text=f"turn-{i}", role="user"))

    window = await buffer.get_window("s1")

    assert len(window) == WINDOW_SIZE


@pytest.mark.asyncio
async def test_window_keeps_the_most_recent_turns_in_chronological_order(buffer: ContextBuffer):
    for i in range(WINDOW_SIZE + 2):
        await buffer.add_turn("s1", TurnRecord(text=f"turn-{i}", role="user"))

    window = await buffer.get_window("s1")

    # Oldest surviving turn first, newest last.
    assert [t.text for t in window] == ["turn-2", "turn-3", "turn-4"]


@pytest.mark.asyncio
async def test_ttl_is_refreshed_on_write(buffer: ContextBuffer, redis_client):
    await buffer.add_turn("s1", TurnRecord(text="hello", role="user"))

    ttl_turns = await redis_client.ttl(_turns_key("s1"))
    ttl_meta = await redis_client.ttl(_meta_key("s1"))

    assert ttl_turns == TTL_SECONDS
    assert ttl_meta == TTL_SECONDS


@pytest.mark.asyncio
async def test_meta_created_at_is_set_once_last_seen_updates(buffer: ContextBuffer, redis_client):
    await buffer.add_turn("s1", TurnRecord(text="first", role="user"))
    created_at_1 = await redis_client.hget(_meta_key("s1"), "created_at")

    await buffer.add_turn("s1", TurnRecord(text="second", role="user"))
    created_at_2 = await redis_client.hget(_meta_key("s1"), "created_at")
    last_seen = await redis_client.hget(_meta_key("s1"), "last_seen")

    assert created_at_1 == created_at_2  # set once, never overwritten
    assert last_seen is not None


@pytest.mark.asyncio
async def test_get_window_on_nonexistent_session_returns_empty_list(buffer: ContextBuffer):
    window = await buffer.get_window("never-existed")

    assert window == []


@pytest.mark.asyncio
async def test_clear_session_removes_all_keys(buffer: ContextBuffer, redis_client):
    await buffer.add_turn("s1", TurnRecord(text="hello", role="user"))

    await buffer.clear_session("s1")

    assert await redis_client.exists(_turns_key("s1")) == 0
    assert await redis_client.exists(_meta_key("s1")) == 0
    assert await buffer.get_window("s1") == []


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_corrupt_the_window(buffer: ContextBuffer):
    # N concurrent add_turn calls against the same session — each one's
    # push+trim+expire runs in its own MULTI/EXEC transaction, so the
    # final window should be well-formed (bounded, valid JSON turns)
    # regardless of interleaving order.
    n = 20
    await asyncio.gather(
        *(buffer.add_turn("s1", TurnRecord(text=f"turn-{i}", role="user")) for i in range(n))
    )

    window = await buffer.get_window("s1")

    assert len(window) == WINDOW_SIZE
    texts = [t.text for t in window]
    assert len(set(texts)) == WINDOW_SIZE  # no duplicated/corrupted entries
    assert all(t.startswith("turn-") for t in texts)
