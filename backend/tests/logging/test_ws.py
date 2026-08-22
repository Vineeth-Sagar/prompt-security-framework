"""WebSocket /ws/live-decisions tests.

Uses Starlette's sync TestClient (not httpx.AsyncClient — websocket
support needs the sync test client's real event-loop-driving
`websocket_connect` context manager), same tool FastAPI's own docs
recommend for testing WebSocket routes.
"""

import asyncio
import json

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.websockets import WebSocketDisconnect

from app.context_buffer.redis_buffer import get_redis_client
from app.db import get_session
from app.logging.decision_logger import LIVE_DECISIONS_CHANNEL
from app.main import app


@pytest.fixture
def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def _create_tables():
        async with eng.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_create_tables())
    yield eng
    loop.run_until_complete(eng.dispose())
    loop.close()


@pytest.fixture
def fake_redis():
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    loop = asyncio.new_event_loop()
    loop.run_until_complete(client.aclose())
    loop.close()


@pytest.fixture
def client(engine, fake_redis):
    async def override_get_session():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.pop(get_session, None)
    app.dependency_overrides.pop(get_redis_client, None)


def _register_and_login(client: TestClient, email: str, password: str) -> dict:
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    response = client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    )
    return response.json()


def test_connection_without_token_is_rejected(client: TestClient):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live-decisions"):
            pass


def test_connection_with_invalid_token_is_rejected(client: TestClient):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live-decisions?token=not-a-real-token"):
            pass


def test_connection_with_valid_admin_token_receives_broadcast_event(
    client: TestClient, fake_redis: FakeAsyncRedis
):
    tokens = _register_and_login(client, "admin@example.com", "adminpass123")

    with client.websocket_connect(
        f"/ws/live-decisions?token={tokens['access_token']}"
    ) as websocket:
        event = {"id": 1, "session_id": "sess-1", "policy_action": "PASS"}

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            fake_redis.publish(LIVE_DECISIONS_CHANNEL, json.dumps(event))
        )
        loop.close()

        received = websocket.receive_text()
        assert json.loads(received) == event


def test_viewer_role_is_rejected_from_the_live_feed(client: TestClient):
    admin_tokens = _register_and_login(client, "admin@example.com", "adminpass123")
    client.post(
        "/api/v1/auth/register",
        json={"email": "viewer@example.com", "password": "viewerpass123", "role": "viewer"},
        headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
    )
    viewer_tokens = _register_and_login(client, "viewer@example.com", "viewerpass123")

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/ws/live-decisions?token={viewer_tokens['access_token']}"
        ):
            pass
