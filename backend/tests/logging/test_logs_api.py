from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import get_session
from app.logging.models import DecisionLog
from app.main import app


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def client(engine):
    async def override_get_session():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.pop(get_session, None)


async def _seed_logs(engine, count: int = 5, action_cycle=("PASS", "BLOCK", "SAFE_REWRITE")):
    async with AsyncSession(engine) as session:
        for i in range(count):
            session.add(
                DecisionLog(
                    session_id=f"sess-{i % 2}",  # two distinct sessions
                    input_modality="text",
                    drift_breakdown={"aggregate": 0.1 * i},
                    ifsr_result={"blocked": False},
                    policy_action=action_cycle[i % len(action_cycle)],
                    matched_rule="rule_x",
                    pii_found=[],
                    latency_ms_per_stage={"total": 1.0},
                    created_at=datetime.now(UTC) - timedelta(hours=count - i),
                )
            )
        await session.commit()


async def _register_and_login(
    client: AsyncClient, email: str, password: str, role: str | None = None
):
    if role is None:
        # bootstrap admin
        await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    tokens = (
        await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    ).json()
    return tokens


def _auth_header(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


# --- pagination ---


@pytest.mark.asyncio
async def test_list_logs_returns_all_seeded_rows_within_default_limit(client: AsyncClient, engine):
    await _seed_logs(engine, count=5)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get("/api/v1/logs", headers=_auth_header(tokens["access_token"]))

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert len(body["items"]) == 5


@pytest.mark.asyncio
async def test_pagination_limit_and_offset(client: AsyncClient, engine):
    await _seed_logs(engine, count=5)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    headers = _auth_header(tokens["access_token"])

    page1 = (await client.get("/api/v1/logs?limit=2&offset=0", headers=headers)).json()
    page2 = (await client.get("/api/v1/logs?limit=2&offset=2", headers=headers)).json()

    assert page1["total"] == 5
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    ids_page1 = {item["id"] for item in page1["items"]}
    ids_page2 = {item["id"] for item in page2["items"]}
    assert ids_page1.isdisjoint(ids_page2)


@pytest.mark.asyncio
async def test_results_are_ordered_newest_first(client: AsyncClient, engine):
    await _seed_logs(engine, count=3)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get("/api/v1/logs", headers=_auth_header(tokens["access_token"]))

    timestamps = [item["created_at"] for item in response.json()["items"]]
    assert timestamps == sorted(timestamps, reverse=True)


# --- filtering ---


@pytest.mark.asyncio
async def test_filter_by_action(client: AsyncClient, engine):
    await _seed_logs(engine, count=6)  # 2 of each action, cycling PASS/BLOCK/SAFE_REWRITE
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get(
        "/api/v1/logs?action=BLOCK", headers=_auth_header(tokens["access_token"])
    )

    body = response.json()
    assert body["total"] == 2
    assert all(item["policy_action"] == "BLOCK" for item in body["items"])


@pytest.mark.asyncio
async def test_filter_by_session_id(client: AsyncClient, engine):
    await _seed_logs(engine, count=5)  # alternates sess-0/sess-1
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get(
        "/api/v1/logs?session_id=sess-0", headers=_auth_header(tokens["access_token"])
    )

    body = response.json()
    assert body["total"] == 3
    assert all(item["session_id"] == "sess-0" for item in body["items"])


@pytest.mark.asyncio
async def test_filter_by_date_range(client: AsyncClient, engine):
    await _seed_logs(engine, count=5)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    headers = _auth_header(tokens["access_token"])

    now = datetime.now(UTC)
    start = (now - timedelta(hours=2)).isoformat()

    # params=, not an f-string URL: the ISO timestamp's "+00:00" offset
    # needs percent-encoding (a raw "+" in a query string decodes as a
    # space), which httpx's params handles and manual interpolation doesn't.
    response = await client.get("/api/v1/logs", params={"start_date": start}, headers=headers)

    body = response.json()
    assert 0 < body["total"] < 5  # some but not all rows fall in the recent window


@pytest.mark.asyncio
async def test_combined_filters(client: AsyncClient, engine):
    await _seed_logs(engine, count=6)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get(
        "/api/v1/logs?session_id=sess-0&action=PASS",
        headers=_auth_header(tokens["access_token"]),
    )

    body = response.json()
    assert all(
        item["session_id"] == "sess-0" and item["policy_action"] == "PASS"
        for item in body["items"]
    )


# --- get by id ---


@pytest.mark.asyncio
async def test_get_log_by_id_returns_full_detail(client: AsyncClient, engine):
    await _seed_logs(engine, count=1)
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    headers = _auth_header(tokens["access_token"])

    listing = (await client.get("/api/v1/logs", headers=headers)).json()
    log_id = listing["items"][0]["id"]

    response = await client.get(f"/api/v1/logs/{log_id}", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == log_id
    assert "drift_breakdown" in response.json()


@pytest.mark.asyncio
async def test_get_nonexistent_log_returns_404(client: AsyncClient):
    tokens = await _register_and_login(client, "admin@example.com", "adminpass123")

    response = await client.get(
        "/api/v1/logs/999999", headers=_auth_header(tokens["access_token"])
    )

    assert response.status_code == 404


# --- auth/RBAC ---


@pytest.mark.asyncio
async def test_unauthenticated_request_is_rejected(client: AsyncClient):
    response = await client.get("/api/v1/logs")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_viewer_role_is_blocked_from_logs(client: AsyncClient):
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    await client.post(
        "/api/v1/auth/register",
        json={"email": "viewer@example.com", "password": "viewerpass123", "role": "viewer"},
        headers=_auth_header(admin_tokens["access_token"]),
    )
    viewer_tokens = await _register_and_login(
        client, "viewer@example.com", "viewerpass123", role="viewer"
    )

    response = await client.get(
        "/api/v1/logs", headers=_auth_header(viewer_tokens["access_token"])
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_analyst_role_can_read_logs(client: AsyncClient, engine):
    await _seed_logs(engine, count=1)
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    await client.post(
        "/api/v1/auth/register",
        json={"email": "analyst@example.com", "password": "analystpass123", "role": "analyst"},
        headers=_auth_header(admin_tokens["access_token"]),
    )
    analyst_tokens = await _register_and_login(
        client, "analyst@example.com", "analystpass123", role="analyst"
    )

    response = await client.get(
        "/api/v1/logs", headers=_auth_header(analyst_tokens["access_token"])
    )

    assert response.status_code == 200
