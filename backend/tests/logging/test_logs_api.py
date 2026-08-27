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


async def _seed_logs_for_user(engine, user_id: int | None, user_email: str | None, count: int = 1):
    """Seed `count` decision logs attributed to one user (or unattributed
    when user_id is None) — the fixture the per-user scoping tests need."""
    async with AsyncSession(engine) as session:
        for i in range(count):
            session.add(
                DecisionLog(
                    session_id=f"{user_email or 'anon'}-sess-{i}",
                    user_id=user_id,
                    user_email=user_email,
                    input_modality="text",
                    drift_breakdown={"aggregate": 0.1},
                    ifsr_result={"blocked": False},
                    policy_action="PASS",
                    matched_rule="rule_x",
                    pii_found=[],
                    latency_ms_per_stage={"total": 1.0},
                    created_at=datetime.now(UTC) - timedelta(minutes=i),
                )
            )
        await session.commit()


async def _register_role(client: AsyncClient, admin_tokens: dict, email: str, role: str) -> dict:
    """Admin-create a user with `role` and return the created row
    (including its `id`, needed to attribute seeded logs)."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": f"{role}pass123", "role": role},
        headers=_auth_header(admin_tokens["access_token"]),
    )
    assert response.status_code == 201
    return response.json()


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
        item["session_id"] == "sess-0" and item["policy_action"] == "PASS" for item in body["items"]
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

    response = await client.get("/api/v1/logs/999999", headers=_auth_header(tokens["access_token"]))

    assert response.status_code == 404


# --- auth/RBAC ---


@pytest.mark.asyncio
async def test_unauthenticated_request_is_rejected(client: AsyncClient):
    response = await client.get("/api/v1/logs")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_viewer_can_read_logs_but_only_their_own(client: AsyncClient, engine):
    # Policy change: viewers used to be blocked from /logs entirely. They
    # now get access, but strictly scoped to decisions they themselves
    # ran — so this asserts both halves: 200 (not 403), and that a
    # decision owned by someone else is invisible to them.
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    viewer = await _register_role(client, admin_tokens, "viewer@example.com", "viewer")
    await _seed_logs_for_user(engine, user_id=viewer["id"], user_email=viewer["email"], count=2)
    # A decision owned by a different user must not appear.
    await _seed_logs_for_user(engine, user_id=999, user_email="someone-else@example.com", count=3)

    viewer_tokens = await _register_and_login(
        client, "viewer@example.com", "viewerpass123", role="viewer"
    )
    response = await client.get("/api/v1/logs", headers=_auth_header(viewer_tokens["access_token"]))

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {row["user_email"] for row in body["items"]} == {"viewer@example.com"}


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


# --- per-user scoping (the cross-user leak this feature fixes) ---


@pytest.mark.asyncio
async def test_analyst_sees_only_their_own_decisions_not_other_users(client: AsyncClient, engine):
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    analyst_a = await _register_role(client, admin_tokens, "ana-a@example.com", "analyst")
    analyst_b = await _register_role(client, admin_tokens, "ana-b@example.com", "analyst")
    await _seed_logs_for_user(engine, analyst_a["id"], analyst_a["email"], count=2)
    await _seed_logs_for_user(engine, analyst_b["id"], analyst_b["email"], count=4)

    tokens = await _register_and_login(
        client, "ana-a@example.com", "analystpass123", role="analyst"
    )
    response = await client.get("/api/v1/logs", headers=_auth_header(tokens["access_token"]))

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {row["user_email"] for row in body["items"]} == {"ana-a@example.com"}


@pytest.mark.asyncio
async def test_non_admin_cannot_widen_scope_via_user_id_param(client: AsyncClient, engine):
    # The ownership filter is server-enforced: passing someone else's
    # user_id must not expose their rows.
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    analyst_a = await _register_role(client, admin_tokens, "ana-a@example.com", "analyst")
    analyst_b = await _register_role(client, admin_tokens, "ana-b@example.com", "analyst")
    await _seed_logs_for_user(engine, analyst_a["id"], analyst_a["email"], count=1)
    await _seed_logs_for_user(engine, analyst_b["id"], analyst_b["email"], count=3)

    tokens = await _register_and_login(
        client, "ana-a@example.com", "analystpass123", role="analyst"
    )
    response = await client.get(
        "/api/v1/logs",
        params={"user_id": analyst_b["id"]},
        headers=_auth_header(tokens["access_token"]),
    )

    assert response.status_code == 200
    body = response.json()
    # Still only analyst A's own row — the param was ignored, not honored.
    assert body["total"] == 1
    assert {row["user_email"] for row in body["items"]} == {"ana-a@example.com"}


@pytest.mark.asyncio
async def test_admin_sees_all_users_decisions(client: AsyncClient, engine):
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    analyst_a = await _register_role(client, admin_tokens, "ana-a@example.com", "analyst")
    analyst_b = await _register_role(client, admin_tokens, "ana-b@example.com", "analyst")
    await _seed_logs_for_user(engine, analyst_a["id"], analyst_a["email"], count=2)
    await _seed_logs_for_user(engine, analyst_b["id"], analyst_b["email"], count=3)
    await _seed_logs_for_user(engine, None, None, count=1)  # unattributed history

    response = await client.get("/api/v1/logs", headers=_auth_header(admin_tokens["access_token"]))

    assert response.status_code == 200
    assert response.json()["total"] == 6  # 2 + 3 + 1, everyone's


@pytest.mark.asyncio
async def test_admin_can_filter_by_user_id(client: AsyncClient, engine):
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    analyst_a = await _register_role(client, admin_tokens, "ana-a@example.com", "analyst")
    analyst_b = await _register_role(client, admin_tokens, "ana-b@example.com", "analyst")
    await _seed_logs_for_user(engine, analyst_a["id"], analyst_a["email"], count=2)
    await _seed_logs_for_user(engine, analyst_b["id"], analyst_b["email"], count=4)

    response = await client.get(
        "/api/v1/logs",
        params={"user_id": analyst_b["id"]},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert {row["user_email"] for row in body["items"]} == {"ana-b@example.com"}


@pytest.mark.asyncio
async def test_get_log_by_id_is_404_for_another_users_log(client: AsyncClient, engine):
    admin_tokens = await _register_and_login(client, "admin@example.com", "adminpass123")
    owner = await _register_role(client, admin_tokens, "owner@example.com", "analyst")
    await _register_role(client, admin_tokens, "other@example.com", "analyst")
    await _seed_logs_for_user(engine, owner["id"], owner["email"], count=1)

    # The single seeded log's id — fetched as its owner (allowed) to learn the id.
    owner_tokens = await _register_and_login(
        client, "owner@example.com", "analystpass123", role="analyst"
    )
    listed = await client.get("/api/v1/logs", headers=_auth_header(owner_tokens["access_token"]))
    log_id = listed.json()["items"][0]["id"]

    # Owner can read it.
    own = await client.get(
        f"/api/v1/logs/{log_id}", headers=_auth_header(owner_tokens["access_token"])
    )
    assert own.status_code == 200

    # A different non-admin gets 404 — existence isn't even confirmed.
    other_tokens = await _register_and_login(
        client, "other@example.com", "analystpass123", role="analyst"
    )
    forbidden = await client.get(
        f"/api/v1/logs/{log_id}", headers=_auth_header(other_tokens["access_token"])
    )
    assert forbidden.status_code == 404

    # Admin can read anyone's.
    admin_read = await client.get(
        f"/api/v1/logs/{log_id}", headers=_auth_header(admin_tokens["access_token"])
    )
    assert admin_read.status_code == 200
