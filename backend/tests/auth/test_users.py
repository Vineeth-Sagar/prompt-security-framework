import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import get_session
from app.main import app


@pytest_asyncio.fixture
async def client():
    """Same in-memory-SQLite-backed client as tests/auth/test_auth.py —
    duplicated rather than shared via a conftest.py, matching this
    repo's existing per-file fixture convention."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async def override_get_session():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.pop(get_session, None)
    await engine.dispose()


async def _register_bootstrap_admin(client: AsyncClient, email: str, password: str) -> dict:
    response = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password}
    )
    assert response.status_code == 201
    return response.json()


async def _login(client: AsyncClient, email: str, password: str) -> dict:
    response = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": password}
    )
    assert response.status_code == 200
    return response.json()


def _auth_header(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


async def _register_as(
    client: AsyncClient, admin_token: str, email: str, password: str, role: str
) -> dict:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "role": role},
        headers=_auth_header(admin_token),
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_list_users_requires_admin(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    await _register_as(
        client, admin_tokens["access_token"], "viewer@example.com", "viewerpass123", "viewer"
    )
    viewer_tokens = await _login(client, "viewer@example.com", "viewerpass123")

    response = await client.get(
        "/api/v1/users", headers=_auth_header(viewer_tokens["access_token"])
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_list_all_users(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    await _register_as(
        client, admin_tokens["access_token"], "analyst@example.com", "analystpass123", "analyst"
    )

    response = await client.get(
        "/api/v1/users", headers=_auth_header(admin_tokens["access_token"])
    )

    assert response.status_code == 200
    emails = {u["email"] for u in response.json()}
    assert emails == {"admin@example.com", "analyst@example.com"}
    assert all(u["is_active"] is True for u in response.json())


@pytest.mark.asyncio
async def test_admin_can_change_another_users_role(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    target = await _register_as(
        client, admin_tokens["access_token"], "viewer@example.com", "viewerpass123", "viewer"
    )

    response = await client.patch(
        f"/api/v1/users/{target['id']}",
        json={"role": "analyst"},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    assert response.status_code == 200
    assert response.json()["role"] == "analyst"


@pytest.mark.asyncio
async def test_admin_can_deactivate_another_user(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    target = await _register_as(
        client, admin_tokens["access_token"], "viewer@example.com", "viewerpass123", "viewer"
    )

    response = await client.patch(
        f"/api/v1/users/{target['id']}",
        json={"is_active": False},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is False


@pytest.mark.asyncio
async def test_deactivated_user_cannot_log_in(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    target = await _register_as(
        client, admin_tokens["access_token"], "viewer@example.com", "viewerpass123", "viewer"
    )
    await client.patch(
        f"/api/v1/users/{target['id']}",
        json={"is_active": False},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    response = await client.post(
        "/api/v1/auth/login", data={"username": "viewer@example.com", "password": "viewerpass123"}
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_deactivating_a_user_invalidates_their_existing_token_immediately(
    client: AsyncClient,
):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    target = await _register_as(
        client, admin_tokens["access_token"], "viewer@example.com", "viewerpass123", "viewer"
    )
    # Log in *before* deactivation — this access token is still
    # unexpired, and should stop working the moment the account is
    # deactivated, not just block future logins.
    viewer_tokens = await _login(client, "viewer@example.com", "viewerpass123")

    await client.patch(
        f"/api/v1/users/{target['id']}",
        json={"is_active": False},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    response = await client.get(
        "/api/v1/auth/me", headers=_auth_header(viewer_tokens["access_token"])
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_cannot_modify_their_own_account(client: AsyncClient):
    admin = await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.patch(
        f"/api/v1/users/{admin['id']}",
        json={"is_active": False},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_non_admin_cannot_update_users(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")
    target = await _register_as(
        client, admin_tokens["access_token"], "analyst@example.com", "analystpass123", "analyst"
    )
    analyst_tokens = await _login(client, "analyst@example.com", "analystpass123")

    response = await client.patch(
        f"/api/v1/users/{target['id']}",
        json={"role": "admin"},
        headers=_auth_header(analyst_tokens["access_token"]),
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_updating_a_nonexistent_user_returns_404(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.patch(
        "/api/v1/users/999999",
        json={"role": "viewer"},
        headers=_auth_header(admin_tokens["access_token"]),
    )

    assert response.status_code == 404
