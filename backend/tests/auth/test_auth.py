from datetime import timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.security import create_access_token, decode_token
from app.db import get_session
from app.main import app


@pytest_asyncio.fixture
async def client():
    """A real (in-memory SQLite) database wired into the app via
    dependency override — not mocked, so this exercises the actual
    SQLModel/SQLAlchemy query paths in security.py/routes/auth.py."""
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


# --- register/login happy path ---


@pytest.mark.asyncio
async def test_bootstrap_registration_creates_an_admin_with_no_auth(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register", json={"email": "admin@example.com", "password": "adminpass123"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "admin@example.com"
    assert body["role"] == "admin"
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_login_happy_path_returns_access_and_refresh_tokens(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")

    tokens = await _login(client, "admin@example.com", "adminpass123")

    assert tokens["token_type"] == "bearer"
    assert decode_token(tokens["access_token"])["type"] == "access"
    assert decode_token(tokens["refresh_token"])["type"] == "refresh"


@pytest.mark.asyncio
async def test_me_returns_the_authenticated_users_profile(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.get("/api/v1/auth/me", headers=_auth_header(tokens["access_token"]))

    assert response.status_code == 200
    assert response.json()["email"] == "admin@example.com"


@pytest.mark.asyncio
async def test_admin_can_register_additional_users_with_a_chosen_role(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "analyst@example.com", "password": "analystpass123", "role": "analyst"},
        headers=_auth_header(tokens["access_token"]),
    )

    assert response.status_code == 201
    assert response.json()["role"] == "analyst"


# --- wrong password rejected ---


@pytest.mark.asyncio
async def test_wrong_password_is_rejected(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")

    response = await client.post(
        "/api/v1/auth/login", data={"username": "admin@example.com", "password": "wrong-password"}
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_for_nonexistent_user_is_rejected(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/login", data={"username": "nobody@example.com", "password": "whatever"}
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_duplicate_email_registration_is_rejected(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "admin@example.com", "password": "different123"},
        headers=_auth_header(tokens["access_token"]),
    )

    assert response.status_code == 400


# --- expired token rejected ---


@pytest.mark.asyncio
async def test_expired_access_token_is_rejected(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    expired_token = create_access_token("admin@example.com", expires_delta=timedelta(seconds=-1))

    response = await client.get("/api/v1/auth/me", headers=_auth_header(expired_token))

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_missing_token_is_rejected(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_cannot_authenticate_a_request(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    tokens = await _login(client, "admin@example.com", "adminpass123")

    response = await client.get(
        "/api/v1/auth/me", headers=_auth_header(tokens["refresh_token"])
    )

    assert response.status_code == 401


# --- refresh flow ---


@pytest.mark.asyncio
async def test_refresh_mints_a_working_new_access_token(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    tokens = await _login(client, "admin@example.com", "adminpass123")

    refresh_response = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert refresh_response.status_code == 200
    new_access_token = refresh_response.json()["access_token"]

    me_response = await client.get("/api/v1/auth/me", headers=_auth_header(new_access_token))
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "admin@example.com"


@pytest.mark.asyncio
async def test_expired_refresh_token_is_rejected(client: AsyncClient):
    from app.auth.security import create_refresh_token

    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    expired_refresh = create_refresh_token(
        "admin@example.com", expires_delta=timedelta(seconds=-1)
    )

    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": expired_refresh})

    assert response.status_code == 401


# --- viewer role blocked from an admin-only route (403) ---


@pytest.mark.asyncio
async def test_viewer_role_is_blocked_from_admin_only_registration(client: AsyncClient):
    await _register_bootstrap_admin(client, "admin@example.com", "adminpass123")
    admin_tokens = await _login(client, "admin@example.com", "adminpass123")

    # Admin creates a viewer.
    await client.post(
        "/api/v1/auth/register",
        json={"email": "viewer@example.com", "password": "viewerpass123", "role": "viewer"},
        headers=_auth_header(admin_tokens["access_token"]),
    )
    viewer_tokens = await _login(client, "viewer@example.com", "viewerpass123")

    # Viewer attempts an admin-only action (registering another user).
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "someone-else@example.com", "password": "somepass123"},
        headers=_auth_header(viewer_tokens["access_token"]),
    )

    assert response.status_code == 403
