"""POST /api/v1/auth/register, POST /api/v1/auth/login,
POST /api/v1/auth/refresh, GET /api/v1/auth/me.

Bootstrap rule: the very first user ever registered (an empty `users`
table) becomes an admin with no auth required at all — otherwise
there's no way to get an admin account into a fresh deployment. Every
registration after that requires an authenticated admin caller.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, EmailStr
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User, UserRole
from app.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_current_user_optional,
    hash_password,
    verify_password,
)
from app.db import get_session

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    # Ignored for the bootstrap admin (see register()) — the first user
    # is always admin regardless of what's requested here.
    role: UserRole = UserRole.viewer


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


async def _users_table_is_empty(session: AsyncSession) -> bool:
    count = (await session.exec(select(func.count()).select_from(User))).one()
    return count == 0


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User | None = Depends(get_current_user_optional),  # noqa: B008
) -> User:
    """Register a new user.

    The first registration against an empty `users` table bootstraps an
    admin account and requires no authentication. Every registration
    after that requires the caller to already be an authenticated admin.
    """
    is_bootstrap = await _users_table_is_empty(session)

    if is_bootstrap:
        role = UserRole.admin
    elif current_user is None or current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only an admin can register new users",
        )
    else:
        role = payload.role

    existing = (await session.exec(select(User).where(User.email == payload.email))).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already registered"
        )

    user = User(email=payload.email, hashed_password=hash_password(payload.password), role=role)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenPair)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),  # noqa: B008
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> TokenPair:
    """Exchange email + password for a token pair.

    Uses the OAuth2 password flow's field names — `username` is the
    user's email, not a separate username.
    """
    user = (await session.exec(select(User).where(User.email == form_data.username))).first()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated"
        )

    return TokenPair(
        access_token=create_access_token(user.email),
        refresh_token=create_refresh_token(user.email),
    )


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    payload: RefreshRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> AccessTokenResponse:
    """Mint a new access token from a still-valid refresh token."""
    token_payload = decode_token(payload.refresh_token)

    if token_payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected a refresh token, not an access token",
        )

    email = token_payload.get("sub")
    user = (await session.exec(select(User).where(User.email == email))).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is deactivated"
        )

    return AccessTokenResponse(access_token=create_access_token(user.email))


@router.get("/me", response_model=UserPublic)
async def me(current_user: User = Depends(get_current_user)) -> User:  # noqa: B008
    return current_user
