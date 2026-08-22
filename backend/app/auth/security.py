"""Password hashing, JWT issuance/verification, and role-guard dependencies.

Access tokens are short-lived (30 min default) and carry `type: "access"`;
refresh tokens are long-lived (7 days default) and carry `type:
"refresh"` — `get_current_user` only ever accepts an access token, so a
leaked refresh token can't be used directly as a bearer credential on a
protected route, only to mint a new access token via
`POST /auth/refresh` (api/routes/auth.py).

`sub` (the JWT subject) is the user's email, not their numeric id — this
project doesn't have a separate stable-identifier requirement beyond
email uniqueness, and email is what login already keys on.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User, UserRole
from app.config import get_settings
from app.db import get_session

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

TokenType = Literal["access", "refresh"]

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
_oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(password, hashed_password)


def _create_token(subject: str, token_type: TokenType, expires_delta: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {"sub": subject, "type": token_type, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, get_settings().jwt_secret, algorithm=ALGORITHM)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    return _create_token(
        subject, "access", expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )


def create_refresh_token(subject: str, expires_delta: timedelta | None = None) -> str:
    return _create_token(
        subject, "refresh", expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )


def decode_token(token: str) -> dict:
    """Decode and validate a JWT's signature and expiry.

    Raises:
        HTTPException: 401, for any decode/signature/expiry failure —
            jose's various JWTError subclasses (including
            ExpiredSignatureError) are all collapsed into one clean 401
            rather than leaking SDK-specific exception types to callers.
    """
    try:
        return jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        ) from exc


async def get_current_user(
    token: str = Depends(_oauth2_scheme),  # noqa: B008
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> User:
    """Resolve the bearer token to a User, or raise 401.

    Rejects a refresh token used where an access token is expected —
    only `type: "access"` tokens authenticate a request.
    """
    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A refresh token cannot be used to authenticate requests",
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        )

    result = await session.exec(select(User).where(User.email == email))
    user = result.first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        )

    if not user.is_active:
        # Re-checked on every request (not just at login) so a token
        # issued before an admin deactivated this account stops working
        # immediately, rather than staying valid until it naturally expires.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is deactivated"
        )

    return user


async def get_current_user_optional(
    token: str | None = Depends(_oauth2_scheme_optional),  # noqa: B008
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> User | None:
    """Like `get_current_user`, but returns None instead of raising when
    there's no token or it's invalid — for routes with a public path
    (e.g. bootstrap registration) that only need to check the caller's
    identity/role when one is actually present."""
    if token is None:
        return None
    try:
        return await get_current_user(token=token, session=session)
    except HTTPException:
        return None


def require_role(*allowed_roles: UserRole):
    """Dependency factory: 403 unless the current user's role is one of `allowed_roles`.

    Usage: `Depends(require_role(UserRole.admin))`, or
    `Depends(require_role(UserRole.admin, UserRole.analyst))` for
    multiple allowed roles.
    """

    async def _check_role(user: User = Depends(get_current_user)) -> User:  # noqa: B008
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return user

    return _check_role
