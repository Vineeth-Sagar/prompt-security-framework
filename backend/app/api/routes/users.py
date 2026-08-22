"""GET /api/v1/users (list), PATCH /api/v1/users/{id} (update role/active).

Admin-only — managing *other* users is a strictly higher-trust
operation than reading the decision log (admin/analyst). Deliberately a
separate router from api/routes/auth.py: that module is about
authenticating *as* a user (register/login/refresh/me), this one is
about an admin managing *other* users — a different concern with a
different trust bar, not just a bigger version of the same thing.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.routes.auth import UserPublic
from app.auth.models import User, UserRole
from app.auth.security import require_role
from app.db import get_session

router = APIRouter(prefix="/api/v1/users", tags=["users"])

_ADMIN_ONLY = (UserRole.admin,)


class UserUpdateRequest(BaseModel):
    """Partial update — only the fields actually provided are changed."""

    role: UserRole | None = None
    is_active: bool | None = None


@router.get("", response_model=list[UserPublic])
async def list_users(
    session: AsyncSession = Depends(get_session),  # noqa: B008
    _admin: User = Depends(require_role(*_ADMIN_ONLY)),  # noqa: B008
) -> list[User]:
    """List every user, oldest first (registration order)."""
    result = await session.exec(select(User).order_by(User.created_at))
    return list(result.all())


@router.patch("/{user_id}", response_model=UserPublic)
async def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    admin: User = Depends(require_role(*_ADMIN_ONLY)),  # noqa: B008
) -> User:
    """Update a user's role and/or active status.

    An admin can't modify their own account through this endpoint — a
    self-demotion or self-deactivation here could lock every admin out
    of the deployment with no way back in. (Bootstrap only creates the
    *first* admin on an empty `users` table; every registration after
    that already requires an authenticated admin caller, so there's no
    self-service recovery path once the last admin account is gone.)
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify your own account via this endpoint",
        )

    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if payload.role is not None:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active

    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
