"""GET /api/v1/logs (paginated, filterable), GET /api/v1/logs/{id} (full detail).

Per-user scoped. Every authenticated user can read the decision log,
but what they see depends on role:

- viewer / analyst see ONLY the decisions they themselves ran. This is
  an access-control boundary, enforced server-side: the owner filter is
  applied to every query and a non-admin cannot widen it, so one user
  can never see another user's prompts/decisions.
- admin sees every decision, and may narrow to one user via the
  optional `user_id` query param (the "filter by user" the log
  dashboard offers only to admins).

Attribution comes from DecisionLog.user_id/user_email, written by the
pipeline route (decision_logger.log_decision). Rows with a null user_id
(pre-attribution history, or unattributed script runs) are visible to
admins only — a non-admin's `user_id == me` filter never matches them.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import User, UserRole
from app.auth.security import get_current_user
from app.db import get_session
from app.logging.models import DecisionLog

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class DecisionLogPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    user_id: int | None
    user_email: str | None
    input_modality: str
    drift_breakdown: dict[str, Any]
    ifsr_result: dict[str, Any]
    policy_action: str
    matched_rule: str
    pii_found: list[Any]
    latency_ms_per_stage: dict[str, float]
    created_at: datetime


class PaginatedLogs(BaseModel):
    items: list[DecisionLogPublic]
    total: int
    limit: int
    offset: int


@router.get("", response_model=PaginatedLogs)
async def list_logs(
    session_id: str | None = None,
    action: str | None = Query(default=None, description="Filter by policy_action"),
    user_id: int | None = Query(
        default=None, description="Admin-only: filter to one user's decisions"
    ),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> PaginatedLogs:
    """List decision logs, newest first, filtered by any combination of
    session_id / action / [start_date, end_date] and paginated via limit+offset.

    Ownership scoping is applied first and is not optional for
    non-admins: an analyst/viewer only ever sees their own rows, and the
    `user_id` param is ignored for them (they can't widen or redirect
    the filter to someone else). For an admin the `user_id` param is a
    normal filter — omitted means all users.
    """
    filters = []

    is_admin = current_user.role == UserRole.admin
    if is_admin:
        # Admins may optionally narrow to one user; no filter => everyone.
        if user_id is not None:
            filters.append(DecisionLog.user_id == user_id)
    else:
        # Hard ownership boundary — server-enforced, ignores any client-
        # supplied user_id so a non-admin can't read another user's log.
        filters.append(DecisionLog.user_id == current_user.id)

    if session_id is not None:
        filters.append(DecisionLog.session_id == session_id)
    if action is not None:
        filters.append(DecisionLog.policy_action == action)
    if start_date is not None:
        filters.append(DecisionLog.created_at >= start_date)
    if end_date is not None:
        filters.append(DecisionLog.created_at <= end_date)

    count_query = select(func.count()).select_from(DecisionLog)
    for condition in filters:
        count_query = count_query.where(condition)
    total = (await session.exec(count_query)).one()

    rows_query = select(DecisionLog)
    for condition in filters:
        rows_query = rows_query.where(condition)
    rows_query = rows_query.order_by(DecisionLog.created_at.desc()).limit(limit).offset(offset)
    rows = (await session.exec(rows_query)).all()

    return PaginatedLogs(items=rows, total=total, limit=limit, offset=offset)


@router.get("/{log_id}", response_model=DecisionLogPublic)
async def get_log(
    log_id: int,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> DecisionLog:
    log = await session.get(DecisionLog, log_id)
    if log is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")

    # Same ownership boundary as the list endpoint, applied to direct-by-
    # id access. A non-admin fetching someone else's log id gets 404, not
    # 403 — not confirming the row exists avoids leaking that another
    # user's decision with that id is there at all.
    if current_user.role != UserRole.admin and log.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")

    return log
