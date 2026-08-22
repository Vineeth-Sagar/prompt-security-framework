"""GET /api/v1/logs (paginated, filterable), GET /api/v1/logs/{id} (full detail).

Both admin- and analyst-gated — viewers don't get access to the
decision log, only the higher-trust roles that actually investigate
flagged conversations.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.auth.models import UserRole
from app.auth.security import require_role
from app.db import get_session
from app.logging.models import DecisionLog

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])

_LOG_READER_ROLES = (UserRole.admin, UserRole.analyst)

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class DecisionLogPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
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
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),  # noqa: B008
    _user=Depends(require_role(*_LOG_READER_ROLES)),  # noqa: B008
) -> PaginatedLogs:
    """List decision logs, newest first, filtered by any combination of
    session_id / action / [start_date, end_date] and paginated via limit+offset."""
    filters = []
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
    _user=Depends(require_role(*_LOG_READER_ROLES)),  # noqa: B008
) -> DecisionLog:
    log = await session.get(DecisionLog, log_id)
    if log is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    return log
