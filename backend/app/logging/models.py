"""Decision log schema — the durable record of every pipeline run.

Stores the full reasoning trace (drift breakdown, IFS-R result, which
policy rule fired, PII found, per-stage latency) as JSON columns rather
than a fully normalized schema: the shape of each of these nested
objects is still evolving alongside SWCSA/IFS-R/output-governance, and
JSON columns mean adding a new field to DriftBreakdown, say, doesn't
require a migration here too. Queryable/filterable fields that matter
for the log-browsing UI (session_id, policy_action, created_at) are
still plain indexed columns, not buried in JSON.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel


class DecisionLog(SQLModel, table=True):
    __tablename__ = "decision_logs"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    input_modality: str
    drift_breakdown: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    ifsr_result: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    policy_action: str = Field(index=True)
    matched_rule: str
    pii_found: list[Any] = Field(sa_column=Column(JSON, nullable=False))
    latency_ms_per_stage: dict[str, float] = Field(sa_column=Column(JSON, nullable=False))
    # timezone=True — see the matching comment on app/auth/models.py's
    # User.created_at for why this isn't just the SQLModel-inferred
    # plain DateTime.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
