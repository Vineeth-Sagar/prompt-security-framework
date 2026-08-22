"""User model and role enum for auth/RBAC.

Pure schema — password hashing and JWT logic live in security.py, route
handlers in api/routes/auth.py. Migrated via Alembic
(backend/alembic/versions/), not `SQLModel.metadata.create_all()`, so
schema changes have an auditable history.
"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class UserRole(str, Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, nullable=False)
    hashed_password: str
    role: UserRole = Field(default=UserRole.viewer, nullable=False)
    # sa_column=Column(DateTime(timezone=True)) rather than SQLModel's
    # default plain `datetime` inference (which maps to Postgres'
    # TIMESTAMP WITHOUT TIME ZONE) — default_factory produces a
    # tz-aware `datetime.now(UTC)`, and asyncpg refuses to write a
    # tz-aware value into a tz-naive column at all (DataError, not a
    # silent truncation). Found by actually running against Postgres —
    # invisible under the SQLite tests use, which doesn't distinguish
    # the two.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
