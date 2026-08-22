"""Async SQLAlchemy engine + session, shared by any module that needs
Postgres — auth (Phase 9) now, explainable logging (Phase 10) next.

A lazy singleton engine, same pattern as get_redis_client()/
get_policy_engine() elsewhere in this project: building the engine
doesn't connect eagerly, so importing this module doesn't require the
database to be reachable.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_settings().database_url, echo=False)
    return _engine


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-dependency-shaped async session generator."""
    async with AsyncSession(get_engine()) as session:
        yield session
