"""Database engine/session management for Plane AI's own PostgreSQL."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

from fastapi import Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

logger = logging.getLogger(__name__)


class DatabaseProtocol(Protocol):
    """What the app needs from a database (lets tests plug in a fake)."""

    def ping(self) -> None: ...
    def dispose(self) -> None: ...


class Database:
    def __init__(self, settings: Settings) -> None:
        # The engine connects lazily: creating it never touches the network.
        self.engine: Engine = create_engine(
            settings.sqlalchemy_database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            echo=False,  # SQL logging is controlled by the logging config (db_echo)
            connect_args={
                "connect_timeout": settings.db_connect_timeout_seconds,
                "application_name": "plane-ai",
            },
        )
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def ping(self) -> None:
        """Raise if the database is unreachable."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def dispose(self) -> None:
        self.engine.dispose()
        logger.info("database connection pool closed")


def get_db_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    database: Database = request.app.state.db
    session = database.session_factory()
    try:
        yield session
    finally:
        session.close()
