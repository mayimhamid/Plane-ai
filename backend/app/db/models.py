"""Import every ORM model module here so ``Base.metadata`` is complete (Alembic autogenerate reads it)."""

from app.db import tables  # noqa: F401
from app.db.base import Base

__all__ = ["Base"]
