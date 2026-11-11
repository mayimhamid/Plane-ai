"""Import every ORM model module here so ``Base.metadata`` is complete.

No models yet. Example once one exists:

    from app.db.tables.conversation import Conversation  # noqa: F401
"""

from app.db.base import Base

__all__ = ["Base"]
