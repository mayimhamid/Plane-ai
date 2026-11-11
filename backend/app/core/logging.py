"""Logging setup.

Human-readable lines in development, one JSON object per line in production
(easy to ship to any log collector). Uvicorn's loggers are routed through the
same handler so all output looks consistent.
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone

from app.core.config import Settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    formatter = "json" if settings.is_production else "text"
    level = settings.log_level or "INFO"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "text": {"format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s"},
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["default"], "level": level},
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "sqlalchemy.engine": {"level": "INFO" if settings.db_echo else "WARNING"},
            },
        }
    )
