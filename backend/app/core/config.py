"""Centralized configuration for Plane AI.

All settings come from environment variables prefixed with ``PLANE_AI_``.
Values are resolved in this order (highest priority first):

1. Real process environment variables
2. ``.env.<APP_ENV>`` file at the repository root (e.g. ``.env.production``)
3. ``.env`` file at the repository root
4. Safe defaults defined below

Secrets are never hardcoded: the database password defaults to empty and
production startup fails unless it is provided.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "PLANE_AI_"

# backend/app/core/config.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class LLMRuntime(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama-cpp"


DEFAULT_RUNTIME_URLS: dict[LLMRuntime, str] = {
    LLMRuntime.OLLAMA: "http://localhost:11434",
    LLMRuntime.LLAMA_CPP: "http://localhost:8080",
}


def _env_files(app_env: str) -> tuple[Path, ...]:
    """Base .env first, then the environment-specific override."""
    return (REPO_ROOT / ".env", REPO_ROOT / f".env.{app_env}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "Plane AI"
    app_env: AppEnv = AppEnv.DEVELOPMENT
    debug: bool | None = None  # derived from app_env when unset
    log_level: str | None = None  # derived from app_env when unset

    # --- API server ---
    api_host: str = "127.0.0.1"  # localhost-only by default; containers set 0.0.0.0
    api_port: int = Field(default=8100, ge=1, le=65535)

    # --- Database (Plane AI's own PostgreSQL, not Plane's) ---
    # Either set DATABASE_URL directly, or the individual DB_* parts.
    database_url: SecretStr | None = None
    db_host: str = "localhost"
    db_port: int = Field(default=5433, ge=1, le=65535)
    db_name: str = "plane_ai"
    db_user: str = "plane_ai"
    db_password: SecretStr = SecretStr("")
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=5, ge=0, le=100)
    db_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    db_echo: bool = False  # log every SQL statement (very noisy)

    # --- Model storage (outside Git) ---
    model_storage_path: Path = REPO_ROOT / "models"

    # --- LLM runtime ---
    llm_runtime: LLMRuntime = LLMRuntime.OLLAMA
    llm_base_url: str | None = None  # derived from llm_runtime when unset
    llm_model_name: str = "qwen2.5:7b-instruct"
    llm_model_file: str | None = None  # GGUF filename for llama-cpp, relative to model_storage_path
    llm_context_length: int = Field(default=8192, ge=512, le=131072)
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0)

    # ------------------------------------------------------------------ validators

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.upper()
        if v not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"invalid log level: {v}")
        return v

    @field_validator("model_storage_path")
    @classmethod
    def _resolve_storage_path(cls, v: Path) -> Path:
        v = v.expanduser()
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @model_validator(mode="after")
    def _apply_derived_defaults(self) -> "Settings":
        is_prod = self.app_env is AppEnv.PRODUCTION
        if self.debug is None:
            self.debug = self.app_env is AppEnv.DEVELOPMENT
        if self.log_level is None:
            self.log_level = "INFO" if is_prod else "DEBUG"
        if self.llm_base_url is None:
            self.llm_base_url = DEFAULT_RUNTIME_URLS[self.llm_runtime]
        self.llm_base_url = self.llm_base_url.rstrip("/")

        if is_prod:
            self._validate_production()
        return self

    def _validate_production(self) -> None:
        errors: list[str] = []
        if self.debug:
            errors.append(f"{ENV_PREFIX}DEBUG must be false in production")
        if self.database_url is None and not self.db_password.get_secret_value():
            errors.append(
                f"{ENV_PREFIX}DB_PASSWORD (or {ENV_PREFIX}DATABASE_URL) is required in production"
            )
        if self.llm_runtime is LLMRuntime.LLAMA_CPP and not self.llm_model_file:
            errors.append(f"{ENV_PREFIX}LLM_MODEL_FILE is required when using llama-cpp in production")
        if errors:
            raise ValueError("; ".join(errors))

    # ------------------------------------------------------------------ helpers

    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def sqlalchemy_database_url(self) -> str:
        """Connection URL. Contains the password - never log it."""
        if self.database_url is not None:
            return self.database_url.get_secret_value()
        from urllib.parse import quote

        password = quote(self.db_password.get_secret_value(), safe="")
        auth = f"{quote(self.db_user, safe='')}:{password}" if password else quote(self.db_user, safe="")
        return f"postgresql+psycopg://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def llm_model_path(self) -> Path | None:
        """Absolute path to the GGUF file for llama-cpp, if configured."""
        if not self.llm_model_file:
            return None
        return self.model_storage_path / self.llm_model_file

    def safe_summary(self) -> dict[str, object]:
        """Settings snapshot that is safe to log (no secrets)."""
        data = self.model_dump(exclude={"database_url", "db_password"})
        data["database_configured_via_url"] = self.database_url is not None
        data["db_password_set"] = bool(self.db_password.get_secret_value())
        return data


def load_settings(**overrides: object) -> Settings:
    """Build a fresh Settings instance (no caching). Useful for tests."""
    app_env = str(
        overrides.get("app_env") or os.environ.get(f"{ENV_PREFIX}APP_ENV", AppEnv.DEVELOPMENT.value)
    ).lower()
    return Settings(_env_file=_env_files(app_env), **overrides)  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings. Call ``get_settings.cache_clear()`` to reload."""
    return load_settings()
