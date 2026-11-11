import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import REPO_ROOT, LLMRuntime, Settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.upper().startswith("PLANE_AI_"):
            monkeypatch.delenv(key)


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)  # ignore any local .env files


def test_development_defaults():
    s = make()
    assert s.debug is True
    assert s.log_level == "DEBUG"
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8100
    assert s.llm_runtime is LLMRuntime.OLLAMA
    assert s.llm_base_url == "http://localhost:11434"
    assert s.llm_context_length == 8192
    assert s.model_storage_path == REPO_ROOT / "models"


def test_env_vars_override(monkeypatch):
    monkeypatch.setenv("PLANE_AI_API_PORT", "9000")
    monkeypatch.setenv("PLANE_AI_LLM_RUNTIME", "llama-cpp")
    monkeypatch.setenv("PLANE_AI_LLM_CONTEXT_LENGTH", "4096")
    s = make()
    assert s.api_port == 9000
    assert s.llm_runtime is LLMRuntime.LLAMA_CPP
    assert s.llm_base_url == "http://localhost:8080"
    assert s.llm_context_length == 4096


def test_production_requires_db_password():
    with pytest.raises(ValidationError, match="DB_PASSWORD"):
        make(app_env="production")


def test_production_rejects_debug():
    with pytest.raises(ValidationError, match="DEBUG"):
        make(app_env="production", debug=True, db_password="x")


def test_production_valid():
    s = make(app_env="production", db_password="s3cr3t")
    assert s.debug is False
    assert s.log_level == "INFO"


def test_database_url_built_and_escaped():
    s = make(db_password="p@ss/word")
    assert s.sqlalchemy_database_url == (
        "postgresql+psycopg://plane_ai:p%40ss%2Fword@localhost:5433/plane_ai"
    )


def test_safe_summary_hides_secrets():
    s = make(db_password="topsecret")
    summary = s.safe_summary()
    assert "topsecret" not in repr(summary)
    assert summary["db_password_set"] is True


def test_invalid_context_length():
    with pytest.raises(ValidationError):
        make(llm_context_length=100)
