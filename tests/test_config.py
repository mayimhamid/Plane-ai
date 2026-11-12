import pytest
from pydantic import ValidationError

from app.core.config import REPO_ROOT, LLMRuntime, Settings


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
        make(app_env="production", debug=True, db_password="x", service_token="t")


def test_production_valid():
    s = make(app_env="production", db_password="s3cr3t", service_token="t")
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


def test_llm_host_used_for_base_url():
    s = make(llm_host="plane-ai-llm", llm_runtime="llama-cpp")
    assert s.llm_base_url == "http://plane-ai-llm:8080"
    assert s.llm_model_path == s.model_storage_path / "Qwen3.5-2B-Q4_K_M.gguf"


def test_model_file_must_be_bare_name():
    with pytest.raises(ValidationError):
        make(llm_model_file="../outside.gguf")


def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    # docker compose passes unset optional vars as empty strings
    monkeypatch.setenv("PLANE_AI_LOG_LEVEL", "")
    monkeypatch.setenv("PLANE_AI_LLM_BASE_URL", "")
    monkeypatch.setenv("PLANE_AI_DEBUG", "")
    s = make()
    assert s.log_level == "DEBUG"
    assert s.llm_base_url == "http://localhost:11434"
    assert s.debug is True
