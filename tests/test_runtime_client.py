"""Runtime client checks against mocked HTTP responses (no real runtime needed)."""

import asyncio

import httpx
import pytest

from app.core.config import Settings
from app.llm.runtime import LlamaCppClient, OllamaClient, RuntimeStatus, create_runtime_client


def settings(**kw) -> Settings:
    return Settings(_env_file=None, llm_host="plane-ai-llm", **kw)


def run_check(client) -> RuntimeStatus:
    async def go():
        try:
            return await client.check()
        finally:
            await client.aclose()

    return asyncio.run(go())


def transport(routes: dict[str, httpx.Response | Exception]) -> httpx.MockTransport:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        result = routes.get(request.url.path)
        if isinstance(result, Exception):
            raise result
        return result or httpx.Response(404)

    t = httpx.MockTransport(handler)
    t.seen = seen  # type: ignore[attr-defined]
    return t


# ---------------------------------------------------------------- factory


def test_factory_selects_client_by_runtime():
    assert isinstance(create_runtime_client(settings(llm_runtime="ollama")), OllamaClient)
    assert isinstance(create_runtime_client(settings(llm_runtime="llama-cpp")), LlamaCppClient)


# ---------------------------------------------------------------- ollama


@pytest.mark.parametrize(
    "models,expected",
    [
        ([{"name": "qwen3.5-2b:latest"}], RuntimeStatus.OK),
        ([{"name": "qwen3.5-2b"}], RuntimeStatus.OK),
        ([{"name": "llama3:latest"}], RuntimeStatus.MODEL_MISSING),
        ([], RuntimeStatus.MODEL_MISSING),
    ],
)
def test_ollama_model_detection(models, expected):
    t = transport({"/api/tags": httpx.Response(200, json={"models": models})})
    client = OllamaClient(settings(llm_runtime="ollama"), transport=t)
    assert run_check(client) is expected
    assert t.seen == ["http://plane-ai-llm:11434/api/tags"]


def test_ollama_unreachable():
    t = transport({"/api/tags": httpx.ConnectError("refused")})
    assert run_check(OllamaClient(settings(llm_runtime="ollama"), transport=t)) is RuntimeStatus.UNAVAILABLE


def test_ollama_bad_json():
    t = transport({"/api/tags": httpx.Response(200, text="not json")})
    assert run_check(OllamaClient(settings(llm_runtime="ollama"), transport=t)) is RuntimeStatus.UNAVAILABLE


# ---------------------------------------------------------------- llama.cpp


def test_llama_cpp_ok():
    t = transport(
        {
            "/health": httpx.Response(200, json={"status": "ok"}),
            "/v1/models": httpx.Response(200, json={"data": [{"id": "qwen3.5-2b"}]}),
        }
    )
    client = LlamaCppClient(settings(llm_runtime="llama-cpp"), transport=t)
    assert run_check(client) is RuntimeStatus.OK
    assert t.seen[0] == "http://plane-ai-llm:8080/health"


def test_llama_cpp_loading():
    t = transport({"/health": httpx.Response(503, json={"error": {"message": "Loading model"}})})
    assert run_check(LlamaCppClient(settings(llm_runtime="llama-cpp"), transport=t)) is RuntimeStatus.LOADING


def test_llama_cpp_wrong_alias():
    t = transport(
        {
            "/health": httpx.Response(200, json={"status": "ok"}),
            "/v1/models": httpx.Response(200, json={"data": [{"id": "other"}]}),
        }
    )
    assert (
        run_check(LlamaCppClient(settings(llm_runtime="llama-cpp"), transport=t))
        is RuntimeStatus.MODEL_MISSING
    )


def test_llama_cpp_unreachable():
    t = transport({"/health": httpx.ConnectTimeout("timeout")})
    assert (
        run_check(LlamaCppClient(settings(llm_runtime="llama-cpp"), transport=t))
        is RuntimeStatus.UNAVAILABLE
    )
