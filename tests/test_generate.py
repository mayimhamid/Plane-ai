"""Generation clients against mocked runtime HTTP (request bodies and stream parsing)."""

import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.llm.effort import Effort, profile_for
from app.llm.generate import GenerationError, LlamaCppGenerator, OllamaGenerator, Usage

LOW = profile_for(Effort.LOW, 8, 8192)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, llm_host="plane-ai-llm", **kw)


def run(coro):
    return asyncio.run(coro)


def test_ollama_complete_sends_effort_options():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hi"}, "prompt_eval_count": 7, "eval_count": 2})

    gen = OllamaGenerator(settings(llm_runtime="ollama"), transport=httpx.MockTransport(handler))
    result = run(gen.complete([{"role": "user", "content": "x"}], LOW, json_schema={"type": "object"}))
    assert result.text == "hi" and result.usage.prompt_tokens == 7
    opts = seen["body"]["options"]
    assert opts == {"num_ctx": 2048, "num_predict": 256, "num_thread": 2, "temperature": 0.1}
    assert seen["body"]["think"] is False and seen["body"]["format"] == {"type": "object"}


def test_ollama_stream():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 3, "eval_count": 2}),
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="\n".join(lines)))
    gen = OllamaGenerator(settings(llm_runtime="ollama"), transport=transport)
    usage = Usage()

    async def collect():
        return [p async for p in gen.stream([{"role": "user", "content": "x"}], LOW, usage)]

    assert run(collect()) == ["Hel", "lo"]
    assert usage.answer_tokens == 2


def test_llama_cpp_stream_and_body():
    seen = {}
    sse = "\n\n".join(
        [
            "data: " + json.dumps({"choices": [{"delta": {"content": "A"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "B"}}]}),
            "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}),
            "data: [DONE]",
        ]
    )

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse)

    gen = LlamaCppGenerator(settings(llm_runtime="llama-cpp"), transport=httpx.MockTransport(handler))
    usage = Usage()

    async def collect():
        return [p async for p in gen.stream([{"role": "user", "content": "x"}], LOW, usage)]

    assert run(collect()) == ["A", "B"]
    assert usage.prompt_tokens == 4
    assert seen["body"]["max_tokens"] == 256
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_runtime_error_becomes_generation_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    gen = OllamaGenerator(settings(llm_runtime="ollama"), transport=transport)
    with pytest.raises(GenerationError):
        run(gen.complete([{"role": "user", "content": "x"}], LOW))


def test_embed_off_without_model():
    gen = OllamaGenerator(settings(llm_runtime="ollama"), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert run(gen.embed(["a"])) is None
