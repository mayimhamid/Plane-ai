"""Text generation on the selected local runtime, with token-by-token streaming.

- Ollama: native ``/api/chat`` (NDJSON stream). Threads, context and answer length are set per request, so
  the effort profile applies fully.
- llama.cpp: OpenAI-compatible ``/v1/chat/completions`` (SSE stream). Threads are fixed at server start.

Thinking is turned off: on a small model it costs many tokens for little gain in these tasks.
Generations go through one semaphore, so a small machine never runs two at once and thrashes.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.core.config import LLMRuntime, Settings
from app.llm.effort import EffortProfile

logger = logging.getLogger(__name__)


# Qwen-style models may still emit a reasoning block; it is never shown to people or parsed as JSON.
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)

# When a job runs, every completed generation is recorded here (prompt + answer) so admins can later rate it and
# export it as training data (distillation / LoRA). `None` = not recording.
current_trace: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("current_trace", default=None)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


class ThinkFilter:
    """Drops `<think>...</think>` from a token stream, even when the tags are split across pieces."""

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, piece: str) -> str:
        self.buffer += piece
        out = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find("</think>")
                if end == -1:
                    self.buffer = self.buffer[-7:]  # keep a possible partial closing tag
                    return "".join(out)
                self.buffer = self.buffer[end + 8 :]
                self.inside = False
                continue
            start = self.buffer.find("<think>")
            if start == -1:
                # hold back a possible partial opening tag at the end
                keep = next((k for k in range(min(6, len(self.buffer)), 0, -1) if "<think>".startswith(self.buffer[-k:])), 0)
                out.append(self.buffer[: len(self.buffer) - keep])
                self.buffer = self.buffer[len(self.buffer) - keep :]
                return "".join(out)
            out.append(self.buffer[:start])
            self.buffer = self.buffer[start + 7 :]
            self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self.inside else self.buffer
        self.buffer = ""
        return rest


class GenerationError(RuntimeError):
    """The runtime could not produce an answer (down, timed out, bad response)."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    answer_tokens: int = 0


@dataclass
class Generation:
    text: str
    usage: Usage = field(default_factory=Usage)


class Generator(Protocol):
    async def complete(
        self, messages: list[dict[str, str]], profile: EffortProfile, json_schema: dict | None = None
    ) -> Generation: ...

    def stream(self, messages: list[dict[str, str]], profile: EffortProfile, usage: Usage) -> AsyncIterator[str]: ...

    async def embed(self, texts: list[str]) -> list[list[float]] | None: ...

    async def aclose(self) -> None: ...


class _HttpGenerator:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = settings.llm_model_name
        self.keep_alive = settings.llm_keep_alive
        self.embed_model = settings.embed_model
        self._lock = asyncio.Semaphore(settings.llm_max_concurrency)
        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url or "",
            timeout=httpx.Timeout(settings.llm_request_timeout_seconds, connect=5.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(
        self, messages: list[dict[str, str]], profile: EffortProfile, json_schema: dict | None = None
    ) -> Generation:
        async with self._lock:
            try:
                result = await self._complete(messages, profile, json_schema)
                result.text = strip_think(result.text)
                trace = current_trace.get()
                if trace is not None:
                    trace.append({"messages": messages, "output": result.text})
                return result
            except httpx.HTTPError as exc:
                raise GenerationError(f"runtime request failed: {exc.__class__.__name__}") from exc
            except (ValueError, KeyError, TypeError) as exc:
                raise GenerationError("unexpected response from the runtime") from exc

    async def stream(self, messages: list[dict[str, str]], profile: EffortProfile, usage: Usage) -> AsyncIterator[str]:
        async with self._lock:
            think = ThinkFilter()
            try:
                async for piece in self._stream(messages, profile, usage):
                    visible = think.feed(piece)
                    if visible:
                        yield visible
                rest = think.flush()
                if rest:
                    yield rest
            except httpx.HTTPError as exc:
                raise GenerationError(f"runtime request failed: {exc.__class__.__name__}") from exc

    async def _complete(self, messages, profile, json_schema) -> Generation:  # pragma: no cover - abstract
        raise NotImplementedError

    def _stream(self, messages, profile, usage) -> AsyncIterator[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        return None


class OllamaGenerator(_HttpGenerator):
    def _body(self, messages, profile: EffortProfile, stream: bool, json_schema: dict | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": profile.context_tokens,
                "num_predict": profile.max_answer_tokens,
                "num_thread": profile.threads,
                "temperature": profile.temperature,
            },
        }
        if json_schema is not None:
            body["format"] = json_schema
        return body

    async def _complete(self, messages, profile, json_schema) -> Generation:
        resp = await self._http.post("/api/chat", json=self._body(messages, profile, False, json_schema))
        resp.raise_for_status()
        data = resp.json()
        return Generation(
            text=data["message"]["content"],
            usage=Usage(int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)),
        )

    async def _stream(self, messages, profile, usage: Usage) -> AsyncIterator[str]:
        async with self._http.stream("POST", "/api/chat", json=self._body(messages, profile, True)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                piece = (data.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if data.get("done"):
                    usage.prompt_tokens = int(data.get("prompt_eval_count") or 0)
                    usage.answer_tokens = int(data.get("eval_count") or 0)

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embed_model or not texts:
            return None
        async with self._lock:
            try:
                resp = await self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning("embedding failed, falling back to lexical search: %s", exc.__class__.__name__)
                return None


class LlamaCppGenerator(_HttpGenerator):
    def _body(self, messages, profile: EffortProfile, stream: bool, json_schema: dict | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": profile.max_answer_tokens,
            "temperature": profile.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
            # Reuse the KV cache of the shared prompt prefix (system prompt, context) between requests.
            "cache_prompt": True,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if json_schema is not None:
            body["response_format"] = {"type": "json_object", "schema": json_schema}
        return body

    async def _complete(self, messages, profile, json_schema) -> Generation:
        resp = await self._http.post("/v1/chat/completions", json=self._body(messages, profile, False, json_schema))
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return Generation(
            text=data["choices"][0]["message"]["content"] or "",
            usage=Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)),
        )

    async def _stream(self, messages, profile, usage: Usage) -> AsyncIterator[str]:
        async with self._http.stream("POST", "/v1/chat/completions", json=self._body(messages, profile, True)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if data.get("usage"):
                    usage.prompt_tokens = int(data["usage"].get("prompt_tokens") or 0)
                    usage.answer_tokens = int(data["usage"].get("completion_tokens") or 0)
                for choice in data.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        yield piece

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embed_model or not texts:
            return None
        async with self._lock:
            try:
                resp = await self._http.post("/v1/embeddings", json={"model": self.embed_model, "input": texts})
                resp.raise_for_status()
                return [row["embedding"] for row in resp.json()["data"]]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning("embedding failed, falling back to lexical search: %s", exc.__class__.__name__)
                return None


def create_generator(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> Generator:
    if settings.llm_runtime is LLMRuntime.OLLAMA:
        return OllamaGenerator(settings, transport)
    return LlamaCppGenerator(settings, transport)
