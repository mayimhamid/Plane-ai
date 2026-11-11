"""Clients for the selected local LLM runtime.

For now this only answers "is the runtime up, and is our model available?",
which feeds ``/ready``. Generation comes in a later phase.

Both runtimes are reached at ``settings.llm_base_url`` (``http://plane-ai-llm:<port>``
in Docker, where both share that network alias).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

import httpx

from app.core.config import LLMRuntime, Settings

logger = logging.getLogger(__name__)

CHECK_TIMEOUT = httpx.Timeout(5.0)


class RuntimeStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"  # can't connect, or an unexpected response
    LOADING = "loading"  # runtime is up but still loading the model
    MODEL_MISSING = "model_missing"  # runtime is up but doesn't serve our model


class RuntimeClient(Protocol):
    runtime: LLMRuntime

    async def check(self) -> RuntimeStatus: ...
    async def aclose(self) -> None: ...


class _HttpRuntimeClient:
    runtime: LLMRuntime

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model_name = settings.llm_model_name
        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url or "",
            timeout=CHECK_TIMEOUT,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get_json(self, path: str) -> dict:
        resp = await self._http.get(path)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"unexpected response from {path}")
        return data


class OllamaClient(_HttpRuntimeClient):
    runtime = LLMRuntime.OLLAMA

    async def check(self) -> RuntimeStatus:
        try:
            data = await self._get_json("/api/tags")
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("ollama check failed: %r", exc)
            return RuntimeStatus.UNAVAILABLE
        served = {m.get("name") for m in data.get("models", []) if isinstance(m, dict)}
        wanted = {self.model_name} if ":" in self.model_name else {self.model_name, f"{self.model_name}:latest"}
        return RuntimeStatus.OK if served & wanted else RuntimeStatus.MODEL_MISSING


class LlamaCppClient(_HttpRuntimeClient):
    runtime = LLMRuntime.LLAMA_CPP

    async def check(self) -> RuntimeStatus:
        try:
            health = await self._http.get("/health")
            if health.status_code == 503:  # llama-server: model still loading
                return RuntimeStatus.LOADING
            health.raise_for_status()
            data = await self._get_json("/v1/models")
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("llama.cpp check failed: %r", exc)
            return RuntimeStatus.UNAVAILABLE
        served = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
        return RuntimeStatus.OK if self.model_name in served else RuntimeStatus.MODEL_MISSING


def create_runtime_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> RuntimeClient:
    if settings.llm_runtime is LLMRuntime.OLLAMA:
        return OllamaClient(settings, transport)
    return LlamaCppClient(settings, transport)
