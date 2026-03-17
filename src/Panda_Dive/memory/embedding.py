"""Embedding client helpers for memory vector reranking."""

import logging
import os
from collections.abc import Awaitable, Callable

import httpx

from ..configuration import Configuration

EmbeddingFunction = Callable[[str], Awaitable[list[float] | None]]


class EmbeddingClient:
    """Simple OpenAI-compatible embedding client."""

    def __init__(
        self,
        *,
        enabled: bool,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 20.0,
    ) -> None:
        """Initialize embedding client settings."""
        self._enabled = enabled
        self._provider = provider.strip().lower()
        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._warned_provider = False
        self._warned_missing_key = False

    async def embed_text(self, text: str) -> list[float] | None:
        """Create an embedding vector for text.

        Returns None when embedding is disabled or unavailable.
        """
        if not self._enabled:
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        if self._provider not in {"openai_compatible", "siliconflow_openai_compatible"}:
            if not self._warned_provider:
                logging.warning("Unsupported memory embedding provider: %s", self._provider)
                self._warned_provider = True
            return None
        if not self._api_key:
            if not self._warned_missing_key:
                logging.warning(
                    "Memory embedding enabled but API key is missing; "
                    "falling back to lexical-only retrieval."
                )
                self._warned_missing_key = True
            return None

        endpoint = f"{self._base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": self._model, "input": cleaned}
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
            data = body.get("data", [])
            if not data:
                return None
            embedding = data[0].get("embedding")
            if not isinstance(embedding, list):
                return None
            return [float(value) for value in embedding]
        except Exception:
            logging.exception("Failed to request memory embedding")
            return None


def build_embedding_function(configurable: Configuration) -> EmbeddingFunction:
    """Build embedding function from runtime configuration."""
    api_key = (
        configurable.memory_embedding_api_key
        or os.getenv("MEMORY_EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    client = EmbeddingClient(
        enabled=configurable.memory_embedding_enabled,
        provider=configurable.memory_embedding_provider,
        model=configurable.memory_embedding_model,
        base_url=configurable.memory_embedding_base_url,
        api_key=api_key,
    )
    return client.embed_text
