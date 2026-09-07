"""
Thin async wrapper around the Ollama HTTP API that adds:
  - native tool/function calling (Ollama's /api/chat "tools" field)
  - automatic fallback to a secondary model if the primary is unavailable
  - a dedicated vision call path for screen analysis
  - streaming text generation for low-latency TTS pipelining
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from core.config_loader import get_settings

logger = logging.getLogger("leti.llm_client")


class OllamaClient:
    def __init__(self):
        self.settings = get_settings()["ollama"]
        self.host = self.settings["host"].rstrip("/")
        self.timeout = self.settings["request_timeout_seconds"]
        self._client = httpx.AsyncClient(base_url=self.host, timeout=self.timeout)

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Reasoning / tool calling
    # ------------------------------------------------------------------ #
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request. Returns the raw Ollama response dict.
        If the primary model fails (connection error / 404 model not pulled),
        retries once against the configured fallback model.
        """
        chosen_model = model or self.settings["reasoning_model"]
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.settings.get("temperature", 0.3),
                "num_ctx": self.settings.get("num_ctx", 16384),
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.ConnectError) as e:
            fallback = self.settings.get("fallback_reasoning_model")
            if model is None and fallback and fallback != chosen_model:
                logger.warning(f"Primary model '{chosen_model}' failed ({e}); trying fallback '{fallback}'")
                payload["model"] = fallback
                resp = await self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields incremental chunks from Ollama's streaming chat endpoint."""
        chosen_model = model or self.settings["reasoning_model"]
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.settings.get("temperature", 0.3),
                "num_ctx": self.settings.get("num_ctx", 16384),
            },
        }
        if tools:
            payload["tools"] = tools

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                yield json.loads(line)

    # ------------------------------------------------------------------ #
    # Vision
    # ------------------------------------------------------------------ #
    async def analyze_image(self, prompt: str, image_base64: str) -> str:
        """Sends a base64-encoded screenshot + prompt to the vision model."""
        payload = {
            "model": self.settings["vision_model"],
            "messages": [
                {"role": "user", "content": prompt, "images": [image_base64]}
            ],
            "stream": False,
            "options": {"num_ctx": self.settings.get("num_ctx", 16384)},
        }
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    async def embed(self, text: str) -> List[float]:
        payload = {"model": self.settings["embedding_model"], "prompt": text}
        resp = await self._client.post("/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #
    async def is_available(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False
