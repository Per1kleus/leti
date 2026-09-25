"""
Thin async wrapper around the Ollama HTTP API that adds:
  - native tool/function calling (Ollama's /api/chat "tools" field)
  - automatic fallback to a secondary model if the primary is unavailable
  - a dedicated vision call path for screen analysis
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from core.config_loader import get_settings

logger = logging.getLogger("leti.llm_client")


class OllamaClient:
    def __init__(self):
        cfg = get_settings()["ollama"]
        # host and timeout are baked into the HTTP client, so unlike the rest of this
        # section they genuinely do need a restart to change.
        self.host = cfg["host"].rstrip("/")
        self.timeout = cfg["request_timeout_seconds"]
        self._client = httpx.AsyncClient(base_url=self.host, timeout=self.timeout)

    @property
    def settings(self) -> Dict[str, Any]:
        """Live, so a model or temperature change applies without a restart."""
        return get_settings()["ollama"]

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

    # ------------------------------------------------------------------ #
    # Streaming
    #
    # chat() above asks for a whole message and waits for it. That is still the
    # right shape for a tool call, a summary, anything a caller needs in one
    # piece - and every existing caller keeps it, unchanged.
    #
    # This is the other shape, for the one place it matters: a spoken answer.
    # Waiting for the last token before saying the first word means silence for
    # as long as the model takes, and the answer is the same either way. Ollama
    # streams it as NDJSON - one JSON object per line - so the text arrives in
    # fragments and the voice can start on the first finished sentence.
    #
    # What comes out is a sequence of events rather than a string, because the
    # caller needs two different things: the fragments, as they arrive, and the
    # assembled message at the end (which carries the tool calls). Yielding both
    # means nothing has to buffer the answer twice.
    # ------------------------------------------------------------------ #

    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        should_stop: Optional[Any] = None,
    ) -> Any:
        """Yields {"text": fragment} as the model writes, then {"done", "message"}.

        `should_stop` is a callable checked before every fragment is handed over.
        When it returns true the response is closed and the generator finishes -
        cooperatively, with no thread to kill and no socket left half-read. The
        caller gets a final event carrying what had arrived, marked `stopped`.

        Never raises for a chunk it cannot read. A malformed line is skipped, a
        chunk with no text in it is simply not text, and a connection that dies
        mid-answer ends the stream with `error` set and whatever arrived intact -
        because a partial answer that says it is partial is worth more than an
        exception on top of text the user already heard.
        """
        payload = {
            "model": model or self.settings["reasoning_model"],
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.settings.get("temperature", 0.3),
                "num_ctx": self.settings.get("num_ctx", 16384),
            },
        }
        if tools:
            payload["tools"] = tools

        content: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        role = "assistant"
        finished = False
        stopped = False
        error = ""

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if should_stop is not None and should_stop():
                        # Break rather than yield from in here. Leaving the
                        # `async with` is what closes the response and cancels
                        # the generation - and a generator suspended at a yield
                        # inside it is never resumed once the caller stops
                        # iterating, so the socket would stay open until the
                        # garbage collector noticed. The event is yielded below,
                        # after the body is closed.
                        stopped = True
                        break
                    chunk = _read_chunk(line)
                    if chunk is None:
                        continue
                    message = chunk.get("message")
                    if isinstance(message, dict):
                        role = message.get("role") or role
                        calls = message.get("tool_calls")
                        if isinstance(calls, list) and calls:
                            tool_calls.extend(c for c in calls if isinstance(c, dict))
                        fragment = message.get("content")
                        if isinstance(fragment, str) and fragment:
                            # Some servers repeat the whole answer in the final
                            # chunk. Appending it would say everything twice, so
                            # a fragment that IS what we already have is dropped.
                            if chunk.get("done") and "".join(content) == fragment:
                                pass
                            else:
                                content.append(fragment)
                                yield {"text": fragment}
                    if chunk.get("done"):
                        finished = True
                        break
        except Exception as e:
            # Includes the connection dying, a non-200, and a read timing out.
            # The text that arrived is still the text that arrived.
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"Streaming from Ollama stopped early ({error}).")

        if not finished and not error and not stopped:
            # The body ended without a done marker. Not fatal - the answer may be
            # complete - but the caller is told rather than left to assume.
            error = "the response ended without a completion marker"
            logger.warning(error)

        yield {"done": True, "stopped": stopped, "error": error,
               "message": _assembled(role, content, tool_calls)}

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


def _read_chunk(line: str) -> Optional[Dict[str, Any]]:
    """One NDJSON line as a dict, or None for anything unreadable.

    Blank lines, keep-alives and half-written objects all read as None. A stream
    is not a document: one bad line is a line to skip, never a reason to throw
    away the answer around it.
    """
    if not line or not line.strip():
        return None
    try:
        chunk = json.loads(line)
    except (ValueError, TypeError):
        logger.debug("Skipped an unreadable line from the model stream.")
        return None
    return chunk if isinstance(chunk, dict) else None


def _assembled(role: str, content: List[str], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The fragments as the one message chat() would have returned.

    Joined in arrival order and exactly once, so the streamed answer and the
    whole-response answer are the same string for the same model output.
    """
    message: Dict[str, Any] = {"role": role, "content": "".join(content)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
