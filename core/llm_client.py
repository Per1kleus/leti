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

# Why a streamed answer ended. Carried on the final event next to the free-text
# `error`, so a caller can say the true thing rather than one sentence for
# everything. See stream_response.
REASON_OK = ""                              # it finished
REASON_STOPPED = "stopped"                  # the user said stop
REASON_MODEL_ERROR = "model_error"          # the server said what was wrong
REASON_UNREACHABLE = "unreachable"          # nothing is listening
REASON_TIMED_OUT = "timed_out"              # no bytes within the timeout
REASON_CUT_OFF = "cut_off"                  # bytes, then the connection died
REASON_NO_ANSWER = "no_answer"              # connected, and nothing came back
REASON_NO_MARKER = "no_completion_marker"   # a clean close with no done flag


class reasons:
    """The same codes, gathered so a caller imports one name instead of eight.

    Not an Enum: the value on the event is a plain string so that a dict of events
    stays JSON, which is what the GUI transport and the diagnostics record carry.
    """

    OK = REASON_OK
    STOPPED = REASON_STOPPED
    MODEL_ERROR = REASON_MODEL_ERROR
    UNREACHABLE = REASON_UNREACHABLE
    TIMED_OUT = REASON_TIMED_OUT
    CUT_OFF = REASON_CUT_OFF
    NO_ANSWER = REASON_NO_ANSWER
    NO_MARKER = REASON_NO_MARKER

    ALL = (OK, STOPPED, MODEL_ERROR, UNREACHABLE, TIMED_OUT, CUT_OFF,
           NO_ANSWER, NO_MARKER)


class OllamaClient:
    def __init__(self):
        cfg = get_settings()["ollama"]
        # host and timeout are baked into the HTTP client, so unlike the rest of this
        # section they genuinely do need a restart to change.
        self.host = cfg["host"].rstrip("/")
        self.timeout = cfg["request_timeout_seconds"]
        # How long to wait for the FIRST byte of a streamed answer, which is a
        # different question from how long to wait between bytes once it is
        # flowing. httpx has one read timeout and it applies to both, so passing
        # request_timeout_seconds meant the gap while the model reads the prompt
        # was measured against a number chosen for a whole request. A full-fallback
        # turn is about 22,300 tokens of tool schemas (see core/tool_router.py) and
        # the model has to read all of them before it can write anything; on a
        # machine where Ollama has put layers on the CPU that is minutes, the read
        # timed out with nothing on the socket, and the user was told the
        # connection had dropped partway through an answer that had never started.
        #
        # So the streamed path gets its own, longer allowance - read live in
        # _stream_timeout rather than baked in here, like everything else in
        # `settings`. The cost is that a generation which genuinely stalls takes
        # this long to notice; that is the right way round, because a slow prefill
        # is ordinary and a stalled generation is not.
        self._client = httpx.AsyncClient(
            base_url=self.host,
            # Connecting to something on this machine either works at once or is
            # not going to. Spending the full request budget on it only delays
            # telling the user that Ollama is not running.
            timeout=httpx.Timeout(float(self.timeout),
                                  connect=min(10.0, float(self.timeout))),
        )

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

        The final event also carries `reason`, one of the REASON_* codes below, and
        `started`, which is whether any text arrived at all. Both exist because
        `error` alone was a free-text string and the caller had nothing to tell the
        cases apart with, so it said one sentence for all of them: "the connection
        to the model stopped partway through". Measured against a real server, seven
        distinct situations produced that sentence, and in five of them nothing had
        arrived - so nothing was partway through anything:

          normal              3 fragments, no error
          long answer       120 fragments, no error
          slow prefill        0 fragments, ReadTimeout
          model not pulled    0 fragments, HTTP 404 - and Ollama had said
                              'model "ghost" not found, try pulling it first',
                              which this code read and threw away
          out of memory       0 fragments, an {"error": ...} line in the stream
                              naming the exact shortfall, also thrown away
          cut mid-answer      2 fragments, then the peer closed
          no done marker      1 fragment, clean close

        Only the sixth is a connection stopping partway through. The fourth and
        fifth are the model server explaining precisely what is wrong, in text this
        function used to discard - which is why "install the model" looked like a
        network fault. Nothing here is reported less loudly than before; each case
        is reported as what it is.
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
        reason = REASON_OK

        try:
            async with self._client.stream("POST", "/api/chat", json=payload,
                                           timeout=self._stream_timeout()) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as bad:
                    # raise_for_status is still what detects it - anything standing
                    # in for a response knows that method. What is new is reading
                    # the body afterwards: on a streaming response it has not been
                    # read, and Ollama puts the real explanation in it ('model "x"
                    # not found, try pulling it first'). The exception on its own
                    # carries the status code and throws the instruction away.
                    raise _ServerSaid(await _explain_status(bad.response)) from bad
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
                    said = chunk.get("error")
                    if isinstance(said, str) and said.strip():
                        # The model server explaining itself mid-stream: out of
                        # memory, a model that failed to load, a template error.
                        # Ollama sends this instead of a done marker, so without
                        # this branch it fell through to "ended without a
                        # completion marker" and the explanation was lost.
                        raise _ServerSaid(said.strip())
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
        except _ServerSaid as e:
            error = str(e)
            reason = REASON_MODEL_ERROR
            logger.warning(f"The model server refused the request: {error}")
        except Exception as e:
            # Includes the connection dying and a read timing out. The text that
            # arrived is still the text that arrived.
            error = f"{type(e).__name__}: {e}"
            reason = _reason_for(e, bool(content))
            logger.warning(f"Streaming from Ollama stopped early ({error}).")

        if not finished and not error and not stopped:
            # The body ended without a done marker. Not fatal - the answer may be
            # complete - but the caller is told rather than left to assume.
            error = "the response ended without a completion marker"
            reason = REASON_NO_MARKER
            logger.warning(error)

        if stopped:
            reason = REASON_STOPPED

        yield {"done": True, "stopped": stopped, "error": error, "reason": reason,
               # Whether the model wrote anything at all. "Partway through" is only
               # true when this is true, and it was false in five of the seven
               # measured failures.
               "started": bool(content),
               "message": _assembled(role, content, tool_calls)}

    def _stream_timeout(self) -> Any:
        """The timeout for one streamed request. See the note in __init__.

        Read from the settings on every call, so raising it does not need a
        restart - the same as the model and the temperature, and unlike the host.
        """
        base = float(self.timeout or 120)
        try:
            configured = self.settings.get("stream_timeout_seconds")
        except Exception:
            configured = None
        return httpx.Timeout(float(configured) if configured else max(base, 600.0),
                             connect=min(10.0, base))

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
    # A health check is a different question from a request, and it gets a
    # different budget. Asking "is anything there" with the request timeout meant
    # that a server which accepts the connection and then says nothing - which is
    # exactly what Ollama looks like while it loads a model - held the answer for
    # up to request_timeout_seconds. Measured at the shipped 120s, against a
    # server that accepts and stalls: still waiting at 8s, and it would have kept
    # waiting for two minutes.
    HEALTH_TIMEOUT_SECONDS = 3.0

    async def is_available(self) -> bool:
        """Whether the model server answers at all, quickly and without raising.

        It used to catch httpx.ConnectError alone. A read timing out is a
        ReadTimeout, which is not a ConnectError, so it went straight past this
        and out of main.py's build_app - where nothing catches it either, so the
        process died with a traceback instead of reporting that Ollama was busy.
        Every failure is "no" here; which failure it was is `probe` below.
        """
        try:
            resp = await self._client.get("/api/tags",
                                          timeout=self.HEALTH_TIMEOUT_SECONDS)
            return resp.status_code == 200
        except Exception:
            return False

    async def probe(self) -> Dict[str, Any]:
        """What state the model server is actually in, for the diagnostics panel.

        One call, one honest answer, in the vocabulary the interface shows. Kept
        here beside the client that makes the request rather than in a second
        module that would have to know the host, the timeout and the shape of the
        reply all over again.
        """
        try:
            resp = await self._client.get("/api/tags",
                                          timeout=self.HEALTH_TIMEOUT_SECONDS)
        except httpx.ConnectError:
            return {"state": "not_running", "detail": "Nothing is listening at "
                                                      f"{self.host}.", "models": []}
        except httpx.TimeoutException:
            # Listening, not answering. Ordinary while it starts or loads a model,
            # and a real answer rather than "not running" - telling somebody to
            # start a server that is already running wastes their time.
            return {"state": "busy",
                    "detail": "The model server is running but did not answer in "
                              f"{self.HEALTH_TIMEOUT_SECONDS:.0f} seconds - it is "
                              "probably still starting or loading a model.",
                    "models": []}
        except Exception as e:
            return {"state": "error", "detail": f"{type(e).__name__}: {e}", "models": []}

        if resp.status_code != 200:
            return {"state": "error",
                    "detail": f"The model server answered HTTP {resp.status_code}.",
                    "models": []}
        try:
            models = [m.get("name", "") for m in resp.json().get("models", [])
                      if isinstance(m, dict) and m.get("name")]
        except Exception:
            models = []
        return {"state": "connected", "detail": "", "models": models}


class _ServerSaid(Exception):
    """The model server explained what was wrong. Its words, not a guess at them.

    A private exception rather than a flag because it can be raised from two
    places - a non-200 status and an {"error": ...} line mid-stream - and both
    have to leave the `async with` so the socket closes before anything is
    yielded, exactly as the stop path does.
    """


async def _explain_status(response: Any) -> str:
    """What a non-200 from the model server actually said.

    Ollama answers a request for a model that was never pulled with 404 and
    {"error": 'model "x" not found, try pulling it first'} - an instruction the
    user can act on. Falls back to the status code when there is no body to read
    or nothing useful in it, which is still better than a bare exception name.
    """
    detail = ""
    try:
        body = await response.aread()
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            detail = parsed["error"].strip()
    except Exception:
        detail = ""
    code = getattr(response, "status_code", "?")
    return detail or f"the model server answered HTTP {code}"


def _reason_for(exc: Exception, any_text: bool) -> str:
    """Which REASON_* an exception is. Reported as what it is, not as one thing."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return REASON_UNREACHABLE
    if isinstance(exc, httpx.TimeoutException):
        return REASON_TIMED_OUT
    # A protocol error, a reset, anything else: cut off if text had arrived, and
    # if none had then the answer never started rather than stopped partway. The
    # connection itself was fine in that case - something went wrong behind it -
    # so it is not reported as nothing listening either.
    return REASON_CUT_OFF if any_text else REASON_NO_ANSWER


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
