"""A single, shared reader for terminal input.

Why this exists rather than calling input() wherever a line is needed:
input() runs in a thread executor, and asyncio.wait_for CANNOT cancel a thread.
When SafetyGuard's confirmation prompt timed out, the executor thread stayed
blocked inside input() holding stdin - so the next line the user typed was
swallowed by that abandoned prompt instead of reaching the chat loop, and every
subsequent timeout stranded another thread competing for the same stdin.

One daemon thread owns stdin for the life of the process and publishes each line
to whoever is currently waiting. A timeout then just stops waiting; the reader
thread is unaffected and the next caller gets the next line. Nothing is left
holding the terminal.

Every terminal read in the app goes through here - the chat loop, the settings
editor, and confirmations - because two stdin consumers would race for lines.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from typing import Optional

EOF = object()
"""Sentinel published when stdin closes (Ctrl-D, or a piped script running out)."""


class _ConsoleReader:
    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> asyncio.Queue:
        """Bind to the running loop and start the reader thread, once."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._queue is not None and self._loop is loop:
                return self._queue

            self._loop = loop
            self._queue = asyncio.Queue()
            queue = self._queue

            def publish(item) -> None:
                # Hop back to the loop thread; Queue is not thread-safe.
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:
                    pass  # loop closed during shutdown - nothing left to notify

            def pump() -> None:
                for line in sys.stdin:
                    publish(line.rstrip("\n"))
                publish(EOF)

            self._thread = threading.Thread(target=pump, name="leti-stdin", daemon=True)
            self._thread.start()
            return queue

    def drain(self) -> None:
        """Discard anything typed before the current prompt was shown.

        Called before prompting so a line the user typed ahead - or an answer to
        a prompt that already timed out - isn't silently accepted as the answer
        to this one.
        """
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def read_line(
        self, prompt: str = "", timeout: Optional[float] = None, drain_stale: bool = True
    ) -> Optional[str]:
        """Show `prompt` and wait for one line.

        Returns the line, or None on timeout or end of input. A timeout leaves no
        thread stranded - it only stops this coroutine waiting.
        """
        queue = self._ensure_started()
        if drain_stale:
            self.drain()
        if prompt:
            print(prompt, end="", flush=True)

        try:
            if timeout is None:
                item = await queue.get()
            else:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            print()  # close the dangling prompt line so later output isn't appended to it
            return None

        if item is EOF:
            return None
        return item


_reader = _ConsoleReader()


async def read_line(
    prompt: str = "", timeout: Optional[float] = None, drain_stale: bool = True
) -> Optional[str]:
    """Read one line from the terminal. See _ConsoleReader.read_line."""
    return await _reader.read_line(prompt, timeout=timeout, drain_stale=drain_stale)


def drain() -> None:
    """Discard buffered/typed-ahead input. See _ConsoleReader.drain."""
    _reader.drain()
