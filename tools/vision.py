"""
Screen capture and multimodal analysis. Captures the primary display (or a
region), optionally downscales for faster inference, and asks the configured
Ollama vision model a question about what's on screen.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Optional

import mss
from PIL import Image

from core.config_loader import get_settings, resolve_path
from core.llm_client import OllamaClient
from tools.base import BaseTool, ToolParameter, ToolResult


class ScreenCapture:
    """Handles raw screenshotting + caching so repeated vision calls in a
    short window don't re-capture unnecessarily."""

    def __init__(self):
        self.settings = get_settings()["vision"]
        self._cache_path: Optional[Path] = None
        self._cache_time: float = 0.0

    def capture(self, region: Optional[dict] = None) -> Path:
        now = time.time()
        max_age = self.settings.get("max_screenshot_age_seconds", 5)
        if (
            region is None
            and self._cache_path
            and (now - self._cache_time) < max_age
            and self._cache_path.exists()
        ):
            return self._cache_path

        out_dir = resolve_path(self.settings["screenshot_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"screen_{int(now * 1000)}.png"

        with mss.mss() as sct:
            monitor = region if region else sct.monitors[1]  # monitor[0] is "all monitors"
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        max_width = self.settings.get("downscale_width")
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)))

        img.save(out_path, format="PNG")

        if region is None:
            self._cache_path = out_path
            self._cache_time = now

        return out_path

    @staticmethod
    def to_base64(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class ReadScreenTool(BaseTool):
    name = "read_screen"
    description = (
        "Capture the current screen and answer a question about what is visible "
        "(e.g. 'what error is shown', 'what app is open', 'summarize this page')."
    )
    parameters = [
        ToolParameter(
            name="question",
            type="string",
            description="What to look for or ask about the current screen contents.",
            required=True,
        ),
    ]

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self.capturer = ScreenCapture()

    async def run(self, question: str, **kwargs) -> ToolResult:
        try:
            path = self.capturer.capture()
            b64 = self.capturer.to_base64(path)
            answer = await self.llm_client.analyze_image(question, b64)
            return ToolResult(success=True, output={"answer": answer, "screenshot_path": str(path)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
