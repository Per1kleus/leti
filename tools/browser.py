"""
Browser automation via Playwright. Keeps a single persistent browser/page
across calls within a session so multi-step web tasks (navigate -> click ->
fill) share state, and closes cleanly on shutdown.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, Playwright

from tools.base import BaseTool, ToolParameter, ToolResult


class BrowserSession:
    """Lazily starts a Chromium instance and keeps one active page."""

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._lock = asyncio.Lock()

    async def get_page(self) -> Page:
        # Launching is several awaits long. Without the lock, two tool calls that
        # both find _page is None start two Chromium instances, and the first is
        # orphaned - never closed, still on screen.
        if self._page is not None:
            return self._page
        async with self._lock:
            if self._page is None:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=False)
                self._page = await self._browser.new_page()
            return self._page

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None


class BrowserNavigateTool(BaseTool):
    name = "browser_navigate"
    description = "Open a URL in the browser."
    parameters = [
        ToolParameter(name="url", type="string", description="Full URL to navigate to."),
    ]

    def __init__(self, session: BrowserSession):
        self.session = session

    async def run(self, url: str, **kwargs) -> ToolResult:
        try:
            page = await self.session.get_page()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            title = await page.title()
            return ToolResult(success=True, output=f"Navigated to '{title}' ({url})")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserReadPageTool(BaseTool):
    name = "browser_read_page"
    description = (
        "Read the visible text of a web page so you can answer from what it actually says. "
        "Pass a url to open it first, or omit url to read the page already open. Use this "
        "after web_search when a result's snippet isn't enough - search gives you links and "
        "summaries, this gives you the page itself."
    )
    parameters = [
        ToolParameter(
            name="url", type="string", required=False,
            description="Page to open and read. Omit to read the currently open page.",
        ),
        ToolParameter(
            name="max_characters", type="number", required=False,
            description="Truncate the text at this many characters (default 8000).",
        ),
    ]

    def __init__(self, session: BrowserSession):
        self.session = session

    async def run(self, url: str = "", max_characters: int = 8000, **kwargs) -> ToolResult:
        try:
            page = await self.session.get_page()
            if url:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            title = await page.title()
            # inner_text() of <body>, not content(): the rendered text a person would
            # read, without the markup, scripts and styling that would otherwise eat
            # the model's context for no benefit.
            text = await page.locator("body").inner_text(timeout=10000)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            max_characters = int(max_characters) if max_characters else 8000
            truncated = len(text) > max_characters
            if truncated:
                text = text[:max_characters]

            return ToolResult(success=True, output={
                "title": title,
                "url": page.url,
                "text": text,
                "truncated": truncated,
            })
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserClickTool(BaseTool):
    name = "browser_click"
    description = "Click an element on the current page identified by visible text or a CSS selector."
    parameters = [
        ToolParameter(name="selector_or_text", type="string", description="CSS selector or visible text to click."),
    ]

    def __init__(self, session: BrowserSession):
        self.session = session

    async def run(self, selector_or_text: str, **kwargs) -> ToolResult:
        try:
            page = await self.session.get_page()
            try:
                await page.click(selector_or_text, timeout=5000)
            except Exception:
                await page.get_by_text(selector_or_text, exact=False).first.click(timeout=5000)
            return ToolResult(success=True, output=f"Clicked '{selector_or_text}'")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BrowserFillFormTool(BaseTool):
    name = "browser_fill_form"
    description = "Fill a form field on the current page identified by a CSS selector."
    parameters = [
        ToolParameter(name="selector", type="string", description="CSS selector of the input field."),
        ToolParameter(name="value", type="string", description="Text to enter into the field."),
    ]

    def __init__(self, session: BrowserSession):
        self.session = session

    async def run(self, selector: str, value: str, **kwargs) -> ToolResult:
        try:
            page = await self.session.get_page()
            await page.fill(selector, value, timeout=5000)
            return ToolResult(success=True, output=f"Filled '{selector}' with provided value.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
