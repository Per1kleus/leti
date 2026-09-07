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
