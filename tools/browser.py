"""
Browser automation via Playwright. Keeps a single persistent browser/page
across calls within a session so multi-step web tasks (navigate -> click ->
fill) share state, and closes cleanly on shutdown.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, Page, Playwright

from tools.base import BaseTool, ToolParameter, ToolResult


class BrowserSession:
    """Lazily starts a Chromium instance and keeps one active page.

    Two ways in, one browser. get_page() returns the visible page the browser_*
    tools drive, where the user can watch and take over. read_text() fetches page
    text in a headless context instead - research reads a dozen pages at once, and
    doing that through the visible page would hijack the window the user is looking
    at and be sequential besides. Both share this object's browser process, so
    research isn't a second browser stack bolted alongside the first.
    """

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._headless_context = None
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

    async def _ensure_headless_context(self):
        """A background context for reading pages, separate from the visible page."""
        async with self._lock:
            if self._headless_context is None:
                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                if self._browser is None:
                    self._browser = await self._playwright.chromium.launch(headless=False)
                # Its own context: research browsing shouldn't inherit or pollute
                # whatever session the user has open in the visible page.
                self._headless_context = await self._browser.new_context()
            return self._headless_context

    async def read_text(self, url: str, timeout_ms: int = 20000, max_characters: int = 6000) -> Dict[str, Any]:
        """Fetch one page's visible text. Returns a dict with ok/url/title/text/error
        rather than raising: research reads many pages and one that fails shouldn't
        abandon the rest."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        context = await self._ensure_headless_context()
        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # A 404 or 500 still renders a page with text on it. Without this check
            # the error boilerplate comes back as a successfully read source, which
            # is worse than a failure: research would cite it.
            if response is not None and response.status >= 400:
                return {
                    "ok": False, "url": url, "title": "", "text": "",
                    "error": f"HTTP {response.status} {response.status_text}".strip(),
                }
            title = await page.title()
            text = await page.locator("body").inner_text(timeout=10000)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            return {
                "ok": True, "url": page.url, "title": title,
                "text": text[:max_characters], "truncated": len(text) > max_characters,
            }
        except Exception as e:
            return {"ok": False, "url": url, "title": "", "text": "", "error": str(e)}
        finally:
            await page.close()

    async def read_many(self, urls: List[str], concurrency: int = 4, **kwargs) -> List[Dict[str, Any]]:
        """Read several pages at once. Sequential reads are what make research feel
        slow; the cap keeps a wide search from opening thirty tabs."""
        semaphore = asyncio.Semaphore(concurrency)

        async def one(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.read_text(url, **kwargs)

        return list(await asyncio.gather(*(one(u) for u in urls)))

    async def close(self):
        if self._headless_context:
            try:
                await self._headless_context.close()
            except Exception:
                pass
            self._headless_context = None
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None


_SHARED_SESSION: Optional[BrowserSession] = None


def get_shared_session() -> BrowserSession:
    """The process-wide BrowserSession.

    main.py constructs one and passes it to the browser tools; set_shared_session
    registers that same object here so modules which need page text (webpage
    watches, research) reach the one browser rather than starting another.
    """
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        _SHARED_SESSION = BrowserSession()
    return _SHARED_SESSION


def set_shared_session(session: BrowserSession) -> None:
    global _SHARED_SESSION
    _SHARED_SESSION = session


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
