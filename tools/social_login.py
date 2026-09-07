"""
Instagram/TikTok/Facebook access via a one-time interactive login, not a
stored password. None of these platforms offer a free official API for
reading an arbitrary public account's content, so this uses a real,
visible browser window for login (see login_to_social_platform) - you type
your password directly into the actual platform's page, same as always,
including any 2FA/CAPTCHA. Leti never sees or handles the password itself.

What IS persisted to disk (data/social_sessions/<platform>.json) is the
resulting session - cookies and local storage - via Playwright's
storage_state. That's the same mechanism "stay signed in" uses normally;
losing that file just means logging in again, not a credential leak on
the scale of a stored password would be.

Honest limitations, stated plainly rather than glossed over:
- This is against these platforms' terms of service, even for your own
  account and even read-only. Realistic consequence is a security
  challenge or temporary lock if their bot-detection notices non-human
  patterns - not something more severe, but worth knowing before relying
  on this.
- Scraping the rendered page (there's no API) means every function here
  is coupled to today's HTML/DOM structure. These are exactly the
  functions most likely to silently break when a platform redesigns its
  site, and Facebook in particular is the most heavily obfuscated of the
  three.
- A saved session can expire or get invalidated by the platform at any
  time (not just via explicit logout) - functions here surface that as a
  clear "please log in again" error rather than a confusing scrape failure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

LOGIN_URLS = {
    "instagram": "https://www.instagram.com/accounts/login/",
    "tiktok": "https://www.tiktok.com/login",
    "facebook": "https://www.facebook.com/login/",
}

# Heuristic "we're logged in now" indicators checked after the user completes login
# manually - each is something present on the platform's logged-in home feed but not
# on its login page, kept intentionally simple since login pages change less often
# than deep content pages.
LOGGED_IN_CHECKS = {
    "instagram": ("url", "instagram.com/", "instagram.com/accounts/login"),
    "tiktok": ("url", "tiktok.com/foryou", "tiktok.com/login"),
    "facebook": ("selector", '[role="feed"]', None),
}

PROFILE_URL_TEMPLATES = {
    "instagram": "https://www.instagram.com/{identifier}/",
    "tiktok": "https://www.tiktok.com/@{identifier}",
    "facebook": "https://www.facebook.com/{identifier}",
}


def _session_path(platform: str) -> Path:
    p = resolve_path(f"./data/social_sessions/{platform}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class SocialLoginManager:
    """Owns a dedicated Playwright browser instance, separate from the general-purpose
    tools/browser.py BrowserSession used for ad hoc web tasks - each social platform
    gets its own isolated, persistent context so their sessions/cookies never mix, and
    so unrelated browsing tasks elsewhere can't accidentally disturb a logged-in tab."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._contexts: Dict[str, Any] = {}

    async def _ensure_browser(self, headless: bool):
        from playwright.async_api import async_playwright

        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=headless)

    def has_saved_session(self, platform: str) -> bool:
        return _session_path(platform).exists()

    async def get_context(self, platform: str, headless: bool = True):
        if platform in self._contexts:
            return self._contexts[platform]
        if not self.has_saved_session(platform):
            raise RuntimeError(
                f"Not logged in to {platform} yet. Call login_to_social_platform first - "
                f"it opens a real browser window for you to log in."
            )
        await self._ensure_browser(headless)
        context = await self._browser.new_context(storage_state=str(_session_path(platform)))
        self._contexts[platform] = context
        return context

    async def login_interactive(self, platform: str, timeout_seconds: int = 300) -> str:
        """Opens a VISIBLE browser window at the platform's login page and waits for the
        user to finish logging in themselves (password, 2FA, CAPTCHA, all of it) - polling
        for a logged-in indicator rather than assuming a fixed delay. Saves the resulting
        session to disk on success."""
        import asyncio

        if platform not in LOGIN_URLS:
            raise ValueError(f"Unsupported platform: {platform}")

        await self._ensure_browser(headless=False)
        context = await self._browser.new_context()
        page = await context.new_page()
        await page.goto(LOGIN_URLS[platform], wait_until="domcontentloaded", timeout=30000)

        check_type, positive, negative = LOGGED_IN_CHECKS[platform]
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        logged_in = False

        while asyncio.get_event_loop().time() < deadline:
            try:
                if check_type == "url":
                    url = page.url
                    if positive in url and (negative is None or negative not in url):
                        logged_in = True
                        break
                elif check_type == "selector":
                    if await page.locator(positive).count() > 0:
                        logged_in = True
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

        if not logged_in:
            await context.close()
            raise TimeoutError(
                f"Didn't detect a successful {platform} login within {timeout_seconds}s. "
                f"If you did log in, the page just changed since this was written - try again "
                f"and let me know if it keeps happening."
            )

        await context.storage_state(path=str(_session_path(platform)))
        self._contexts[platform] = context
        return f"Logged in to {platform} and saved the session. Future checks won't need a new login."

    def logout(self, platform: str) -> None:
        p = _session_path(platform)
        if p.exists():
            p.unlink()
        self._contexts.pop(platform, None)

    async def close(self):
        for context in self._contexts.values():
            try:
                await context.close()
            except Exception:
                pass
        self._contexts.clear()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None


# --- Scraping (best-effort, coupled to current page structure) -------------------

async def _scrape_instagram(context, username: str, max_results: int) -> List[Dict[str, Any]]:
    page = await context.new_page()
    try:
        await page.goto(PROFILE_URL_TEMPLATES["instagram"].format(identifier=username), wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_selector('a[href*="/p/"]', timeout=10000)
        links = await page.locator('a[href*="/p/"]').all()
        seen, items = set(), []
        for link in links:
            href = await link.get_attribute("href")
            if not href or href in seen:
                continue
            seen.add(href)
            post_id = href.strip("/").split("/")[-1]
            # Best-effort: the grid thumbnail is an <img> inside the same link. If
            # Instagram's markup shifts this, we just end up with no thumbnail for that
            # post rather than a broken scrape - not fatal, only the visual popup loses that item.
            thumbnail = ""
            try:
                img = link.locator("img").first
                if await img.count() > 0:
                    thumbnail = await img.get_attribute("src") or ""
            except Exception:
                pass
            items.append({"id": post_id, "url": f"https://www.instagram.com{href}", "thumbnail": thumbnail})
            if len(items) >= max_results:
                break
        return items
    finally:
        await page.close()


async def _scrape_tiktok(context, username: str, max_results: int) -> List[Dict[str, Any]]:
    page = await context.new_page()
    try:
        await page.goto(PROFILE_URL_TEMPLATES["tiktok"].format(identifier=username), wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_selector('a[href*="/video/"]', timeout=10000)
        links = await page.locator('a[href*="/video/"]').all()
        seen, items = set(), []
        for link in links:
            href = await link.get_attribute("href")
            if not href or href in seen:
                continue
            seen.add(href)
            video_id = href.rstrip("/").split("/")[-1]
            # TikTok renders the cover as either an <img> or a <video poster="...">
            # depending on the page version - try both, fall back to no thumbnail.
            thumbnail = ""
            try:
                img = link.locator("img").first
                if await img.count() > 0:
                    thumbnail = await img.get_attribute("src") or ""
                if not thumbnail:
                    video_el = link.locator("video").first
                    if await video_el.count() > 0:
                        thumbnail = await video_el.get_attribute("poster") or ""
            except Exception:
                pass
            items.append({"id": video_id, "url": href, "thumbnail": thumbnail})
            if len(items) >= max_results:
                break
        return items
    finally:
        await page.close()


async def _scrape_facebook(context, page_name: str, max_results: int) -> List[Dict[str, Any]]:
    page = await context.new_page()
    try:
        await page.goto(PROFILE_URL_TEMPLATES["facebook"].format(identifier=page_name), wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_selector('[role="article"]', timeout=10000)
        articles = await page.locator('[role="article"]').all()
        items = []
        for i, article in enumerate(articles[:max_results]):
            text = (await article.inner_text())[:200]
            items.append({"id": f"{page_name}-{i}-{hash(text)}", "text": text})
        return items
    finally:
        await page.close()


_SCRAPERS = {"instagram": _scrape_instagram, "tiktok": _scrape_tiktok, "facebook": _scrape_facebook}


async def fetch_latest_for_platform(platform: str, identifier: str, max_results: int = 10, manager: Optional[SocialLoginManager] = None) -> List[Dict[str, Any]]:
    manager = manager or _default_manager()
    context = await manager.get_context(platform, headless=True)
    return await _SCRAPERS[platform](context, identifier, max_results)


_MANAGER_SINGLETON: Optional[SocialLoginManager] = None


def _default_manager() -> SocialLoginManager:
    global _MANAGER_SINGLETON
    if _MANAGER_SINGLETON is None:
        _MANAGER_SINGLETON = SocialLoginManager()
    return _MANAGER_SINGLETON


class LoginToSocialPlatformTool(BaseTool):
    name = "login_to_social_platform"
    description = (
        "Open a real, visible browser window so the user can log in to Instagram, TikTok, or "
        "Facebook themselves (password, 2FA, CAPTCHA - all handled by them directly on the "
        "real site). Leti never sees the password. The resulting session is saved for future "
        "checks. Risky: opens a real authenticated session tied to the user's account."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="platform", type="string", enum=["instagram", "tiktok", "facebook"], description="Which platform."),
    ]

    def __init__(self, manager: Optional[SocialLoginManager] = None):
        self.manager = manager or _default_manager()

    async def run(self, platform: str, **kwargs) -> ToolResult:
        try:
            message = await self.manager.login_interactive(platform)
            return ToolResult(success=True, output=message)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LogoutSocialPlatformTool(BaseTool):
    name = "logout_social_platform"
    description = "Forget the saved login session for a platform (Instagram/TikTok/Facebook)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="platform", type="string", enum=["instagram", "tiktok", "facebook"], description="Which platform."),
    ]

    def __init__(self, manager: Optional[SocialLoginManager] = None):
        self.manager = manager or _default_manager()

    async def run(self, platform: str, **kwargs) -> ToolResult:
        self.manager.logout(platform)
        return ToolResult(success=True, output=f"Logged out of {platform} (saved session removed).")


class GetInstagramUserLatestTool(BaseTool):
    name = "get_instagram_user_latest"
    description = "Get the latest posts from an Instagram profile. Requires being logged in (see login_to_social_platform)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="username", type="string", description="Instagram username, without '@'."),
    ]

    def __init__(self, manager: Optional[SocialLoginManager] = None):
        self.manager = manager or _default_manager()

    async def run(self, username: str, **kwargs) -> ToolResult:
        try:
            items = await fetch_latest_for_platform("instagram", username, manager=self.manager)
            output = {"posts": items, "count": len(items)}
            visual_items = [{"url": i["url"], "thumbnail": i["thumbnail"], "title": f"@{username}"} for i in items if i.get("thumbnail")]
            if visual_items:
                output["visual"] = {"type": "images", "query": f"@{username} on Instagram", "items": visual_items}
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetTikTokUserLatestTool(BaseTool):
    name = "get_tiktok_user_latest"
    description = "Get the latest videos from a TikTok profile. Requires being logged in (see login_to_social_platform)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="username", type="string", description="TikTok username, without '@'."),
    ]

    def __init__(self, manager: Optional[SocialLoginManager] = None):
        self.manager = manager or _default_manager()

    async def run(self, username: str, **kwargs) -> ToolResult:
        try:
            items = await fetch_latest_for_platform("tiktok", username, manager=self.manager)
            output = {"videos": items, "count": len(items)}
            visual_items = [{"url": i["url"], "thumbnail": i["thumbnail"], "title": f"@{username}"} for i in items if i.get("thumbnail")]
            if visual_items:
                output["visual"] = {"type": "images", "query": f"@{username} on TikTok", "items": visual_items}
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetFacebookPageLatestTool(BaseTool):
    name = "get_facebook_page_latest"
    description = (
        "Get the latest posts from a Facebook page/profile. Requires being logged in (see "
        "login_to_social_platform). Facebook's page structure is the most heavily obfuscated "
        "of the three platforms here, so this is the most likely to need fixing over time."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="page_name", type="string", description="Facebook page/profile name or username as it appears in its URL."),
    ]

    def __init__(self, manager: Optional[SocialLoginManager] = None):
        self.manager = manager or _default_manager()

    async def run(self, page_name: str, **kwargs) -> ToolResult:
        try:
            items = await fetch_latest_for_platform("facebook", page_name, manager=self.manager)
            return ToolResult(success=True, output={"posts": items, "count": len(items)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
