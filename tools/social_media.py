"""
Social media inspection: YouTube and Reddit via their free, official,
no-login-required read APIs/endpoints, plus a generic persistent "watch"
system that works across every platform this project supports (these two,
plus Instagram/TikTok/Facebook via tools/social_login.py's authenticated
scraping - imported lazily below so this module still works standalone if
that one's Playwright dependency isn't needed).

Watch semantics ("remind me independent of time passed"): each watch
stores last_seen_id, the identifier of the most recent item seen as of the
last check - not a timestamp window. Checking a week-old watch or a
five-minute-old one behaves identically: everything newer than
last_seen_id is reported, however much of a backlog that is. The very
first check on a new watch establishes the baseline silently (no "new!"
spam for content that already existed before you asked to watch it) -
only checks after that report anything.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

DEFAULT_REDDIT_USER_AGENT = "leti-personal-assistant/1.0 (by /u/leti-user)"
# "webpage" extends the watch mechanism to any URL rather than adding a second
# monitoring system: add/list/remove/check, the last_seen_id diffing and the
# stored state are all the same machinery, and a page is just another source
# whose "latest item" is its current content.
WATCHABLE_PLATFORMS = [
    "youtube", "reddit_user", "reddit_subreddit", "instagram", "tiktok", "facebook", "webpage",
]


# --- Reddit (public JSON, no login required) ------------------------------------

def _reddit_settings() -> Dict[str, Any]:
    return get_settings().get("social_media", {}).get("reddit", {})


async def _reddit_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ua = _reddit_settings().get("user_agent", DEFAULT_REDDIT_USER_AGENT)
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": ua}) as client:
        resp = await client.get(f"https://www.reddit.com{path}.json", params=params or {})
        resp.raise_for_status()
        return resp.json()


def _reddit_post_summary(child: Dict[str, Any]) -> Dict[str, Any]:
    d = child["data"]
    thumb = d.get("thumbnail", "")
    return {
        "id": d["id"],
        "title": d["title"],
        "subreddit": d["subreddit"],
        "author": d.get("author", "[deleted]"),
        "score": d.get("score", 0),
        "num_comments": d.get("num_comments", 0),
        "url": f"https://www.reddit.com{d['permalink']}",
        "created_utc": d.get("created_utc"),
        # Reddit uses placeholder strings ("self", "default", "nsfw", "spoiler", "") for
        # anything that isn't a real image - only a real http(s) URL is an actual thumbnail.
        "thumbnail": thumb if thumb.startswith("http") else "",
    }


async def get_subreddit_posts(subreddit: str, sort: str = "hot", limit: int = 10, min_score: int = 0) -> List[Dict[str, Any]]:
    data = await _reddit_get(f"/r/{subreddit}/{sort}", {"limit": limit})
    posts = [_reddit_post_summary(c) for c in data.get("data", {}).get("children", [])]
    return [p for p in posts if p["score"] >= min_score]


async def get_reddit_user_posts(username: str, limit: int = 10) -> List[Dict[str, Any]]:
    data = await _reddit_get(f"/user/{username}/submitted", {"limit": limit, "sort": "new"})
    return [_reddit_post_summary(c) for c in data.get("data", {}).get("children", [])]


async def search_reddit(query: str, sort: str = "top", time_filter: str = "week", limit: int = 15, min_score: int = 20) -> List[Dict[str, Any]]:
    data = await _reddit_get("/search", {"q": query, "sort": sort, "t": time_filter, "limit": limit})
    posts = [_reddit_post_summary(c) for c in data.get("data", {}).get("children", [])]
    return [p for p in posts if p["score"] >= min_score]


class GetSubredditPostsTool(BaseTool):
    name = "get_subreddit_posts"
    description = "Get posts from a specific subreddit (hot/new/rising/top), optionally filtered by minimum score."
    parameters: List[ToolParameter] = [
        ToolParameter(name="subreddit", type="string", description="Subreddit name, without 'r/'."),
        ToolParameter(name="sort", type="string", required=False, enum=["hot", "new", "rising", "top"], description="Default 'hot'."),
        ToolParameter(name="limit", type="number", required=False, description="Max posts, default 10."),
        ToolParameter(name="min_score", type="number", required=False, description="Filter out posts below this score. Default 0."),
    ]

    async def run(self, subreddit: str, sort: str = "hot", limit: int = 10, min_score: int = 0, **kwargs) -> ToolResult:
        try:
            posts = await get_subreddit_posts(subreddit, sort, limit, min_score)
            output = {"posts": posts, "count": len(posts)}
            visual_items = [{"url": p["url"], "thumbnail": p["thumbnail"], "title": p["title"]} for p in posts if p.get("thumbnail")]
            if visual_items:
                output["visual"] = {"type": "images", "query": f"r/{subreddit}", "items": visual_items}
            return ToolResult(success=True, output=output)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Reddit API error: {e.response.status_code}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SearchRedditTool(BaseTool):
    name = "search_reddit"
    description = "Search all of Reddit for a keyword/topic, filtered by a minimum score so only genuinely notable posts show up."
    parameters: List[ToolParameter] = [
        ToolParameter(name="query", type="string", description="Search terms."),
        ToolParameter(name="time_filter", type="string", required=False, enum=["hour", "day", "week", "month", "year", "all"], description="Default 'week'."),
        ToolParameter(name="min_score", type="number", required=False, description="Default 20 - filters out low-traction noise."),
    ]

    async def run(self, query: str, time_filter: str = "week", min_score: int = 20, **kwargs) -> ToolResult:
        try:
            posts = await search_reddit(query, "top", time_filter, 15, min_score)
            output = {"posts": posts, "count": len(posts)}
            visual_items = [{"url": p["url"], "thumbnail": p["thumbnail"], "title": p["title"]} for p in posts if p.get("thumbnail")]
            if visual_items:
                output["visual"] = {"type": "images", "query": query, "items": visual_items}
            return ToolResult(success=True, output=output)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Reddit API error: {e.response.status_code}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# --- YouTube (official Data API v3, needs a free API key) ------------------------

def _youtube_settings() -> Dict[str, str]:
    cfg = get_settings().get("social_media", {}).get("youtube", {})
    if not cfg.get("api_key"):
        raise RuntimeError(
            "No social_media.youtube.api_key in config/settings.yaml. Create a free API key "
            "in Google Cloud Console with the YouTube Data API v3 enabled."
        )
    return cfg


async def _youtube_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _youtube_settings()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://www.googleapis.com/youtube/v3/{path}",
            params={**params, "key": cfg["api_key"]},
        )
        resp.raise_for_status()
        return resp.json()


async def _resolve_channel(identifier: str) -> Dict[str, Any]:
    """Resolves a handle (@name), legacy username, or raw channel ID to a channel's
    id + uploads-playlist id, trying the most specific match first."""
    identifier = identifier.strip()
    if identifier.startswith("UC") and len(identifier) == 24:
        data = await _youtube_get("channels", {"part": "contentDetails,snippet", "id": identifier})
    else:
        handle = identifier if identifier.startswith("@") else f"@{identifier}"
        data = await _youtube_get("channels", {"part": "contentDetails,snippet", "forHandle": handle})

    items = data.get("items", [])
    if not items:
        # Last resort: keyword search for the closest matching channel.
        search = await _youtube_get("search", {"part": "snippet", "q": identifier, "type": "channel", "maxResults": 1})
        search_items = search.get("items", [])
        if not search_items:
            raise RuntimeError(f"No YouTube channel found for '{identifier}'.")
        channel_id = search_items[0]["snippet"]["channelId"]
        data = await _youtube_get("channels", {"part": "contentDetails,snippet", "id": channel_id})
        items = data.get("items", [])
        if not items:
            raise RuntimeError(f"No YouTube channel found for '{identifier}'.")

    item = items[0]
    return {
        "channel_id": item["id"],
        "title": item["snippet"]["title"],
        "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


async def get_channel_uploads(identifier: str, max_results: int = 10) -> List[Dict[str, Any]]:
    channel = await _resolve_channel(identifier)
    data = await _youtube_get("playlistItems", {
        "part": "snippet,contentDetails",
        "playlistId": channel["uploads_playlist_id"],
        "maxResults": max_results,
    })
    videos = []
    for item in data.get("items", []):
        snippet = item["snippet"]
        thumbnails = snippet.get("thumbnails", {})
        videos.append({
            "id": item["contentDetails"]["videoId"],
            "title": snippet["title"],
            "channel": channel["title"],
            "published_at": snippet["publishedAt"],
            "url": f"https://www.youtube.com/watch?v={item['contentDetails']['videoId']}",
            "thumbnail": (thumbnails.get("medium") or thumbnails.get("default") or {}).get("url", ""),
        })
    return videos


async def search_youtube_trending(query: str, days: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
    """Finds recent videos matching a query, then flags ones with a view count high
    relative to their channel's subscriber count - a video outperforming its channel's
    normal reach is a concrete "before it gets big" signal, not just a recent upload."""
    from datetime import datetime, timedelta, timezone

    published_after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    search = await _youtube_get("search", {
        "part": "snippet", "q": query, "type": "video", "order": "viewCount",
        "publishedAfter": published_after, "maxResults": max_results,
    })
    video_ids = [item["id"]["videoId"] for item in search.get("items", [])]
    if not video_ids:
        return []

    stats = await _youtube_get("videos", {"part": "statistics,snippet", "id": ",".join(video_ids)})
    channel_ids = list({item["snippet"]["channelId"] for item in stats.get("items", [])})
    channels = await _youtube_get("channels", {"part": "statistics", "id": ",".join(channel_ids)})
    sub_counts = {c["id"]: int(c["statistics"].get("subscriberCount", 0) or 0) for c in channels.get("items", [])}

    results = []
    for item in stats.get("items", []):
        views = int(item["statistics"].get("viewCount", 0) or 0)
        channel_id = item["snippet"]["channelId"]
        subs = sub_counts.get(channel_id, 0)
        ratio = round(views / subs, 2) if subs > 0 else None
        results.append({
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "views": views,
            "channel_subscribers": subs,
            "views_per_subscriber": ratio,
            "published_at": item["snippet"]["publishedAt"],
            "url": f"https://www.youtube.com/watch?v={item['id']}",
            "thumbnail": (item["snippet"].get("thumbnails", {}).get("medium") or item["snippet"].get("thumbnails", {}).get("default") or {}).get("url", ""),
        })
    results.sort(key=lambda r: (r["views_per_subscriber"] or 0), reverse=True)
    return results


class GetYouTubeChannelLatestTool(BaseTool):
    name = "get_youtube_channel_latest"
    description = "Get the latest uploads from a YouTube channel by handle (e.g. '@mkbhd') or channel ID."
    parameters: List[ToolParameter] = [
        ToolParameter(name="channel", type="string", description="Channel handle or ID."),
        ToolParameter(name="max_results", type="number", required=False, description="Default 10."),
    ]

    async def run(self, channel: str, max_results: int = 10, **kwargs) -> ToolResult:
        try:
            videos = await get_channel_uploads(channel, max_results)
            visual_items = [{"url": v["url"], "thumbnail": v["thumbnail"], "title": v["title"]} for v in videos if v.get("thumbnail")]
            output = {"videos": videos, "count": len(videos)}
            if visual_items:
                output["visual"] = {"type": "images", "query": f"{channel} latest uploads", "items": visual_items}
            return ToolResult(success=True, output=output)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"YouTube API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SearchYouTubeTrendingTool(BaseTool):
    name = "search_youtube_trending"
    description = (
        "Find recent YouTube videos on a topic that are outperforming their channel's normal "
        "reach (high views relative to subscriber count) - a concrete early-traction signal, "
        "not just 'recently uploaded'."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="query", type="string", description="Topic/keywords to search."),
        ToolParameter(name="days", type="number", required=False, description="How far back to look. Default 7."),
    ]

    async def run(self, query: str, days: int = 7, **kwargs) -> ToolResult:
        try:
            results = await search_youtube_trending(query, days)
            visual_items = [{"url": v["url"], "thumbnail": v["thumbnail"], "title": v["title"]} for v in results if v.get("thumbnail")]
            output = {"videos": results, "count": len(results)}
            if visual_items:
                output["visual"] = {"type": "images", "query": query, "items": visual_items}
            return ToolResult(success=True, output=output)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"YouTube API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# --- Generic watch list (works across all platforms) ------------------------------

def _watches_path() -> Path:
    cfg = get_settings().get("social_media", {})
    p = resolve_path(cfg.get("watches_file_path", "./data/social_watches.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_watches() -> List[Dict[str, Any]]:
    p = _watches_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def _save_watches(watches: List[Dict[str, Any]]) -> None:
    atomic_write_json(_watches_path(), watches)


async def _fetch_webpage_state(url: str) -> List[Dict[str, Any]]:
    """One "item" representing the page as it is now, identified by a hash of its
    text. When the page changes the id changes, which the watch loop already reads
    as new content - no separate change-detection path needed.

    Read through the shared BrowserSession so a watched page sees the same
    rendering as everything else Leti reads, including JavaScript-built pages.
    """
    from tools.browser import get_shared_session

    page = await get_shared_session().read_text(url, max_characters=20000)
    if not page.get("ok"):
        raise RuntimeError(f"Couldn't read {url}: {page.get('error', 'no content')}")

    text = page.get("text", "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    excerpt = text[:400]
    return [{
        "id": digest,
        "url": page.get("url", url),
        "title": page.get("title", ""),
        "excerpt": excerpt,
        "characters": len(text),
    }]


async def _fetch_latest_for_watch(watch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns the platform's current latest items for a watch, most-recent-first,
    each with at least an 'id' field usable for last_seen_id diffing."""
    platform, identifier = watch["platform"], watch["identifier"]

    if platform == "youtube":
        return await get_channel_uploads(identifier, max_results=15)
    if platform == "reddit_user":
        return await get_reddit_user_posts(identifier, limit=15)
    if platform == "reddit_subreddit":
        return await get_subreddit_posts(identifier, sort="new", limit=15)
    if platform == "webpage":
        return await _fetch_webpage_state(identifier)
    if platform in ("instagram", "tiktok", "facebook"):
        # Lazy import: keeps this module usable without Playwright installed if the
        # user only cares about YouTube/Reddit watches.
        from tools.social_login import fetch_latest_for_platform
        return await fetch_latest_for_platform(platform, identifier)

    raise ValueError(f"Unknown platform: {platform}")


class AddSocialWatchTool(BaseTool):
    name = "add_social_watch"
    description = (
        "Watch something for changes and get told whenever you next check - regardless of how "
        "much time has passed, not just 'recently'. Covers YouTube channels, Reddit "
        "users/subreddits, (if logged in) Instagram/TikTok/Facebook, and with platform "
        "'webpage' any URL at all, which reports when the page's content changes. Use this "
        "for 'tell me when this page/pricing/job board changes'."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="platform", type="string", enum=WATCHABLE_PLATFORMS, description="Which platform, or 'webpage' for any URL."),
        ToolParameter(name="identifier", type="string", description="Channel handle, username, subreddit name, or - for 'webpage' - the full URL."),
        ToolParameter(name="label", type="string", required=False, description="Friendly name for this watch."),
    ]

    async def run(self, platform: str, identifier: str, label: str = "", **kwargs) -> ToolResult:
        try:
            watches = _load_watches()
            entry = {
                "id": uuid.uuid4().hex[:8],
                "platform": platform,
                "identifier": identifier,
                "label": label or identifier,
                "last_seen_id": None,
                "last_checked": None,
                "created_at": time.time(),
            }
            watches.append(entry)
            _save_watches(watches)
            return ToolResult(success=True, output=f"Watching {entry['label']} on {platform}. I'll flag new content next time I check.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListSocialWatchesTool(BaseTool):
    name = "list_social_watches"
    description = "List all currently configured social media watches."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"watches": _load_watches()})


class RemoveSocialWatchTool(BaseTool):
    name = "remove_social_watch"
    description = "Stop watching an account/subreddit, by its watch id (from list_social_watches)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="watch_id", type="string", description="The watch's id."),
    ]

    async def run(self, watch_id: str, **kwargs) -> ToolResult:
        watches = _load_watches()
        remaining = [w for w in watches if w["id"] != watch_id]
        if len(remaining) == len(watches):
            return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
        _save_watches(remaining)
        return ToolResult(success=True, output=f"Stopped watching {watch_id}.")


class CheckSocialWatchesTool(BaseTool):
    name = "check_social_watches"
    description = (
        "Check every configured watch for new content since it was last checked, however long "
        "ago that was. Call this at the start of a session to catch up, or any time the user "
        "asks 'anything new from X'. Only reports genuinely new items, not the whole history."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        watches = _load_watches()
        if not watches:
            return ToolResult(success=True, output={"summary": "No watches configured.", "updates": []})

        updates = []
        errors = []
        changed = False

        for watch in watches:
            try:
                items = await _fetch_latest_for_watch(watch)
            except Exception as e:
                errors.append({"watch": watch["label"], "error": str(e)})
                continue

            if not items:
                watch["last_checked"] = time.time()
                changed = True
                continue

            latest_id = items[0]["id"]
            if watch["last_seen_id"] is None:
                # First-ever check: establish baseline, don't spam "new" for pre-existing content.
                watch["last_seen_id"] = latest_id
                watch["last_checked"] = time.time()
                changed = True
                continue

            seen_ids = {i["id"] for i in items}
            if watch["last_seen_id"] in seen_ids:
                new_items = []
                for i in items:
                    if i["id"] == watch["last_seen_id"]:
                        break
                    new_items.append(i)
            else:
                # Backlog exceeded what we fetched (last_seen_id fell off the fetched window) -
                # report everything fetched rather than silently dropping older new items.
                new_items = items

            if new_items:
                updates.append({"watch": watch["label"], "platform": watch["platform"], "new_items": new_items})
                watch["last_seen_id"] = latest_id
                changed = True

            watch["last_checked"] = time.time()
            changed = True

        if changed:
            _save_watches(watches)

        summary = f"{len(updates)} watch(es) have new content." if updates else "Nothing new."
        return ToolResult(success=True, output={"summary": summary, "updates": updates, "errors": errors})


# --- Trend-tool integration ---------------------------------------------------

async def gather_social_signals(niche: str) -> str:
    """Best-effort live grounding for scout_find_trends: pulls real current chatter/traction
    for a niche from whichever platforms are configured, and skips (never raises) anything
    that isn't. Returns "" if nothing at all is available, so the caller can proceed with
    pure model-knowledge trend generation as a fallback."""
    sections = []

    try:
        posts = await search_reddit(niche, "top", "week", 10, min_score=30)
        if posts:
            lines = [f"- \"{p['title']}\" (r/{p['subreddit']}, score {p['score']}, {p['num_comments']} comments)" for p in posts[:5]]
            sections.append("Reddit (top discussions this week):\n" + "\n".join(lines))
    except Exception:
        pass

    try:
        videos = await search_youtube_trending(niche, days=7, max_results=10)
        standout = [v for v in videos if (v["views_per_subscriber"] or 0) >= 0.5][:5]
        if standout:
            lines = [
                f"- \"{v['title']}\" by {v['channel']} ({v['views']:,} views vs {v['channel_subscribers']:,} subscribers - "
                f"{v['views_per_subscriber']}x ratio)"
                for v in standout
            ]
            sections.append("YouTube (outperforming their channel's normal reach):\n" + "\n".join(lines))
    except Exception:
        pass

    if not sections:
        return ""
    return "Live social signals gathered for this niche:\n\n" + "\n\n".join(sections)
