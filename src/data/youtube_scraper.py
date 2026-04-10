"""Fetch comments from a YouTube video without an API key.

Uses the ``youtube-comment-downloader`` package which scrapes the public
comments endpoint.  Falls back gracefully when the network is unavailable
or the video has comments disabled.
"""

from __future__ import annotations

import re
from typing import Any


def _parse_votes(value: object) -> int:
    """Convert vote counts like ``'220k'`` or ``'1.5m'`` to int."""
    if isinstance(value, int):
        return value
    s = str(value).strip().replace(",", "")
    if not s:
        return 0
    multiplier = 1
    if s[-1].lower() == "k":
        multiplier, s = 1_000, s[:-1]
    elif s[-1].lower() == "m":
        multiplier, s = 1_000_000, s[:-1]
    try:
        return int(float(s) * multiplier)
    except (ValueError, OverflowError):
        return 0


def extract_video_id(url: str) -> str | None:
    """Return the 11-character video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_comments(
    url: str,
    max_comments: int = 100,
    sort_by: int = 1,
) -> list[dict[str, Any]]:
    """Download up to *max_comments* from *url*.

    Parameters
    ----------
    url:
        Any valid YouTube URL (watch, share, shorts, embed).
    max_comments:
        Upper bound on the number of comments to return.
    sort_by:
        0 = popular first, 1 = newest first (default).

    Returns
    -------
    A list of dicts with keys ``text``, ``author``, ``time``, ``likes``.
    Returns an empty list on any failure.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return []

    try:
        from youtube_comment_downloader import YoutubeCommentDownloader  # type: ignore
    except ImportError:
        return []

    downloader = YoutubeCommentDownloader()
    comments: list[dict[str, Any]] = []
    try:
        generator = downloader.get_comments_from_url(
            f"https://www.youtube.com/watch?v={video_id}",
            sort_by=sort_by,
        )
        for raw in generator:
            comments.append(
                {
                    "text": str(raw.get("text", "")),
                    "author": str(raw.get("author", "Anonymous")),
                    "time": str(raw.get("time", "")),
                    "likes": _parse_votes(raw.get("votes", 0)),
                }
            )
            if len(comments) >= max_comments:
                break
    except Exception:
        pass

    return comments
