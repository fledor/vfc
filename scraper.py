"""Download audio from YouTube videos/playlists using yt-dlp."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yt_dlp

from config import AUDIO_DIR

logger = logging.getLogger(__name__)

# Upper bound on playlist positions to scan while filling max_videos.
_SCAN_CAP = 500


def _playlist_scan_end(max_videos: int) -> int:
    """How many playlist slots to fetch (private / failed slots are skipped)."""
    if max_videos <= 0:
        return 1
    return min(_SCAN_CAP, max(max_videos * 20, max_videos + 10))


def download_audio(
    url: str,
    output_dir: Path = AUDIO_DIR,
    max_videos: int = 1,
) -> list[dict]:
    """Download audio from a YouTube URL (video or playlist).

    Returns metadata only for videos that have a ``.wav`` on disk (cached or
    newly downloaded): ``{"video_id", "title", "channel", "duration", "audio_path"}``.

    For playlists, private, unavailable, or failed downloads are skipped; the
    scraper keeps scanning forward until ``max_videos`` usable videos are found
    or the scanned portion of the playlist ends (see ``_SCAN_CAP``).
    """
    collected: list[dict] = []

    def _progress_hook(d: dict) -> None:
        if d["status"] == "finished":
            logger.info("Downloaded: %s", d.get("filename", "?"))

    scan_end = _playlist_scan_end(max_videos)

    opts_extract: dict = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            },
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "playlistend": scan_end,
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(opts_extract) as ydl:
        info = ydl.extract_info(url, download=False)

    if info is None:
        logger.error("Could not extract info from URL: %s", url)
        return []

    entries = info.get("entries", [info])

    opts_dl: dict = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            },
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    for entry in entries:
        if len(collected) >= max_videos:
            break
        if entry is None:
            logger.info("Skipping unavailable playlist entry.")
            continue
        video_id = entry.get("id")
        if not video_id:
            logger.info("Skipping playlist entry without id.")
            continue

        audio_file = output_dir / f"{video_id}.wav"
        meta = {
            "video_id": video_id,
            "title": entry.get("title", ""),
            "channel": entry.get("channel") or entry.get("uploader", ""),
            "duration": entry.get("duration"),
            "audio_path": str(audio_file),
        }
        meta_path = output_dir / f"{video_id}.meta.json"

        if audio_file.exists():
            logger.info("Audio already cached: %s — skipping download.", audio_file)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            collected.append(meta)
            continue

        dl_url = (
            entry.get("webpage_url")
            or entry.get("url")
            or f"https://www.youtube.com/watch?v={video_id}"
        )
        try:
            with yt_dlp.YoutubeDL(opts_dl) as ydl:
                ydl.download([dl_url])
        except Exception:
            logger.exception("Failed to download: %s", dl_url)
            continue

        if not audio_file.exists():
            logger.warning(
                "No audio file after download for %s — skipping.", video_id
            )
            continue

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        collected.append(meta)

    if len(collected) < max_videos:
        logger.warning(
            "Only %d/%d videos after scanning up to %d playlist positions "
            "(private, unavailable, or failed downloads are skipped).",
            len(collected),
            max_videos,
            scan_end,
        )

    return collected
