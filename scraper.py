"""Download audio from YouTube videos/playlists using yt-dlp."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yt_dlp

from config import AUDIO_DIR

logger = logging.getLogger(__name__)


def download_audio(
    url: str,
    output_dir: Path = AUDIO_DIR,
    max_videos: int = 1,
) -> list[dict]:
    """Download audio from a YouTube URL (video or playlist).

    Returns a list of metadata dicts, one per downloaded video:
        {"video_id", "title", "duration", "audio_path"}
    """
    collected: list[dict] = []

    def _progress_hook(d: dict) -> None:
        if d["status"] == "finished":
            logger.info("Downloaded: %s", d.get("filename", "?"))

    opts: dict = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            },
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "playlistend": max_videos,
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries", [info]) if info else []
    to_download: list[str] = []

    for entry in entries:
        if entry is None:
            continue
        video_id = entry["id"]
        audio_file = output_dir / f"{video_id}.wav"
        meta = {
            "video_id": video_id,
            "title": entry.get("title", ""),
            "channel": entry.get("channel") or entry.get("uploader", ""),
            "duration": entry.get("duration"),
            "audio_path": str(audio_file),
        }
        collected.append(meta)

        meta_path = output_dir / f"{video_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        if audio_file.exists():
            logger.info("Audio already cached: %s — skipping download.", audio_file)
        else:
            to_download.append(entry.get("webpage_url") or entry["url"])

    if to_download:
        opts_dl = {**opts}
        with yt_dlp.YoutubeDL(opts_dl) as ydl:
            for vid_url in to_download:
                ydl.download([vid_url])

    return collected
