"""Pipeline orchestration: scrape -> transcribe -> detect -> deduplicate -> fact-check."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from claim_detector import detect_claims
from deduplicator import deduplicate
from fact_checker import check_claims
from scraper import download_audio
from transcriber import transcribe

logger = logging.getLogger(__name__)

DEFAULT_PLAYLIST = (
    "https://www.youtube.com/playlist?list=PLdPrKDvwrog6SPKzMAVh1d6cPVtGtDIeT"
)


def _print_report(claims: list[dict], video_id: str) -> None:
    logger.info("=== Report for %s ===", video_id)
    for i, claim in enumerate(claims, 1):
        text = claim.get("claim_text_normalised") or claim["claim_text"]
        score = claim.get("cw_score_max", "?")
        relevant = claim.get("relevant_factchecks", [])
        logger.info(
            "  [%d] (cw=%.2f) %s",
            i,
            float(score) if score != "?" else 0,
            text[:120],
        )
        if relevant:
            for j, fc in enumerate(relevant, 1):
                fc_title = fc.get("title") or fc.get("claim") or str(fc)[:100]
                logger.info("       FC-%d: %s", j, fc_title)
        else:
            logger.info("       No relevant fact-checks found.")


def run_pipeline(
    playlist_url: str,
    max_videos: int = 1,
    skip_factcheck: bool = False,
) -> None:
    logger.info("=== Step 1: Scraping audio from YouTube ===")
    videos = download_audio(playlist_url, max_videos=max_videos)
    logger.info("Downloaded %d video(s).", len(videos))

    for video in videos:
        video_id = video["video_id"]
        audio_path = Path(video["audio_path"])

        if not audio_path.exists():
            logger.error("Audio file not found: %s — skipping.", audio_path)
            continue

        logger.info("=== Step 2: Transcribing %s ===", video_id)
        segments = transcribe(audio_path)
        logger.info("Transcribed %d segments.", len(segments))

        logger.info("=== Step 3: Detecting checkworthy claims for %s ===", video_id)
        claims = detect_claims(segments, video_id)
        logger.info("Detected %d checkworthy claim(s).", len(claims))

        if not claims:
            logger.info("No claims detected — skipping remaining steps.")
            continue

        logger.info("=== Step 4: Deduplicating claims ===")
        claims = deduplicate(claims)
        logger.info("%d claim(s) after deduplication.", len(claims))

        if not skip_factcheck:
            logger.info("=== Step 5: Searching fact-check knowledge base ===")
            claims = check_claims(claims, video_id)

        _print_report(claims, video_id)

    logger.info("=== Pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Video fact-checking pipeline")
    parser.add_argument(
        "--playlist-url",
        default=DEFAULT_PLAYLIST,
        help="YouTube playlist or video URL",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=1,
        help="Maximum number of videos to process (default: 1)",
    )
    parser.add_argument(
        "--skip-factcheck",
        action="store_true",
        help="Skip KB search and relevance judgment (steps 4-5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(
            args.playlist_url,
            args.max_videos,
            skip_factcheck=args.skip_factcheck,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
