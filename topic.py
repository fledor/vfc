"""Extract the topic of a video from its metadata and transcript opening."""

from __future__ import annotations

import logging

from config import OPENAI_API_KEY, OPENAI_MODEL, get_openai_client

logger = logging.getLogger(__name__)

TOPIC_PROMPT = """\
Given the metadata and opening transcript of a video, summarise the topic \
in one concise sentence (max 20 words) in the SAME LANGUAGE as the transcript. \
Focus on the specific subject matter being discussed, not the format of the show.

Output only the topic sentence, nothing else.\
"""


def extract_topic(
    segments: list[dict],
    language: str,
    video_title: str = "",
    channel: str = "",
    max_seconds: float = 120.0,
) -> str:
    """Return a one-line topic description for the video.

    Uses the YouTube title, channel name, and first ~2 minutes of transcript.
    Falls back to the video title if no API key is available.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — using video title as topic.")
        return video_title or "unknown topic"

    opening_lines = []
    for seg in segments:
        if seg["start"] > max_seconds:
            break
        opening_lines.append(seg["text"])
    opening_text = " ".join(opening_lines)[:2000]

    meta_parts = []
    if video_title:
        meta_parts.append(f"Title: {video_title}")
    if channel:
        meta_parts.append(f"Channel: {channel}")
    meta_parts.append(f"Language: {language}")
    meta_block = "\n".join(meta_parts)

    user_msg = f"METADATA:\n{meta_block}\n\nTRANSCRIPT OPENING:\n{opening_text}"

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TOPIC_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        topic = (response.choices[0].message.content or "").strip()
        if topic:
            logger.info("Extracted topic: %s", topic)
            return topic
    except Exception:
        logger.exception("Topic extraction failed.")

    return video_title or "unknown topic"
