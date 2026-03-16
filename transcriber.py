"""Transcribe audio files using faster-whisper (CTranslate2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from config import TRANSCRIPTS_DIR, WHISPER_LANGUAGE, WHISPER_MODEL

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info("Loading whisper model '%s' (this may take a moment)…", WHISPER_MODEL)
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="auto")
        logger.info("Model loaded.")
    return _model


def transcribe(
    audio_path: str | Path,
    output_dir: Path = TRANSCRIPTS_DIR,
    language: str = WHISPER_LANGUAGE,
) -> list[dict]:
    """Transcribe an audio file and save segment-level JSON + plain text.

    Returns a list of segment dicts: {"start", "end", "text"}.
    """
    audio_path = Path(audio_path)
    video_id = audio_path.stem

    model = _get_model()
    logger.info("Transcribing %s (language=%s)…", audio_path.name, language)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    logger.info(
        "Detected language: %s (probability %.2f)",
        info.language,
        info.language_probability,
    )

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    json_path = output_dir / f"{video_id}.json"
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    logger.info("Saved transcript JSON: %s (%d segments)", json_path, len(segments))

    txt_path = output_dir / f"{video_id}.txt"
    txt_path.write_text(
        "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    )
    logger.info("Saved transcript TXT: %s", txt_path)

    return segments
