"""Detect checkworthy claims using a fine-tuned transformer classifier.

Default model: Sami92/XLM-R-Large-ClaimDetection (XLM-R Large fine-tuned on
German data from GermEval 2021 + GPT-4o-annotated Telegram posts).
Labels: "factual" (= contains a verifiable claim) vs "non-factual".

Pipeline: merge whisper segments into sentences -> classify each sentence ->
group adjacent checkworthy sentences -> optionally normalise claims with LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CLAIMS_DIR,
    CW_CONFIDENCE_THRESHOLD,
    CW_MODEL_NAME,
    CW_MODEL_PATH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None

# Label that corresponds to "this is a factual / checkworthy claim".
# Sami92/XLM-R-Large-ClaimDetection uses "factual" vs "non-factual";
# MultiCW models use label index 1 for checkworthy.  We resolve at runtime.
_POSITIVE_LABEL_ID: int | None = None

NORMALISE_PROMPT = """\
You are a claim extraction assistant. Given a passage from a German talk show \
transcript that has been flagged as containing a checkworthy claim, extract the \
core factual claim as a single concise sentence in German. Remove filler words, \
speech artefacts, and irrelevant context. If the passage contains multiple \
distinct claims, return each on its own line. Output only the claim(s), nothing else.\
"""


def _get_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _tokenizer, _model, _POSITIVE_LABEL_ID
    if _tokenizer is None or _model is None:
        model_id = CW_MODEL_PATH or CW_MODEL_NAME
        logger.info("Loading checkworthiness model '%s'…", model_id)
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)
        _model.eval()

        id2label = _model.config.id2label or {}
        _POSITIVE_LABEL_ID = _resolve_positive_label(id2label)
        logger.info(
            "Model loaded (labels: %s, positive_id=%s).",
            id2label,
            _POSITIVE_LABEL_ID,
        )
    return _tokenizer, _model


def _resolve_positive_label(id2label: dict) -> int:
    """Figure out which output index means 'checkworthy / factual claim'."""
    for idx, label in id2label.items():
        low = str(label).lower()
        if low in ("factual", "checkworthy", "check-worthy", "claim", "yes", "1"):
            return int(idx)
    # Fallback: assume last index is the positive class
    return max(id2label.keys(), default=1)


def _merge_segments_into_sentences(
    segments: list[dict],
    max_words: int = 40,
) -> list[dict]:
    """Concatenate adjacent whisper segments into sentence-sized windows.

    Uses punctuation cues (period, question mark, exclamation mark) as natural
    break points, falling back to a word-count limit.
    """
    sentences: list[dict] = []
    buf_texts: list[str] = []
    buf_indices: list[int] = []
    buf_start: float | None = None
    word_count = 0

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        if buf_start is None:
            buf_start = seg["start"]

        buf_texts.append(text)
        buf_indices.append(i)
        word_count += len(text.split())
        ends_sentence = text[-1] in ".?!…"

        if ends_sentence or word_count >= max_words:
            sentences.append({
                "text": " ".join(buf_texts),
                "start": buf_start,
                "end": seg["end"],
                "segment_indices": list(buf_indices),
            })
            buf_texts = []
            buf_indices = []
            buf_start = None
            word_count = 0

    if buf_texts:
        sentences.append({
            "text": " ".join(buf_texts),
            "start": buf_start,
            "end": segments[buf_indices[-1]]["end"],
            "segment_indices": list(buf_indices),
        })

    return sentences


@torch.no_grad()
def _classify_sentences(
    sentences: list[dict],
    batch_size: int = 16,
) -> list[dict]:
    """Run checkworthiness classification on each sentence."""
    tokenizer, model = _get_model()
    device = next(model.parameters()).device
    pos_id = _POSITIVE_LABEL_ID or 1

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        texts = [s["text"] for s in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        for j, sent in enumerate(batch):
            sent["cw_score"] = probs[j, pos_id].item()

    return sentences


def _group_checkworthy(
    sentences: list[dict],
    threshold: float = CW_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Group adjacent sentences that exceed the checkworthiness threshold."""
    claims: list[dict] = []
    current: dict | None = None

    for sent in sentences:
        if sent["cw_score"] >= threshold:
            if current is None:
                current = {
                    "texts": [sent["text"]],
                    "start": sent["start"],
                    "end": sent["end"],
                    "segment_indices": list(sent["segment_indices"]),
                    "cw_scores": [sent["cw_score"]],
                }
            else:
                current["texts"].append(sent["text"])
                current["end"] = sent["end"]
                current["segment_indices"].extend(sent["segment_indices"])
                current["cw_scores"].append(sent["cw_score"])
        else:
            if current is not None:
                claims.append(_finalise_group(current))
                current = None

    if current is not None:
        claims.append(_finalise_group(current))

    return claims


def _finalise_group(group: dict) -> dict:
    scores = group["cw_scores"]
    return {
        "claim_text": " ".join(group["texts"]),
        "start": group["start"],
        "end": group["end"],
        "segment_indices": group["segment_indices"],
        "cw_score_mean": sum(scores) / len(scores),
        "cw_score_max": max(scores),
    }


def _normalise_claims_llm(claims: list[dict], model: str = OPENAI_MODEL) -> list[dict]:
    """Use an LLM to clean up claim text (remove speech artefacts, extract core claim)."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set – skipping LLM claim normalisation.")
        return claims

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    for claim in claims:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NORMALISE_PROMPT},
                    {"role": "user", "content": claim["claim_text"]},
                ],
                temperature=0.0,
            )
            normalised = (response.choices[0].message.content or "").strip()
            if normalised:
                claim["claim_text_normalised"] = normalised
        except Exception:
            logger.exception("LLM normalisation failed for claim: %s", claim["claim_text"][:80])

    return claims


def detect_claims(
    segments: list[dict],
    video_id: str,
    output_dir: Path = CLAIMS_DIR,
    normalise: bool = True,
) -> list[dict]:
    """Detect checkworthy claims from transcript segments.

    1. Merge whisper segments into sentences.
    2. Classify each sentence with the checkworthiness model.
    3. Group adjacent checkworthy sentences.
    4. Optionally normalise claims with an LLM.

    Returns a list of claim dicts.
    """
    logger.info("Merging %d segments into sentences…", len(segments))
    sentences = _merge_segments_into_sentences(segments)
    logger.info("Created %d sentences.", len(sentences))

    logger.info("Classifying sentences for checkworthiness…")
    sentences = _classify_sentences(sentences)

    cw_count = sum(1 for s in sentences if s["cw_score"] >= CW_CONFIDENCE_THRESHOLD)
    logger.info(
        "%d / %d sentences above threshold (%.2f).",
        cw_count,
        len(sentences),
        CW_CONFIDENCE_THRESHOLD,
    )

    claims = _group_checkworthy(sentences)
    logger.info("Grouped into %d claim span(s).", len(claims))

    if normalise and claims:
        logger.info("Normalising claims with LLM…")
        claims = _normalise_claims_llm(claims)

    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(json.dumps(claims, ensure_ascii=False, indent=2))
    logger.info("Saved %d claims to %s", len(claims), output_path)

    return claims
