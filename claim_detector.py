"""Detect checkworthy claims using a hybrid classifier + LLM pipeline.

Classifier: language-routed ClaimBuster models (3-class: NFS/UFS/CFS).
  - English: whispAI/ClaimBuster-DeBERTaV2  (84.2% accuracy)
  - German and others: Nithiwat/mdeberta-v3-base_claimbuster (83%, multilingual)

Pipeline:
  1. Merge Whisper segments into sentences
  2. Classify each sentence (keep only CFS above threshold)
  3. For each CFS sentence, call GPT-4o-mini with surrounding context to
     extract a concise, self-contained claim and fix transcription errors
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
    CW_MODEL_FALLBACK,
    CW_MODEL_NAME,
    CW_MODEL_PATH,
    CW_MODELS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

_models: dict[str, tuple[AutoTokenizer, AutoModelForSequenceClassification, int]] = {}

EXTRACT_PROMPT_TEMPLATE = """\
You are a claim extraction assistant for a fact-checking pipeline.

VIDEO TOPIC: {topic}

You receive a SENTENCE that a classifier flagged as potentially check-worthy, \
together with surrounding CONTEXT from the transcript.

Your task:
1. Decide whether the sentence truly contains a verifiable factual claim \
(numbers, dates, statistics, specific events, attributable statements). \
If it is merely an opinion, rhetorical question, vague assertion, or \
conversational filler, respond with exactly: NONE
2. If it does contain one or more verifiable claims, extract each as a \
concise, FULLY DECONTEXTUALISED sentence in the original language of the \
transcript. Each claim must be understandable without any surrounding context:
   - Resolve ALL pronouns to the actual names of people, parties, or \
institutions using the context.
   - Name the specific event, election, or time period being discussed.
   - Summarise rather than quote verbatim.
   - Fix obvious transcription errors in names.
   - Do NOT include claims that are merely someone's subjective assessment \
unless they contain a specific verifiable fact.
3. Output one claim per line. Output nothing else (no numbering, no \
explanation).\
"""


def _resolve_cfs_label(id2label: dict) -> int:
    """Find the label index for 'Check-worthy Factual Statement'."""
    for idx, label in id2label.items():
        low = str(label).lower()
        if "check-worthy" in low or "check_worthy" in low or low == "cfs":
            return int(idx)
    for idx, label in id2label.items():
        low = str(label).lower()
        if low in ("factual", "checkworthy", "claim", "yes", "1"):
            return int(idx)
    return max(id2label.keys(), default=1)


def _get_model(
    language: str = "de",
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, int]:
    """Load (and cache) the checkworthiness model for the given language."""
    if CW_MODEL_PATH:
        model_id = CW_MODEL_PATH
    elif CW_MODEL_NAME:
        model_id = CW_MODEL_NAME
    else:
        model_id = CW_MODELS.get(language, CW_MODEL_FALLBACK)

    if model_id in _models:
        return _models[model_id]

    logger.info("Loading checkworthiness model '%s' (lang=%s)…", model_id, language)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    id2label = model.config.id2label or {}
    cfs_id = _resolve_cfs_label(id2label)
    logger.info("Model loaded (labels: %s, CFS_id=%s).", id2label, cfs_id)

    _models[model_id] = (tokenizer, model, cfs_id)
    return tokenizer, model, cfs_id


def _merge_segments_into_sentences(
    segments: list[dict],
    max_words: int = 40,
) -> list[dict]:
    """Concatenate adjacent Whisper segments into sentence-sized windows.

    Uses punctuation cues as natural break points, falling back to a
    word-count limit.
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
    language: str = "de",
    batch_size: int = 16,
) -> list[dict]:
    """Classify each sentence and store the CFS probability."""
    tokenizer, model, cfs_id = _get_model(language)
    device = next(model.parameters()).device

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
            sent["cfs_score"] = probs[j, cfs_id].item()
            sent["predicted_label"] = logits[j].argmax(-1).item()

    return sentences


def _extract_claims_llm(
    sentences: list[dict],
    cfs_indices: list[int],
    topic: str = "",
    context_window: int = 2,
    model: str = OPENAI_MODEL,
) -> list[dict]:
    """For each CFS sentence, send it with context to GPT-4o-mini for extraction."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set – returning raw CFS sentences.")
        return [
            {
                "claim_text": sentences[i]["text"],
                "start": sentences[i]["start"],
                "end": sentences[i]["end"],
                "segment_indices": sentences[i]["segment_indices"],
                "cfs_score": sentences[i]["cfs_score"],
            }
            for i in cfs_indices
        ]

    system_prompt = EXTRACT_PROMPT_TEMPLATE.format(topic=topic or "unknown")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    claims: list[dict] = []

    for idx in cfs_indices:
        sent = sentences[idx]

        ctx_start = max(0, idx - context_window)
        ctx_end = min(len(sentences), idx + context_window + 1)
        context_lines = []
        for ci in range(ctx_start, ctx_end):
            marker = ">>>" if ci == idx else "   "
            context_lines.append(f"{marker} {sentences[ci]['text']}")
        context_block = "\n".join(context_lines)

        user_msg = (
            f"SENTENCE (flagged as check-worthy):\n{sent['text']}\n\n"
            f"CONTEXT:\n{context_block}"
        )

        logger.debug("LLM extract for sentence %d: %s", idx, sent["text"][:80])

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            answer = (response.choices[0].message.content or "").strip()

            if answer.upper() == "NONE":
                logger.debug("  LLM discarded sentence %d.", idx)
                continue

            for line in answer.splitlines():
                line = line.strip()
                if not line:
                    continue
                claims.append({
                    "claim_text": line,
                    "claim_original": sent["text"],
                    "start": sent["start"],
                    "end": sent["end"],
                    "segment_indices": sent["segment_indices"],
                    "cfs_score": sent["cfs_score"],
                })
        except Exception:
            logger.exception(
                "LLM extraction failed for sentence %d: %s",
                idx,
                sent["text"][:80],
            )
            claims.append({
                "claim_text": sent["text"],
                "claim_original": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
                "segment_indices": sent["segment_indices"],
                "cfs_score": sent["cfs_score"],
            })

    return claims


def detect_claims(
    segments: list[dict],
    video_id: str,
    language: str = "de",
    topic: str = "",
    output_dir: Path = CLAIMS_DIR,
) -> list[dict]:
    """Detect checkworthy claims from transcript segments.

    1. Merge Whisper segments into sentences.
    2. Classify each sentence with the language-appropriate ClaimBuster model.
    3. Filter to CFS (check-worthy factual) sentences above threshold.
    4. Use GPT-4o-mini to extract concise claims from each CFS sentence.

    Returns a list of claim dicts.
    """
    logger.info("Merging %d segments into sentences…", len(segments))
    sentences = _merge_segments_into_sentences(segments)
    logger.info("Created %d sentences.", len(sentences))

    logger.info("Classifying sentences for checkworthiness (lang=%s)…", language)
    sentences = _classify_sentences(sentences, language=language)

    cfs_indices = [
        i
        for i, s in enumerate(sentences)
        if s["cfs_score"] >= CW_CONFIDENCE_THRESHOLD
    ]
    logger.info(
        "%d / %d sentences classified as CFS (threshold=%.2f).",
        len(cfs_indices),
        len(sentences),
        CW_CONFIDENCE_THRESHOLD,
    )

    if not cfs_indices:
        logger.info("No check-worthy sentences found.")
        output_path = output_dir / f"{video_id}.json"
        output_path.write_text("[]")
        return []

    logger.info("Extracting claims with LLM…")
    claims = _extract_claims_llm(sentences, cfs_indices, topic=topic)
    logger.info("Extracted %d claim(s) from %d CFS sentences.", len(claims), len(cfs_indices))

    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(json.dumps(claims, ensure_ascii=False, indent=2))
    logger.info("Saved %d claims to %s", len(claims), output_path)

    return claims
