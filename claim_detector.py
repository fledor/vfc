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
import re
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
    OPENAI_EXTRACT_MODEL,
)

logger = logging.getLogger(__name__)

_models: dict[str, tuple[AutoTokenizer, AutoModelForSequenceClassification, int]] = {}

# Strips leading numbering like "1. ", "2) ", "3: " from CoT output lines.
_NUM_PREFIX_RE = re.compile(r"^[\d]+[\.\):\-]\s*", re.ASCII)


def _parse_claim_lines(answer: str) -> list[str]:
    """Extract claim strings from the LLM's chain-of-thought output.

    Primary path: collect lines prefixed with ``CLAIM:`` (tolerating leading
    numbering like ``2. CLAIM:``). Fallback: if no such prefix exists, treat
    non-empty lines that are not reasoning prefixes (SUBJECT/STATEMENT/
    VERIFIABLE) as claims. Returns an empty list when the model answered
    ``NONE``.
    """
    if answer.strip().upper() == "NONE":
        return []

    claim_lines: list[str] = []
    for line in answer.splitlines():
        stripped = _NUM_PREFIX_RE.sub("", line.strip())
        if stripped.upper().startswith("CLAIM:"):
            claim_lines.append(stripped[len("CLAIM:"):].strip())

    # Fallback: no CLAIM: prefix found — treat plain lines as claims,
    # skipping reasoning-prefix lines.
    if not claim_lines:
        for line in answer.splitlines():
            stripped = _NUM_PREFIX_RE.sub("", line.strip())
            if not stripped:
                continue
            upper = stripped.upper()
            if any(upper.startswith(p) for p in (
                "SUBJECT:", "STATEMENT:", "VERIFIABLE:",
            )):
                continue
            if upper == "NONE":
                return []
            claim_lines.append(stripped)

    return [c for c in claim_lines if c]


EXTRACT_PROMPT_TEMPLATE = """\
You are a claim extraction assistant for a fact-checking pipeline.

VIDEO TOPIC: {topic}

You receive a transcript FRAGMENT flagged as potentially check-worthy, \
with surrounding CONTEXT (the flagged fragment is marked with >>>).

Important: this transcript was produced by automatic speech recognition. \
Each fragment is just a timed cutoff — NOT a complete sentence. A single \
spoken statement typically spans multiple fragments. You MUST read the \
surrounding context to understand the full statement being made.

Your task:
1. Read the flagged fragment together with its context. First, reason \
briefly about the following:
   SUBJECT: Who or what is this fragment about, and what specific event, \
conflict, or situation is being discussed? Use the VIDEO TOPIC and \
surrounding context to identify the concrete referent. Trace pronouns, \
implicit subjects, and contextual references back to specific names, \
countries, or events.
   STATEMENT: What complete factual statement is being made? Combine \
information from the flagged fragment and its context as needed.
   VERIFIABLE: Is this a verifiable factual claim, or purely opinion / \
rhetorical question / vague assertion / filler? (yes or no)

2. If VERIFIABLE is yes, output the claim as:
   CLAIM: <a single complete, grammatically correct declarative sentence>

   Rules for the CLAIM line:
   - The sentence must be self-contained and understandable without context.
   - Use the referents you identified in SUBJECT — do not guess or default \
to a generic entity.
   - Use context to disambiguate numbers and abbreviations.
   - If someone is quoting another person, preserve the attribution.
   - Be careful about speaker boundaries: do not merge content from \
a different speaker into the claim.
   - Do not add opinions or conclusions that were not stated. However, \
you MUST resolve implicit references using the context: if the surrounding \
sentences make clear which event, conflict, or entity the speaker means, \
include that in the claim. This is decontextualisation, not adding information.
   - Fix only obvious transcription errors in proper nouns.
   - If the claim mentions a quantity, percentage, or statistic, specify \
both the metric AND the specific event, conflict, or situation it pertains \
to — a reader must know exactly which real-world context the numbers belong to.
   - Output the claim in the SAME LANGUAGE as the transcript.

3. If VERIFIABLE is no, respond with exactly: NONE\
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
    min_words: int = 6,
) -> list[dict]:
    """Concatenate adjacent Whisper segments into sentence-sized windows.

    Uses punctuation cues as natural break points, falling back to a
    word-count limit.  Very short sentences (< *min_words*) are merged
    into the previous sentence to avoid orphan fragments like "Auf 140
    Tage." being separated from the sentence they complete.
    """
    raw: list[dict] = []
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
            raw.append({
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
        raw.append({
            "text": " ".join(buf_texts),
            "start": buf_start,
            "end": segments[buf_indices[-1]]["end"],
            "segment_indices": list(buf_indices),
        })

    # Merge short sentences back into the previous one
    sentences: list[dict] = []
    for sent in raw:
        wc = len(sent["text"].split())
        if sentences and wc < min_words:
            prev = sentences[-1]
            prev["text"] = prev["text"] + " " + sent["text"]
            prev["end"] = sent["end"]
            prev["segment_indices"] = prev["segment_indices"] + sent["segment_indices"]
        else:
            sentences.append(sent)

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
    context_window: int = 5,
    model: str = OPENAI_EXTRACT_MODEL,
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

        logger.debug("LLM extract for sentence %d: %s", idx, sent["text"])

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            answer = (response.choices[0].message.content or "").strip()

            claim_lines = _parse_claim_lines(answer)
            if not claim_lines:
                logger.debug("  LLM discarded sentence %d.", idx)
                continue

            for claim_text in claim_lines:
                claims.append({
                    "claim_text": claim_text,
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
                sent["text"],
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
