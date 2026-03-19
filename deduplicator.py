"""Deduplicate claims using embeddings + LLM subsumption check.

Two-stage approach:
  1. Cosine similarity on sentence embeddings catches near-identical pairs.
  2. For borderline pairs (between a soft and hard threshold), an LLM decides
     whether one claim subsumes or duplicates the other.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from config import (
    DEDUP_MODEL_NAME,
    DEDUP_SIMILARITY_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

_st_model: SentenceTransformer | None = None

SUBSUMPTION_PROMPT = """\
You are a deduplication assistant. Given two claims (A and B) extracted from \
the same transcript, decide whether they are duplicates or one subsumes the other.

Respond with exactly one of:
- DUPLICATE  (they express the same factual assertion)
- SUBSUMES_A (claim A contains everything in B, keep A)
- SUBSUMES_B (claim B contains everything in A, keep B)
- DISTINCT   (they are about different facts, keep both)\
"""

SOFT_THRESHOLD = 0.65


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        logger.info("Loading sentence-transformer '%s'…", DEDUP_MODEL_NAME)
        _st_model = SentenceTransformer(DEDUP_MODEL_NAME)
        logger.info("Sentence-transformer loaded.")
    return _st_model


def _llm_subsumption_check(text_a: str, text_b: str) -> str:
    """Ask the LLM whether two claims are duplicates/subsumptions."""
    if not OPENAI_API_KEY:
        return "DISTINCT"

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUBSUMPTION_PROMPT},
                {"role": "user", "content": f"A: {text_a}\nB: {text_b}"},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        for label in ("DUPLICATE", "SUBSUMES_A", "SUBSUMES_B", "DISTINCT"):
            if label in answer:
                return label
    except Exception:
        logger.exception("LLM subsumption check failed.")
    return "DISTINCT"


def deduplicate(
    claims: list[dict],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Remove duplicate and subsumed claims.

    Hard threshold (``threshold``, default 0.85): auto-merge without LLM.
    Soft threshold (0.65 – ``threshold``): use LLM subsumption check.
    Below soft threshold: always keep both.

    For each group, keeps the longest claim text (most informative).
    """
    if len(claims) <= 1:
        return claims

    model = _get_model()
    texts = [c["claim_text"] for c in claims]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sim_matrix = embeddings @ embeddings.T

    merged: list[bool] = [False] * len(claims)
    deduplicated: list[dict] = []

    for i in range(len(claims)):
        if merged[i]:
            continue

        group_indices = [i]
        for j in range(i + 1, len(claims)):
            if merged[j]:
                continue
            sim = float(sim_matrix[i][j])

            if sim >= threshold:
                group_indices.append(j)
                merged[j] = True
            elif sim >= SOFT_THRESHOLD:
                verdict = _llm_subsumption_check(texts[i], texts[j])
                if verdict in ("DUPLICATE", "SUBSUMES_A", "SUBSUMES_B"):
                    group_indices.append(j)
                    merged[j] = True
                    logger.debug(
                        "LLM merged (sim=%.2f, %s): '%s' + '%s'",
                        sim, verdict, texts[i][:50], texts[j][:50],
                    )

        best_idx = max(group_indices, key=lambda k: len(texts[k]))
        representative = dict(claims[best_idx])

        if len(group_indices) > 1:
            representative["duplicates"] = [
                texts[k] for k in group_indices if k != best_idx
            ]

        deduplicated.append(representative)

    removed = len(claims) - len(deduplicated)
    if removed:
        logger.info(
            "Deduplication: %d -> %d claims (%d duplicates removed).",
            len(claims),
            len(deduplicated),
            removed,
        )
    else:
        logger.info("No duplicates found among %d claims.", len(claims))

    return deduplicated
