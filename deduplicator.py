"""Deduplicate claims using multilingual sentence embeddings."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from config import DEDUP_MODEL_NAME, DEDUP_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

_st_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        logger.info("Loading sentence-transformer '%s'…", DEDUP_MODEL_NAME)
        _st_model = SentenceTransformer(DEDUP_MODEL_NAME)
        logger.info("Sentence-transformer loaded.")
    return _st_model


def _claim_text(claim: dict) -> str:
    return claim.get("claim_text_normalised") or claim["claim_text"]


def deduplicate(
    claims: list[dict],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Remove near-duplicate claims based on cosine similarity.

    For each duplicate group, keeps the claim with the highest cw_score_max.
    Adds a ``duplicates`` field listing merged claim texts.
    """
    if len(claims) <= 1:
        return claims

    model = _get_model()
    texts = [_claim_text(c) for c in claims]
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
            if sim_matrix[i][j] >= threshold:
                group_indices.append(j)
                merged[j] = True

        best_idx = max(group_indices, key=lambda k: claims[k].get("cw_score_max", 0))
        representative = dict(claims[best_idx])

        if len(group_indices) > 1:
            representative["duplicates"] = [
                _claim_text(claims[k])
                for k in group_indices
                if k != best_idx
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
