"""Retrieve fact-checks from the knowledge base API and judge relevance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests
from openai import OpenAI

from config import (
    FACTCHECKS_DIR,
    KB_API_KEY,
    KB_API_URL,
    KB_INCLUDE_EVIDENCE,
    KB_SEARCH_LIMIT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

RELEVANCE_PROMPT = """\
You are a relevance judge for a fact-checking system.

Given an original CLAIM from a German talk show and a FACT-CHECK result from a \
knowledge base, determine whether the fact-check is relevant to the claim.

A fact-check is relevant if it addresses the same or a closely related factual \
assertion. Minor differences in wording or scope are acceptable as long as the \
core factual question is the same.

Respond with exactly one word: "relevant" or "irrelevant".\
"""


def _search_kb(query: str) -> list[dict]:
    """Search the fact-check knowledge base API."""
    if not KB_API_KEY:
        raise RuntimeError(
            "KB_API_KEY not set. Add it to .env or export it."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KB_API_KEY}",
    }
    payload: dict = {
        "query": query,
        "limit": KB_SEARCH_LIMIT,
        "include_evidence": KB_INCLUDE_EVIDENCE,
    }

    resp = requests.post(KB_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", data.get("data", [data]))
    return []


def _judge_relevance(
    claim_text: str,
    factcheck: dict,
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> bool:
    """Use an LLM to decide whether a fact-check is relevant to the claim."""
    fc_text = json.dumps(factcheck, ensure_ascii=False, indent=2)
    user_msg = f"CLAIM:\n{claim_text}\n\nFACT-CHECK:\n{fc_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        return answer.startswith("relevant") and not answer.startswith("irrelevant")
    except Exception:
        logger.exception("Relevance judgment failed for claim: %s", claim_text[:80])
        return False


def check_claims(
    claims: list[dict],
    video_id: str,
    output_dir: Path = FACTCHECKS_DIR,
) -> list[dict]:
    """For each claim, search the KB and judge relevance of returned fact-checks.

    Enriches each claim dict with ``factchecks`` (all results) and
    ``relevant_factchecks`` (filtered by LLM relevance).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Required for relevance judgment."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    for i, claim in enumerate(claims):
        query = claim.get("claim_text_normalised") or claim["claim_text"]
        logger.info(
            "Searching KB for claim %d/%d: %s",
            i + 1,
            len(claims),
            query[:80],
        )

        try:
            results = _search_kb(query)
        except Exception:
            logger.exception("KB search failed for query: %s", query[:80])
            claim["factchecks"] = []
            claim["relevant_factchecks"] = []
            continue

        logger.info("  Got %d result(s) from KB.", len(results))
        claim["factchecks"] = results

        relevant = []
        for fc in results:
            if _judge_relevance(query, fc, client):
                relevant.append(fc)

        claim["relevant_factchecks"] = relevant
        logger.info(
            "  %d / %d fact-check(s) judged relevant.",
            len(relevant),
            len(results),
        )

    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(json.dumps(claims, ensure_ascii=False, indent=2))
    logger.info("Saved fact-check results to %s", output_path)

    return claims
