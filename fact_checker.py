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

Given an original CLAIM from a video transcript and a FACT-CHECK article from \
a knowledge base, determine whether the fact-check is relevant to the claim.

Before answering, reason through these three checks:
1. SUBJECT: Do the claim and fact-check refer to the same person, party, \
organisation, or entity?
2. EVENT: Do they refer to the same event, election, situation, or time period?
3. ASSERTION: Is the core factual assertion the same — not just overlapping \
keywords or numbers?

A fact-check is relevant ONLY if all three checks pass. Minor differences in \
wording or scope are acceptable as long as the core factual question is the same.

Respond in this exact format (three short lines of reasoning, then the verdict):
SUBJECT: <match/mismatch — one sentence>
EVENT: <match/mismatch — one sentence>
ASSERTION: <match/mismatch — one sentence>
VERDICT: <relevant or irrelevant>\
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


QUERY_REWRITE_PROMPT = """\
You are a search query optimizer for a fact-checking knowledge base.

Given a CLAIM extracted from a video transcript, produce one or two short \
search queries that will retrieve relevant fact-checks. Use the same language \
as the claim.

Rules:
- QUERY 1 (always): The core factual assertion, concise (max ~15 words). \
Fix transcription errors in names. Remove filler and hedging.
- QUERY 2 (only if the claim contains reported speech or attribution, e.g. \
"X said Y" or "according to X"): A second query focusing on the attributed \
assertion itself, without the speaker.
- Output one query per line. Nothing else.\
"""


def _rewrite_queries(
    claim_text: str,
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> list[str]:
    """Reformulate a claim into one or two KB search queries."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": claim_text},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        queries = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if queries:
            return queries
    except Exception:
        logger.exception("Query rewrite failed, using original claim.")
    return [claim_text]


def _format_factcheck(fc: dict) -> str:
    """Extract only the fields relevant for relevance judgment."""
    parts = []
    for key in ("title", "claim", "verdict"):
        val = fc.get(key)
        if val:
            parts.append(f"{key.upper()}: {val}")
    return "\n".join(parts) if parts else json.dumps(fc, ensure_ascii=False)


def _judge_relevance(
    claim_text: str,
    factcheck: dict,
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> bool:
    """Use an LLM to decide whether a fact-check is relevant to the claim."""
    fc_text = _format_factcheck(factcheck)
    user_msg = f"CLAIM:\n{claim_text}\n\nFACT-CHECK:\n{fc_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        last_line = answer.strip().splitlines()[-1] if answer.strip() else ""
        return "relevant" in last_line and "irrelevant" not in last_line
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
        claim_text = claim["claim_text"]
        queries = _rewrite_queries(claim_text, client)
        claim["search_queries"] = queries
        logger.info(
            "Searching KB for claim %d/%d: %s",
            i + 1,
            len(claims),
            claim_text[:80],
        )
        for q in queries:
            logger.info("  query: %s", q[:80])

        seen_ids: set[str] = set()
        results: list[dict] = []
        for q in queries:
            try:
                batch = _search_kb(q)
            except Exception:
                logger.exception("KB search failed for query: %s", q[:80])
                continue
            for fc in batch:
                aid = fc.get("article_id", fc.get("url", ""))
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    results.append(fc)

        logger.info("  Got %d unique result(s) from KB.", len(results))
        claim["factchecks"] = results

        relevant = []
        for fc in results:
            if _judge_relevance(claim_text, fc, client):
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
