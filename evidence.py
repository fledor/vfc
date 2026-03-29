"""Search authoritative web sources for evidence supporting or refuting claims."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from config import (
    EVIDENCE_DOMAINS,
    EVIDENCE_MAX_RESULTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TAVILY_API_KEY,
)

logger = logging.getLogger(__name__)


class _TavilyCreditsExhausted(Exception):
    """Raised when Tavily API credits are used up."""

# ---------------------------------------------------------------------------
# LLM prompt for evidence assessment
# ---------------------------------------------------------------------------

EVIDENCE_ASSESS_PROMPT = """\
You are an evidence assessment assistant for a fact-checking pipeline.

You receive a CLAIM from a video transcript and EVIDENCE snippets \
retrieved from authoritative sources (government sites, official \
statistics, wire services, international organisations).

Your task:
1. Read the evidence carefully and determine whether it supports, \
refutes, or is insufficient to verify the claim.
2. Respond in this exact format:

EVIDENCE: <1-2 sentence summary of what the sources say that is \
relevant to the claim. If nothing relevant, say "No relevant \
evidence found.">
ASSESSMENT: <one of: SUPPORTS | REFUTES | INSUFFICIENT | UNRELATED>

Rules for ASSESSMENT:
- SUPPORTS: The evidence clearly corroborates the claim's core assertion.
- REFUTES: The evidence clearly contradicts the claim's core assertion.
- INSUFFICIENT: The evidence is topically related but does not \
conclusively support or refute the claim (e.g. partial data, \
different time period, related but different assertion).
- UNRELATED: The evidence has no meaningful connection to the claim.\
"""

EVIDENCE_QUERY_PROMPT = """\
You are a search query optimizer. Given a CLAIM from a video transcript, \
produce exactly 1 concise English search query that would find official \
statistics, government statements, or authoritative reports relevant to \
verifying this claim. Focus on the specific facts, numbers, or events \
mentioned.

Output one query. Nothing else.\
"""


# ---------------------------------------------------------------------------
# Tavily search
# ---------------------------------------------------------------------------

def _search_tavily(
    query: str,
    domains: list[str] | None = None,
    max_results: int = EVIDENCE_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Search Tavily with domain restrictions, return result dicts."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — skipping web evidence search.")
        return []

    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=domains or EVIDENCE_DOMAINS,
        )
        results = response.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]
    except Exception as exc:
        exc_str = str(exc).lower()
        if "limit" in exc_str or "quota" in exc_str or "429" in exc_str:
            raise _TavilyCreditsExhausted(str(exc)) from exc
        logger.exception("Tavily search failed for query: %s", query)
        return []


# ---------------------------------------------------------------------------
# Query generation for evidence search
# ---------------------------------------------------------------------------

def _rewrite_evidence_queries(
    claim_text: str,
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> list[str]:
    """Generate English search queries optimised for official/authoritative sources."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVIDENCE_QUERY_PROMPT},
                {"role": "user", "content": claim_text},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
        queries = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if queries:
            return queries
    except Exception:
        logger.exception("Evidence query rewrite failed for: %s", claim_text)
    return [claim_text]


# ---------------------------------------------------------------------------
# LLM evidence assessment
# ---------------------------------------------------------------------------

def _assess_evidence(
    claim_text: str,
    snippets: list[dict[str, Any]],
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> dict[str, str]:
    """Use LLM to assess whether retrieved evidence supports/refutes the claim."""
    if not snippets:
        return {"summary": "No evidence found.", "assessment": "INSUFFICIENT"}

    evidence_block = "\n\n".join(
        f"SOURCE: {s['title']} ({s['url']})\n{s['content']}"
        for s in snippets
    )

    user_msg = f"CLAIM:\n{claim_text}\n\nEVIDENCE:\n{evidence_block}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVIDENCE_ASSESS_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        answer = (response.choices[0].message.content or "").strip()

        summary = ""
        assessment = "INSUFFICIENT"
        for line in answer.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("EVIDENCE:"):
                summary = stripped[len("EVIDENCE:"):].strip()
            elif upper.startswith("ASSESSMENT:"):
                raw_val = stripped[len("ASSESSMENT:"):].strip().upper()
                if raw_val in ("SUPPORTS", "REFUTES", "INSUFFICIENT", "UNRELATED"):
                    assessment = raw_val

        return {"summary": summary, "assessment": assessment}
    except Exception:
        logger.exception("Evidence assessment failed for claim: %s", claim_text)
        return {"summary": "Assessment failed.", "assessment": "INSUFFICIENT"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_evidence(
    claims: list[dict],
    language: str = "de",
) -> list[dict]:
    """Search authoritative web sources for evidence on each claim.

    Enriches each claim dict with ``web_evidence``:
      - ``sources``: list of {title, url, content, score}
      - ``summary``: LLM-generated summary of relevant evidence
      - ``assessment``: SUPPORTS | REFUTES | INSUFFICIENT | UNRELATED
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — skipping web evidence.")
        return claims

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — skipping evidence assessment.")
        return claims

    client = OpenAI(api_key=OPENAI_API_KEY)

    credits_exhausted = False

    for i, claim in enumerate(claims):
        claim_text = claim["claim_text"]
        logger.info(
            "Evidence search %d/%d: %s",
            i + 1, len(claims), claim_text,
        )

        if credits_exhausted:
            logger.info("  Skipped (Tavily credits exhausted).")
            continue

        queries = _rewrite_evidence_queries(claim_text, client)
        for q in queries:
            logger.info("  evidence query: %s", q)

        seen_urls: set[str] = set()
        all_snippets: list[dict] = []

        try:
            for q in queries:
                for result in _search_tavily(q):
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_snippets.append(result)
        except _TavilyCreditsExhausted:
            logger.warning(
                "Tavily API credits exhausted — skipping web evidence "
                "for remaining %d claim(s).",
                len(claims) - i,
            )
            credits_exhausted = True
            continue

        logger.info("  Got %d unique source(s) from authoritative domains.", len(all_snippets))

        result = _assess_evidence(claim_text, all_snippets, client)
        logger.info(
            "  Assessment: %s — %s",
            result["assessment"],
            result["summary"][:120] if result["summary"] else "(none)",
        )

        claim["web_evidence"] = {
            "sources": all_snippets,
            "queries": queries,
            "summary": result["summary"],
            "assessment": result["assessment"],
        }

    return claims
