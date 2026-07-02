"""Retrieve fact-checks from multiple sources and judge relevance."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    DEDUP_MODEL_NAME,
    FACTCHECKS_DIR,
    GOOGLE_FC_API_KEY,
    GOOGLE_FC_MAX_AGE_DAYS,
    GOOGLE_FC_PAGE_SIZE,
    PRIVATE_KB_API_KEY,
    PRIVATE_KB_API_URL,
    PRIVATE_KB_INCLUDE_EVIDENCE,
    PRIVATE_KB_SEARCH_LIMIT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_RELEVANCE_MODEL,
)

logger = logging.getLogger(__name__)

_st_model: SentenceTransformer | None = None

RELEVANCE_PROMPT = """\
You are a relevance judge for a fact-checking system.

Given a CLAIM from a video transcript and a FACT-CHECK entry, decide whether \
the fact-check is relevant to verifying the claim.

Note: fact-check databases often frame their CLAIM field as viral social-media \
content (e.g. "Video claims X" or "Post says Y"). A fact-check is still \
**relevant** if its investigation addresses the same real-world proposition.

Evaluate in order — any failing check means **irrelevant**:
1. MEDIA — Does the fact-check's CLAIM investigate whether a specific video, \
image, or post is authentic or fabricated (signalled by phrases like \
"Video shows…", "claims to show…", "shared on Facebook claims…")? If yes, it \
is relevant ONLY when the transcript claim is itself about the authenticity of \
that specific media — not when the claim is a general real-world assertion.
2. SUBJECT — Do both refer to the same person, country, or entity?
3. ASSERTION — Do both address the same underlying factual question about the \
same specific event, incident, or policy? Two claims can cite different figures \
and still match, provided they investigate the same event. What is NOT the same \
question: the same type of metric or statistic applied to a different event, \
conflict, or context.
4. TEMPORAL — Does the fact-check apply to the time frame of the claim?

Respond in this exact format:
MEDIA: <media authenticity check? yes/no — if yes, state why relevant or irrelevant>
SUBJECT: <match/mismatch — one sentence>
ASSERTION: <match/mismatch — one sentence>
TEMPORAL: <match/mismatch — one sentence>
VERDICT: <relevant or irrelevant>\
"""

PRE_FILTER_SIM_THRESHOLD = 0.28


def _get_st_model() -> SentenceTransformer:
    """Load sentence-transformer model (shared with deduplicator)."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(DEDUP_MODEL_NAME)
    return _st_model


# ---------------------------------------------------------------------------
# Optional private fact-check knowledge base search
# ---------------------------------------------------------------------------

def _search_kb(query: str) -> list[dict]:
    """Search an optional private fact-check knowledge base API.

    Requires both ``PRIVATE_KB_API_URL`` and ``PRIVATE_KB_API_KEY``; if either
    is unset the source is skipped.
    """
    if not (PRIVATE_KB_API_URL and PRIVATE_KB_API_KEY):
        logger.debug("Private KB not configured — skipping KB search.")
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PRIVATE_KB_API_KEY}",
    }
    payload: dict = {
        "query": query,
        "limit": PRIVATE_KB_SEARCH_LIMIT,
        "include_evidence": PRIVATE_KB_INCLUDE_EVIDENCE,
    }

    resp = requests.post(PRIVATE_KB_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict]
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results", data.get("data", [data]))
    else:
        results = []

    for fc in results:
        fc.setdefault("origin", "kb")
    return results


# ---------------------------------------------------------------------------
# Google Fact Check Tools API
# ---------------------------------------------------------------------------

_GOOGLE_FC_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _search_google_fc(query: str, language: str = "de") -> list[dict]:
    """Search Google Fact Check Claim Search API and normalise results.

    Runs two requests -- one with ``languageCode`` and one without -- to catch
    both language-specific and cross-language fact-checks, then deduplicates.
    """
    if not GOOGLE_FC_API_KEY:
        logger.debug("GOOGLE_FACTCHECK_API_KEY not set — skipping Google FC search.")
        return []

    base_params: dict = {
        "key": GOOGLE_FC_API_KEY,
        "query": query,
        "pageSize": GOOGLE_FC_PAGE_SIZE,
    }
    if GOOGLE_FC_MAX_AGE_DAYS > 0:
        base_params["maxAgeDays"] = GOOGLE_FC_MAX_AGE_DAYS

    seen_urls: set[str] = set()
    results: list[dict] = []

    param_variants = [
        {**base_params, "languageCode": language},
        base_params,
    ]

    for params in param_variants:
        try:
            resp = requests.get(_GOOGLE_FC_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Google FC search failed for query: %s", query)
            continue

        for claim_obj in data.get("claims", []):
            claim_text = claim_obj.get("text", "")
            claimant = claim_obj.get("claimant", "")
            for review in claim_obj.get("claimReview", []):
                url = review.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                publisher = review.get("publisher", {})
                results.append({
                    "source": publisher.get("name", publisher.get("site", "")),
                    "title": review.get("title"),
                    "claim": claim_text,
                    "claimant": claimant,
                    "verdict": review.get("textualRating"),
                    "url": url,
                    "review_date": review.get("reviewDate"),
                    "origin": "google",
                })

    return results


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """\
You are a search query optimizer for a fact-checking knowledge base.

Given a CLAIM extracted from a video transcript, produce search queries \
that will retrieve relevant fact-checks.

Rules:
- QUERY 1 (always): The core factual assertion in the ORIGINAL language, \
concise (max ~15 words). Fix transcription errors in names. Remove filler.
- QUERY 2 (only if the claim contains reported speech or attribution): \
A second query focusing on the attributed assertion itself, without the \
speaker. Same language as the claim.
- ENGLISH QUERY (always, if the claim is NOT in English): An English \
translation of QUERY 1, since many fact-check databases have better \
English coverage.
- Output one query per line. Nothing else.\
"""


def _rewrite_queries(
    claim_text: str,
    client: OpenAI,
    model: str = OPENAI_MODEL,
) -> list[str]:
    """Reformulate a claim into one or two search queries."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
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
        logger.exception("Query rewrite failed, using original claim.")
    return [claim_text]


# ---------------------------------------------------------------------------
# Pre-filtering and relevance judgment
# ---------------------------------------------------------------------------

def _format_factcheck(fc: dict) -> str:
    """Extract only the fields relevant for relevance judgment."""
    parts = []
    for key in ("title", "claim", "verdict"):
        val = fc.get(key)
        if val:
            parts.append(f"{key.upper()}: {val}")
    return "\n".join(parts) if parts else json.dumps(fc, ensure_ascii=False)


def _pre_filter(
    claim_text: str,
    results: list[dict],
) -> list[dict]:
    """Remove obviously irrelevant fact-checks before LLM judgment.

    1. Drop entries where both claim and verdict are empty.
    2. Drop entries below a cosine similarity floor.
    """
    filtered: list[dict] = []
    texts_for_sim: list[tuple[int, str]] = []

    for i, fc in enumerate(results):
        fc_claim = fc.get("claim") or ""
        fc_verdict = fc.get("verdict") or ""
        if not fc_claim and not fc_verdict:
            logger.debug("  Pre-filter: dropping result with null claim+verdict.")
            continue
        filtered.append(fc)
        compare_text = fc_claim or fc.get("title", "")
        if compare_text:
            texts_for_sim.append((len(filtered) - 1, compare_text))

    if not texts_for_sim or not filtered:
        return filtered

    model = _get_st_model()
    claim_emb = model.encode([claim_text], normalize_embeddings=True)
    fc_texts = [t for _, t in texts_for_sim]
    fc_embs = model.encode(fc_texts, normalize_embeddings=True)
    sims = (claim_emb @ fc_embs.T)[0]

    keep_indices: set[int] = set()
    for (idx, _), sim in zip(texts_for_sim, sims):
        if float(sim) >= PRE_FILTER_SIM_THRESHOLD:
            keep_indices.add(idx)
        else:
            logger.debug(
                "  Pre-filter: dropping result (sim=%.3f < %.2f): %s",
                float(sim),
                PRE_FILTER_SIM_THRESHOLD,
                fc_texts[texts_for_sim.index((idx, _))],
            )

    # Keep entries that had no text to compare (benefit of the doubt)
    all_sim_indices = {idx for idx, _ in texts_for_sim}
    return [
        fc
        for i, fc in enumerate(filtered)
        if i not in all_sim_indices or i in keep_indices
    ]


def _parse_relevance(answer: str) -> dict:
    """Parse structured MEDIA/SUBJECT/ASSERTION/TEMPORAL/VERDICT response."""
    result: dict[str, str] = {}
    for line in answer.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip().upper()] = val.strip()
    return result


def _is_consistent_relevant(parsed: dict) -> bool:
    """Check that the reasoning doesn't contradict a 'relevant' verdict.

    Returns False (irrelevant) when:
    - VERDICT is missing or says "irrelevant"
    - MEDIA gate fires (yes = media-authenticity FC) and reasoning says irrelevant
    - Any of SUBJECT / ASSERTION / TEMPORAL starts with "mismatch"
    """
    verdict = parsed.get("VERDICT", "").lower()
    if "irrelevant" in verdict or "relevant" not in verdict:
        return False

    media = parsed.get("MEDIA", "").lower()
    if media.startswith("yes") and "irrelevant" in media:
        return False

    for key in ("SUBJECT", "ASSERTION", "TEMPORAL"):
        val = parsed.get(key, "").lower()
        if val.startswith("mismatch"):
            return False

    return True


def _call_and_parse(
    claim_text: str,
    factcheck: dict,
    client: OpenAI,
    model: str,
) -> dict:
    """Make a single LLM relevance call and parse the structured response."""
    fc_text = _format_factcheck(factcheck)
    user_msg = f"CLAIM:\n{claim_text}\n\nFACT-CHECK:\n{fc_text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RELEVANCE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
        timeout=30,
    )
    answer = (response.choices[0].message.content or "").strip()
    parsed = _parse_relevance(answer)
    parsed["_raw"] = answer
    return parsed


def _judge_relevance(
    claim_text: str,
    factcheck: dict,
    client: OpenAI,
    model: str = OPENAI_RELEVANCE_MODEL,
) -> tuple[bool, dict]:
    """Judge relevance with consistency check and confirmation call.

    Returns (is_relevant, reasoning_dict).
    """
    try:
        parsed = _call_and_parse(claim_text, factcheck, client, model)
        if not _is_consistent_relevant(parsed):
            return False, parsed

        # Confirmation: second independent call must also agree
        parsed2 = _call_and_parse(claim_text, factcheck, client, model)
        if not _is_consistent_relevant(parsed2):
            parsed2["_confirmed"] = False
            return False, parsed2

        parsed["_confirmed"] = True
        return True, parsed
    except Exception:
        logger.exception("Relevance judgment failed for claim: %s", claim_text)
        return False, {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_claims(
    claims: list[dict],
    video_id: str,
    language: str = "de",
    output_dir: Path = FACTCHECKS_DIR,
) -> list[dict]:
    """For each claim, search KB + Google FC, pre-filter, and judge relevance.

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
            "Searching for claim %d/%d: %s",
            i + 1,
            len(claims),
            claim_text,
        )
        for q in queries:
            logger.info("  query: %s", q)

        seen_urls: set[str] = set()
        results: list[dict] = []

        def _run_kb(q: str) -> list[dict]:
            return _search_kb(q)

        def _run_google(q: str) -> list[dict]:
            return _search_google_fc(q, language=language)

        with ThreadPoolExecutor(max_workers=len(queries) * 2) as pool:
            search_futures: dict = {}
            for q in queries:
                search_futures[pool.submit(_run_kb, q)] = ("kb", q)
                search_futures[pool.submit(_run_google, q)] = ("google", q)

            for future in as_completed(search_futures):
                source, q = search_futures[future]
                try:
                    for fc in future.result():
                        url = fc.get("url", fc.get("article_id", ""))
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(fc)
                except Exception:
                    logger.exception(
                        "%s search failed for query: %s", source.upper(), q,
                    )

        logger.info(
            "  Got %d unique result(s) (KB + Google).", len(results),
        )
        claim["factchecks"] = results

        filtered = _pre_filter(claim_text, results)
        logger.info(
            "  %d result(s) after pre-filtering (from %d).",
            len(filtered),
            len(results),
        )

        relevant = []
        if filtered:
            with ThreadPoolExecutor(max_workers=min(len(filtered), 3)) as pool:
                futures = {
                    pool.submit(_judge_relevance, claim_text, fc, client, OPENAI_RELEVANCE_MODEL): fc
                    for fc in filtered
                }
                for future in as_completed(futures):
                    is_relevant, reasoning = future.result()
                    fc = futures[future]
                    fc["_relevance_reasoning"] = reasoning
                    if is_relevant:
                        relevant.append(fc)

        claim["relevant_factchecks"] = relevant
        logger.info(
            "  %d / %d fact-check(s) judged relevant.",
            len(relevant),
            len(filtered),
        )

    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(json.dumps(claims, ensure_ascii=False, indent=2))
    logger.info("Saved fact-check results to %s", output_path)

    return claims
