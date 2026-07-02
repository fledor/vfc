"""Fact-check relevance parsing and the consistency gate.

The consistency check is the safeguard that rejects a fact-check whose stated
reasoning contradicts its own 'relevant' verdict. If it lets a mismatch
through, irrelevant fact-checks get attached to claims.
"""
from __future__ import annotations

import fact_checker
from fact_checker import _is_consistent_relevant, _parse_relevance


def test_parse_relevance_extracts_fields():
    answer = (
        "MEDIA: no\n"
        "SUBJECT: match — both about Germany\n"
        "ASSERTION: match — same gas-import question\n"
        "TEMPORAL: match\n"
        "VERDICT: relevant"
    )
    parsed = _parse_relevance(answer)
    assert parsed["VERDICT"] == "relevant"
    assert parsed["SUBJECT"].startswith("match")


def test_consistent_relevant_passes():
    parsed = {
        "MEDIA": "no",
        "SUBJECT": "match",
        "ASSERTION": "match",
        "TEMPORAL": "match",
        "VERDICT": "relevant",
    }
    assert _is_consistent_relevant(parsed) is True


def test_irrelevant_verdict_rejected():
    parsed = {"SUBJECT": "match", "ASSERTION": "match",
              "TEMPORAL": "match", "VERDICT": "irrelevant"}
    assert _is_consistent_relevant(parsed) is False


def test_missing_verdict_rejected():
    parsed = {"SUBJECT": "match", "ASSERTION": "match", "TEMPORAL": "match"}
    assert _is_consistent_relevant(parsed) is False


def test_assertion_mismatch_overrides_relevant_verdict():
    # Verdict says relevant but the reasoning contradicts it — must reject.
    parsed = {
        "MEDIA": "no",
        "SUBJECT": "match",
        "ASSERTION": "mismatch — different conflict",
        "TEMPORAL": "match",
        "VERDICT": "relevant",
    }
    assert _is_consistent_relevant(parsed) is False


def test_media_gate_rejects_when_media_fc_but_irrelevant():
    parsed = {
        "MEDIA": "yes — this fact-check is about video authenticity, irrelevant here",
        "SUBJECT": "match",
        "ASSERTION": "match",
        "TEMPORAL": "match",
        "VERDICT": "relevant",
    }
    assert _is_consistent_relevant(parsed) is False


def test_check_claims_no_crash_when_no_factchecks(monkeypatch, tmp_path):
    """Regression: an empty post-filter set must not spin up a 0-worker pool.

    ThreadPoolExecutor(max_workers=0) raises ValueError, which previously
    aborted the whole run whenever a claim had no surviving fact-checks.
    """
    monkeypatch.setattr(fact_checker, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(fact_checker, "get_openai_client", lambda: object())
    monkeypatch.setattr(fact_checker, "_rewrite_queries", lambda text, client, model=None: [text])
    monkeypatch.setattr(fact_checker, "_search_kb", lambda q: [])
    monkeypatch.setattr(fact_checker, "_search_google_fc", lambda q, language="de": [])

    claims = [{"claim_text": "An obscure claim with no matching fact-checks."}]
    out = fact_checker.check_claims(claims, "vid_test", output_dir=tmp_path)

    assert out[0]["factchecks"] == []
    assert out[0]["relevant_factchecks"] == []
