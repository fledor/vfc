"""Fact-check relevance parsing and the consistency gate.

The consistency check is the safeguard that rejects a fact-check whose stated
reasoning contradicts its own 'relevant' verdict. If it lets a mismatch
through, irrelevant fact-checks get attached to claims.
"""
from __future__ import annotations

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
