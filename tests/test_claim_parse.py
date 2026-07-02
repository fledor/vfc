"""Parsing the LLM's chain-of-thought claim output.

This is a common failure point: the model varies its formatting (numbered
lines, reasoning prefixes, bare claims, or "NONE"), and a parsing slip either
drops real claims or leaks reasoning text as claims.
"""
from __future__ import annotations

from claim_detector import _parse_claim_lines


def test_extracts_claim_prefixed_line():
    answer = "SUBJECT: Germany\nVERIFIABLE: yes\nCLAIM: Germany imported 40% of its gas from Russia in 2021."
    assert _parse_claim_lines(answer) == [
        "Germany imported 40% of its gas from Russia in 2021."
    ]


def test_tolerates_leading_numbering():
    answer = "1. SUBJECT: X\n2. CLAIM: The AfD received 20% in the last election."
    assert _parse_claim_lines(answer) == [
        "The AfD received 20% in the last election."
    ]


def test_multiple_claims():
    answer = "CLAIM: First claim.\nCLAIM: Second claim."
    assert _parse_claim_lines(answer) == ["First claim.", "Second claim."]


def test_none_answer_yields_no_claims():
    assert _parse_claim_lines("NONE") == []
    assert _parse_claim_lines("  none  ") == []


def test_fallback_skips_reasoning_prefixes():
    # No CLAIM: prefix — the bare assertion should survive, prefixes dropped.
    answer = "SUBJECT: oil price\nSTATEMENT: something\nOil hit $120 a barrel in June."
    assert _parse_claim_lines(answer) == ["Oil hit $120 a barrel in June."]


def test_empty_lines_ignored():
    answer = "CLAIM: A real claim.\n\n   \n"
    assert _parse_claim_lines(answer) == ["A real claim."]
