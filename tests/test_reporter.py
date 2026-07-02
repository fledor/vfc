"""Report classification: verified vs inconclusive vs not-shown.

The report distinguishes claims it could *verify* (fact-check match or
conclusive web evidence) from those with only INSUFFICIENT evidence, which are
surfaced but not counted as verified. UNRELATED / no-evidence claims are not
shown at all.
"""
from __future__ import annotations

import json

import reporter


def test_has_evidence_excludes_unrelated():
    assert reporter._has_evidence({"web_evidence": {"assessment": "SUPPORTS"}})
    assert reporter._has_evidence({"web_evidence": {"assessment": "INSUFFICIENT"}})
    assert not reporter._has_evidence({"web_evidence": {"assessment": "UNRELATED"}})
    assert not reporter._has_evidence({})


def test_is_verified():
    assert reporter._is_verified({"relevant_factchecks": [{"source": "x"}]})
    assert reporter._is_verified({"web_evidence": {"assessment": "REFUTES"}})
    assert not reporter._is_verified({"web_evidence": {"assessment": "INSUFFICIENT"}})
    assert not reporter._is_verified({})


def test_generate_results_counts_and_status(tmp_path):
    claims = [
        {"claim_text": "fc match", "relevant_factchecks": [{"source": "x", "verdict": "false"}]},
        {"claim_text": "supported", "web_evidence": {"assessment": "SUPPORTS", "summary": "s"}},
        {"claim_text": "inconclusive", "web_evidence": {"assessment": "INSUFFICIENT", "summary": "s"}},
        {"claim_text": "unrelated", "web_evidence": {"assessment": "UNRELATED"}},
        {"claim_text": "nothing"},
    ]
    json_path, md_path = reporter.generate_results(claims, "vid_test", output_dir=tmp_path)

    data = json.loads(json_path.read_text())
    assert data["total_claims"] == 5
    assert data["verified_claims"] == 2       # fc match + SUPPORTS
    assert data["inconclusive_claims"] == 1   # INSUFFICIENT only
    assert len(data["claims"]) == 3           # UNRELATED and empty are not shown

    statuses = {c["claim_text"]: c["status"] for c in data["claims"]}
    assert statuses == {
        "fc match": "verified",
        "supported": "verified",
        "inconclusive": "inconclusive",
    }

    md = md_path.read_text()
    assert "2 verified" in md
    assert "1 with inconclusive evidence" in md
