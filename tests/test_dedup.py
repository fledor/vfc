"""Deduplication merge decisions.

The risk here is over-merging (silently dropping distinct claims) or
under-merging. We inject controlled embeddings so the cosine matrix is
deterministic, and stub the LLM so the soft-threshold branch is exercised
without a network call.
"""
from __future__ import annotations

import numpy as np
import pytest

import deduplicator


class _StubModel:
    """Returns the embedding assigned to each text via a lookup dict."""

    def __init__(self, mapping):
        self._mapping = mapping

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([self._mapping[t] for t in texts], dtype=float)


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    # Never let tests load the real sentence-transformer.
    monkeypatch.setattr(deduplicator, "_get_model", lambda: _no_real_model.model)


def _install_model(monkeypatch, mapping):
    _no_real_model.model = _StubModel(mapping)


def test_identical_embeddings_hard_merge(monkeypatch):
    a = "The AfD received 20 percent in the election."
    b = "In the election, the AfD got 20%."  # shorter
    _install_model(monkeypatch, {a: [1.0, 0.0], b: [1.0, 0.0]})  # cosine 1.0
    claims = [{"claim_text": a}, {"claim_text": b}]
    out = deduplicator.deduplicate(claims, threshold=0.85)
    assert len(out) == 1
    # keeps the longest text as representative
    assert out[0]["claim_text"] == a
    assert b in out[0]["duplicates"]


def test_orthogonal_embeddings_kept_separate(monkeypatch):
    a = "Germany imported 40% of its gas from Russia."
    b = "Oil prices rose to 120 dollars a barrel."
    _install_model(monkeypatch, {a: [1.0, 0.0], b: [0.0, 1.0]})  # cosine 0.0
    out = deduplicator.deduplicate([{"claim_text": a}, {"claim_text": b}])
    assert len(out) == 2


def test_soft_threshold_uses_llm(monkeypatch):
    a = "The government raised the defence budget significantly this year."
    b = "Defence spending went up under the current government."
    # cosine ~0.6 -> between SOFT_THRESHOLD (0.55) and hard 0.85
    v = 1 / np.sqrt(2)
    _install_model(monkeypatch, {a: [1.0, 0.0], b: [0.6, 0.8]})
    calls = []

    def fake_llm(x, y):
        calls.append((x, y))
        return "DUPLICATE"

    monkeypatch.setattr(deduplicator, "_llm_subsumption_check", fake_llm)
    out = deduplicator.deduplicate([{"claim_text": a}, {"claim_text": b}], threshold=0.85)
    assert calls, "LLM subsumption check should run for a soft-threshold pair"
    assert len(out) == 1


def test_soft_threshold_llm_says_distinct(monkeypatch):
    a = "The government raised the defence budget significantly this year."
    b = "Defence spending went up under the current government."
    _install_model(monkeypatch, {a: [1.0, 0.0], b: [0.6, 0.8]})
    monkeypatch.setattr(deduplicator, "_llm_subsumption_check", lambda x, y: "DISTINCT")
    out = deduplicator.deduplicate([{"claim_text": a}, {"claim_text": b}], threshold=0.85)
    assert len(out) == 2


def test_single_claim_returned_unchanged():
    assert deduplicator.deduplicate([{"claim_text": "only one"}]) == [
        {"claim_text": "only one"}
    ]
