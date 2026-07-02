"""Transcript segment merging and classifier label routing.

Segment merging turns raw Whisper segments into sentence-sized windows;
off-by-one or bad short-merge logic corrupts every downstream step. Label
routing picks which model output column is the check-worthy class — a wrong
guess silently inverts the classifier.
"""
from __future__ import annotations

from claim_detector import _merge_segments_into_sentences, _resolve_cfs_label


def _seg(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_merges_until_sentence_end():
    segments = [
        _seg("Die Inflation", 0.0, 1.0),
        _seg("ist gestiegen.", 1.0, 2.0),
        _seg("Das ist ein weiterer vollständiger Satz mit vielen Woertern hier.", 2.0, 3.0),
    ]
    sentences = _merge_segments_into_sentences(segments, min_words=3)
    assert sentences[0]["text"] == "Die Inflation ist gestiegen."
    assert sentences[0]["start"] == 0.0
    assert sentences[0]["end"] == 2.0
    assert sentences[0]["segment_indices"] == [0, 1]


def test_short_trailing_sentence_merged_back():
    segments = [
        _seg("Der Wehrdienst dauert nach dem neuen Modell eine gewisse Zeit.", 0.0, 2.0),
        _seg("Auf 140 Tage.", 2.0, 3.0),  # < min_words -> merge into previous
    ]
    sentences = _merge_segments_into_sentences(segments, min_words=6)
    assert len(sentences) == 1
    assert sentences[0]["text"].endswith("Auf 140 Tage.")
    assert sentences[0]["end"] == 3.0
    assert sentences[0]["segment_indices"] == [0, 1]


def test_word_limit_forces_break():
    long_seg = _seg(" ".join(["wort"] * 50) + " ende", 0.0, 5.0)
    sentences = _merge_segments_into_sentences([long_seg], max_words=40)
    assert len(sentences) == 1  # single segment can't be split, but still emitted


def test_empty_segments_skipped():
    segments = [_seg("  ", 0.0, 1.0), _seg("Ein echter Satz hier drin.", 1.0, 2.0)]
    sentences = _merge_segments_into_sentences(segments, min_words=3)
    assert len(sentences) == 1
    assert sentences[0]["segment_indices"] == [1]


def test_resolve_cfs_label_matches_checkworthy():
    id2label = {0: "Non-Factual Sentence", 1: "Unimportant Factual Sentence",
                2: "Check-worthy Factual Sentence"}
    assert _resolve_cfs_label(id2label) == 2


def test_resolve_cfs_label_secondary_keywords():
    assert _resolve_cfs_label({0: "no", 1: "claim"}) == 1


def test_resolve_cfs_label_fallback_to_max():
    # No recognisable label -> fall back to the highest index.
    assert _resolve_cfs_label({0: "foo", 1: "bar", 2: "baz"}) == 2
