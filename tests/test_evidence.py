"""Web-evidence assessment parsing.

The LLM returns free-form text with an ASSESSMENT: line; we must map it to
exactly one of SUPPORTS / REFUTES / INSUFFICIENT / UNRELATED and default
safely (INSUFFICIENT) on anything unexpected.
"""
from __future__ import annotations

from types import SimpleNamespace

from evidence import _assess_evidence


class _FakeClient:
    """Minimal stand-in for the OpenAI client returning a canned completion."""

    def __init__(self, content):
        self._content = content
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: response)
        )


_SNIPPET = [{"title": "Reuters", "url": "https://reuters.com/x", "content": "..."}]


def test_no_snippets_returns_insufficient():
    result = _assess_evidence("some claim", [], client=_FakeClient(""))
    assert result["assessment"] == "INSUFFICIENT"


def test_parses_supports():
    answer = "EVIDENCE: Official data confirms the figure.\nASSESSMENT: SUPPORTS"
    result = _assess_evidence("claim", _SNIPPET, client=_FakeClient(answer))
    assert result["assessment"] == "SUPPORTS"
    assert result["summary"] == "Official data confirms the figure."


def test_parses_refutes_case_insensitive():
    answer = "EVIDENCE: The real number is lower.\nassessment: refutes"
    result = _assess_evidence("claim", _SNIPPET, client=_FakeClient(answer))
    assert result["assessment"] == "REFUTES"


def test_unknown_label_defaults_to_insufficient():
    answer = "EVIDENCE: unclear\nASSESSMENT: MAYBE"
    result = _assess_evidence("claim", _SNIPPET, client=_FakeClient(answer))
    assert result["assessment"] == "INSUFFICIENT"


def test_missing_assessment_line_defaults_insufficient():
    answer = "EVIDENCE: just a summary, no verdict line"
    result = _assess_evidence("claim", _SNIPPET, client=_FakeClient(answer))
    assert result["assessment"] == "INSUFFICIENT"
