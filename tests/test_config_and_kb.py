"""Config validation and the optional-KB guard.

validate_config must fail fast when the one required key is missing, and the
private-KB search must be a no-op unless both its URL and key are configured
(so the pipeline runs cleanly for users without that private endpoint).
"""
from __future__ import annotations

import pytest

import config
import fact_checker


def test_validate_config_raises_without_openai_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    with pytest.raises(SystemExit):
        config.validate_config()


def test_validate_config_passes_with_openai_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    # Should not raise; optional integrations just get logged.
    config.validate_config()


def test_kb_search_skipped_when_unconfigured(monkeypatch):
    monkeypatch.setattr(fact_checker, "PRIVATE_KB_API_URL", "")
    monkeypatch.setattr(fact_checker, "PRIVATE_KB_API_KEY", "")
    assert fact_checker._search_kb("any query") == []


def test_kb_search_skipped_when_only_key_set(monkeypatch):
    # URL missing -> still skip, no request attempted.
    monkeypatch.setattr(fact_checker, "PRIVATE_KB_API_URL", "")
    monkeypatch.setattr(fact_checker, "PRIVATE_KB_API_KEY", "secret")

    def _boom(*a, **k):
        raise AssertionError("no HTTP request should be made when KB is unconfigured")

    monkeypatch.setattr(fact_checker.requests, "post", _boom)
    assert fact_checker._search_kb("any query") == []
