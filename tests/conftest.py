"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _disable_build_call_cache(monkeypatch):
    """Disable the resumable build-call cache for every test by default.

    The cache is keyed by request content and persists under outputs/, so
    leaving it on would let mocked-provider tests write into (and read stale
    responses from) the real cache and cross-contaminate each other. Tests that
    specifically exercise the cache re-enable it against a tmp dir.
    """
    monkeypatch.setenv("PILOT_BUILD_CALL_CACHE", "0")
