"""Tests for the content-hash result cache.

Standard-library only (no fastapi/redis needed): the cache falls back to an
in-memory backend, and an unreachable/absent Redis must degrade gracefully.
"""

from __future__ import annotations

from veritas.api.cache import ResultCache, content_hash


def test_content_hash_is_deterministic_and_distinct():
    assert content_hash(b"hello") == content_hash(b"hello")
    assert len(content_hash(b"hello")) == 64
    assert content_hash(b"hello") != content_hash(b"world")


def test_memory_backend_when_no_redis_url():
    cache = ResultCache(redis_url=None)
    assert cache.backend == "memory"


def test_set_get_roundtrip_memory():
    cache = ResultCache(redis_url=None)
    digest = content_hash(b"abc")
    assert cache.get(digest) is None
    cache.set(digest, {"verdict": "fake", "confidence": 0.9})
    assert cache.get(digest) == {"verdict": "fake", "confidence": 0.9}


def test_unreachable_redis_degrades_to_memory():
    # A bad port (or missing redis package) must never raise — just fall back.
    cache = ResultCache(redis_url="redis://127.0.0.1:6399/0")
    assert cache.backend == "memory"
    digest = content_hash(b"xyz")
    cache.set(digest, {"ok": True})
    assert cache.get(digest) == {"ok": True}
