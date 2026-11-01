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


def test_memory_entries_expire(monkeypatch):
    import veritas.api.cache as cache_module

    clock = [100.0]
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock[0])
    cache = ResultCache(ttl_seconds=5)
    cache.set("image", {"score": 0.8})
    clock[0] = 105.0
    assert cache.get("image") is None


def test_memory_evicts_oldest_when_full():
    cache = ResultCache(max_entries=2)
    cache.set("old", {"score": 0.1})
    cache.set("middle", {"score": 0.2})
    cache.set("new", {"score": 0.3})
    assert cache.get("old") is None
    assert cache.get("middle") == {"score": 0.2}
    assert cache.get("new") == {"score": 0.3}


def test_redis_write_failure_switches_reads_to_memory(monkeypatch):
    import sys
    from types import SimpleNamespace

    class BrokenRedis:
        def ping(self):
            return True

        def setex(self, *args):
            raise ConnectionError("disconnected")

        def get(self, *args):
            return None

    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(Redis=SimpleNamespace(from_url=lambda *args, **kwargs: BrokenRedis())),
    )
    cache = ResultCache(redis_url="redis://localhost:6379/0")
    cache.set("image", {"score": 0.7})
    assert cache.backend == "memory"
    assert cache.get("image") == {"score": 0.7}
