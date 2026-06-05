"""Content-addressed result cache.

Keyed by the SHA-256 of the uploaded bytes so identical uploads return instantly
without re-running inference. Backed by Redis when reachable; otherwise it
transparently falls back to a process-local dict so the API still works (and
tests run) with no Redis server.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResultCache:
    def __init__(
        self, redis_url: str | None = None, *, ttl_seconds: int = 86_400, namespace: str = "veritas"
    ):
        self.ttl = ttl_seconds
        self.namespace = namespace
        self._redis = None
        self._memory: dict[str, str] = {}
        self.backend = "memory"
        if redis_url:
            self._try_connect(redis_url)

    def _try_connect(self, redis_url: str) -> None:
        try:
            import redis

            client = redis.Redis.from_url(redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            # Unreachable Redis must never break inference — degrade to memory.
            self._redis = None
            self.backend = "memory"

    def _key(self, digest: str) -> str:
        return f"{self.namespace}:{digest}"

    def get(self, digest: str) -> dict[str, Any] | None:
        raw: str | None
        if self._redis is not None:
            try:
                value = self._redis.get(self._key(digest))
                raw = value.decode("utf-8") if value else None
            except Exception:
                raw = None
        else:
            raw = self._memory.get(digest)
        return json.loads(raw) if raw else None

    def set(self, digest: str, value: dict[str, Any]) -> None:
        raw = json.dumps(value)
        if self._redis is not None:
            try:
                self._redis.setex(self._key(digest), self.ttl, raw)
                return
            except Exception:
                pass
        self._memory[digest] = raw


__all__ = ["ResultCache", "content_hash"]
