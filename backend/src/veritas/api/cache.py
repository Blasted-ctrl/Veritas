"""Bounded, expiring result cache with a consistent in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResultCache:
    def __init__(
        self,
        redis_url: str | None = None,
        *,
        ttl_seconds: int = 86_400,
        namespace: str = "veritas",
        max_entries: int = 128,
        max_bytes: int = 67_108_864,
    ):
        self.ttl = ttl_seconds
        self.namespace = namespace
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._redis = None
        self._memory: OrderedDict[str, tuple[float, str, int]] = OrderedDict()
        self._memory_bytes = 0
        self.backend = "memory"
        if redis_url:
            self._try_connect(redis_url)

    def _try_connect(self, redis_url: str) -> None:
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url, socket_connect_timeout=0.5, socket_timeout=0.5, retry_on_timeout=False
            )
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            self._use_memory()

    def _use_memory(self) -> None:
        self._redis = None
        self.backend = "memory"

    def _key(self, digest: str) -> str:
        return f"{self.namespace}:{digest}"

    def _discard(self, digest: str) -> None:
        entry = self._memory.pop(digest, None)
        if entry:
            self._memory_bytes -= entry[2]

    def _expire(self) -> None:
        now = time.monotonic()
        for digest, (expires, _, _) in list(self._memory.items()):
            if expires <= now:
                self._discard(digest)

    def get(self, digest: str) -> dict[str, Any] | None:
        raw = None
        if self._redis is not None:
            try:
                value = self._redis.get(self._key(digest))
                raw = value.decode("utf-8") if value else None
            except Exception:
                self._use_memory()
        if self._redis is None:
            self._expire()
            entry = self._memory.get(digest)
            raw = entry[1] if entry else None
        if not raw:
            return None
        try:
            result = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return result if isinstance(result, dict) else None

    def set(self, digest: str, value: dict[str, Any]) -> None:
        raw = json.dumps(value)
        size = len(raw.encode("utf-8"))
        if self.ttl <= 0 or self.max_entries <= 0 or size > self.max_bytes:
            return
        if self._redis is not None:
            try:
                self._redis.setex(self._key(digest), self.ttl, raw)
                return
            except Exception:
                self._use_memory()
        self._expire()
        self._discard(digest)
        while self._memory and (
            len(self._memory) >= self.max_entries or self._memory_bytes + size > self.max_bytes
        ):
            self._discard(next(iter(self._memory)))
        self._memory[digest] = (time.monotonic() + self.ttl, raw, size)
        self._memory_bytes += size


__all__ = ["ResultCache", "content_hash"]
