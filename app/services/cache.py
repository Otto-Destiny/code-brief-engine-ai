from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class CacheItem(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int, max_entries: int = 128) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._items: OrderedDict[str, CacheItem[T]] = OrderedDict()

    async def get(self, key: str) -> T | None:
        self._purge_expired()
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item.value

    async def set(self, key: str, value: T) -> None:
        self._purge_expired()
        self._items[key] = CacheItem(value=value, expires_at=time.monotonic() + self._ttl_seconds)
        self._items.move_to_end(key)
        while len(self._items) > self._max_entries:
            self._items.popitem(last=False)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, item in self._items.items() if item.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
