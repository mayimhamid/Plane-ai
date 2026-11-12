"""A small in-process TTL cache for answers to identical requests.

Asking for the same summary twice (a page reload, two managers opening the same analysis) costs nothing the
second time. Keys are hashes of the full request, so a changed input is a different entry.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int, max_items: int = 256) -> None:
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @staticmethod
    def key(*parts: object) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        if self.ttl <= 0:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if self.ttl <= 0:
            return
        self._items[key] = (time.monotonic() + self.ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
