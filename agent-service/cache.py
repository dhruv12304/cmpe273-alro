"""Inventory cache with TTL for the Agent Service."""

from __future__ import annotations

import time
from typing import Optional


class InventoryCache:
    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._data: Optional[dict] = None
        self._timestamp: Optional[float] = None

    def set(self, data: dict) -> None:
        """Store inventory snapshot with current timestamp."""
        self._data = dict(data)
        self._timestamp = time.monotonic()

    def get(self) -> tuple[Optional[dict], bool]:
        """
        Returns (data, is_stale).
        is_stale=True if age > ttl_seconds OR no data has been set.
        data=None only if cache has never been populated.
        """
        if self._data is None or self._timestamp is None:
            return None, True
        age = time.monotonic() - self._timestamp
        is_stale = age > self._ttl
        return dict(self._data), is_stale

    def invalidate(self) -> None:
        """Clear cache, forcing next get() to re-fetch from Inventory Service."""
        self._data = None
        self._timestamp = None
