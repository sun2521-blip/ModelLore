"""A small in-memory, per-client rate limiter for the expensive endpoints.

This is a basic guard for a single local process, not a distributed limiter:
counts live in memory, reset on restart, and are not shared between workers.
"""

import math
import threading
import time
from collections.abc import Callable

from fastapi import HTTPException, Request

from api.limits import RATE_LIMIT_WINDOW_SECONDS, RATE_LIMITS


class FixedWindowRateLimiter:
    """Allow up to ``limit`` hits per key in each fixed time window."""

    def __init__(
        self,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_keys: int = 10_000,
    ) -> None:
        self._window_seconds = window_seconds
        self._clock = clock
        self._max_tracked_keys = max_tracked_keys
        self._windows: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> float | None:
        """Count one request. Return seconds to wait if over the limit, else None."""
        now = self._clock()

        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))

            if now - window_start >= self._window_seconds:
                window_start, count = now, 0

            if count >= limit:
                return self._window_seconds - (now - window_start)

            if key not in self._windows and len(self._windows) >= self._max_tracked_keys:
                self._forget_expired(now)

            self._windows[key] = (window_start, count + 1)
            return None

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()

    def _forget_expired(self, now: float) -> None:
        expired = [
            key
            for key, (window_start, _) in self._windows.items()
            if now - window_start >= self._window_seconds
        ]
        for key in expired:
            del self._windows[key]


limiter = FixedWindowRateLimiter(RATE_LIMIT_WINDOW_SECONDS)


def rate_limit(bucket: str) -> Callable[[Request], None]:
    """A FastAPI dependency that limits one bucket of endpoints per client IP."""

    def check(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        retry_after = limiter.hit(f"{bucket}:{client}", RATE_LIMITS[bucket])

        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests; try again shortly",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    return check
