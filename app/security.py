import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse


class SlidingWindowLimiter:
    """In-process sliding-window rate limiter keyed by client address."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        hits = self._hits[key]
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) < self.max_requests:
            hits.append(now)
            return 0
        return max(1, int(hits[0] + self.window_seconds - now) + 1)


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limited_response(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": str(retry_after)},
    )
