class RateLimiter:
    """Simple fixed-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: float):
        raise NotImplementedError("Not implemented")

    def is_allowed(self, key: str, current_time: float | None = None) -> bool:
        raise NotImplementedError("Not implemented")

    def record(self, key: str, current_time: float | None = None) -> None:
        raise NotImplementedError("Not implemented")


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid. Implement manually, no libraries."""
    raise NotImplementedError("Not implemented")
