import time


class URLShortener:
    """In-memory URL shortener with expiration and click tracking."""

    def __init__(
        self,
        default_ttl_seconds: float = 3600.0,
        code_length: int = 6,
        rate_limit_requests: int = 10,
        rate_limit_window: float = 60.0,
    ):
        raise NotImplementedError("Not implemented")

    def shorten(
        self,
        url: str,
        client_ip: str = "127.0.0.1",
        custom_code: str | None = None,
        ttl_seconds: float | None = None,
        current_time: float | None = None,
    ) -> dict:
        raise NotImplementedError("Not implemented")

    def resolve(
        self,
        short_code: str,
        current_time: float | None = None,
    ) -> str:
        raise NotImplementedError("Not implemented")

    def get_stats(self, short_code: str) -> dict:
        raise NotImplementedError("Not implemented")

    def _generate_code(self) -> str:
        raise NotImplementedError("Not implemented")
