# URL Shortener — Benchmark Task

Implement an in-memory URL shortener service in:

- `urlshort/service.py`
- `urlshort/validators.py`

You must implement **exactly** the classes and functions specified below. Do not change the file structure. Do not add dependencies. Do not modify tests.

The evaluator will run both public sanity tests and separate hidden tests.

---

## Overview

Build an in-memory URL shortener that creates short codes for long URLs, tracks clicks, handles expiration, and enforces rate limits. All state is stored in memory (no database, no files).

---

## Validators — `validators.py`

```python
def is_valid_url(url: str) -> bool:
    """Check if a URL is valid.

    A valid URL must:
    - Start with http:// or https://
    - Have a non-empty domain after the scheme
    - The domain must contain at least one dot
    - The domain must not start or end with a dot or hyphen
    - Only contain valid URL characters

    Returns True if valid, False otherwise.
    Do NOT use urllib.parse or any URL parsing library.
    Implement validation manually with string operations.
    """
    ...

class RateLimiter:
    """Simple fixed-window rate limiter.

    Tracks the number of operations per key within a time window.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum allowed requests per window per key.
            window_seconds: Length of the time window in seconds.
        """
        ...

    def is_allowed(self, key: str, current_time: float | None = None) -> bool:
        """Check if a request is allowed for the given key.

        If current_time is None, use time.time().
        Returns True if the request is within limits, False if rate limited.
        Each call to is_allowed (even if denied) does NOT count as a request.
        Only calls that return True count toward the limit.
        """
        ...

    def record(self, key: str, current_time: float | None = None) -> None:
        """Record a successful request for the given key.

        Call this AFTER a request has been allowed and processed.
        """
        ...
```

URL validation rules (implement manually, no libraries):
- Must start with `http://` or `https://`
- After the scheme, extract the domain (everything up to the first `/`, `?`, `#`, or end of string)
- Domain must contain at least one `.`
- Domain must not start or end with `.` or `-`
- Domain segments (split by `.`) must each be 1-63 characters
- Domain characters: alphanumeric and hyphens only
- Port numbers after `:` are allowed (e.g., `http://localhost:8080/path`)
- Path, query, fragment after the domain are allowed but not validated in detail
- Empty string → False
- Whitespace in the URL → False

---

## Service — `service.py`

```python
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
        """Initialize the URL shortener.

        Args:
            default_ttl_seconds: Default time-to-live for shortened URLs.
            code_length: Length of generated short codes.
            rate_limit_requests: Max shorten requests per window per IP.
            rate_limit_window: Rate limit window in seconds.
        """
        ...

    def shorten(
        self,
        url: str,
        client_ip: str = "127.0.0.1",
        custom_code: str | None = None,
        ttl_seconds: float | None = None,
        current_time: float | None = None,
    ) -> dict:
        """Create a shortened URL.

        Args:
            url: The long URL to shorten.
            client_ip: Client IP for rate limiting.
            custom_code: Optional custom short code. If None, generate one.
            ttl_seconds: Optional TTL override. If None, use default_ttl_seconds.
            current_time: Optional timestamp override (for testing). If None, use time.time().

        Returns:
            dict with keys:
                "short_code": str — the generated or custom code
                "url": str — the original URL
                "created_at": float — creation timestamp
                "expires_at": float — expiration timestamp

        Raises:
            ValueError: If URL is invalid, code already exists, code contains
                        non-alphanumeric characters, or rate limit exceeded.
        """
        ...

    def resolve(
        self,
        short_code: str,
        current_time: float | None = None,
    ) -> str:
        """Look up the original URL for a short code.

        Increments the click counter for this code.

        Args:
            short_code: The short code to look up.
            current_time: Optional timestamp override. If None, use time.time().

        Returns:
            The original URL string.

        Raises:
            KeyError: If the code does not exist.
            ValueError: If the link has expired. The error message must
                        contain "expired".
        """
        ...

    def get_stats(self, short_code: str) -> dict:
        """Get statistics for a shortened URL.

        Returns:
            dict with keys:
                "short_code": str
                "url": str
                "created_at": float
                "expires_at": float
                "clicks": int
                "is_expired": bool (check against current time)

        Raises:
            KeyError: If the code does not exist.
        """
        ...

    def _generate_code(self) -> str:
        """Generate a unique short code.

        The code must be exactly self.code_length characters long,
        consisting of lowercase letters and digits.
        Must not collide with any existing code.
        Use a deterministic approach: hash-based, not random.
        Specifically: maintain an internal counter, and for each new code,
        convert the counter value to a base-36 string (0-9, a-z),
        left-pad with zeros to code_length. Increment counter after each use.

        Raises:
            RuntimeError: If all possible codes are exhausted.
        """
        ...
```

Short code generation rules:
- Deterministic: based on an incrementing counter converted to base-36
- Counter starts at 0 → code `"000000"`, counter 1 → `"000001"`, ..., counter 36 → `"000010"`
- Left-padded with zeros to `code_length`
- Custom codes must be purely alphanumeric (a-z, A-Z, 0-9). Reject others with ValueError.
- Custom codes are case-sensitive: `"MyCode"` and `"mycode"` are different codes.
- If a generated code collides with an existing code (custom or generated), skip it and try the next counter value.

Expiration rules:
- Each link has a `created_at` and `expires_at` timestamp
- `resolve()` checks expiration: if `current_time >= expires_at`, raise ValueError
- `get_stats()` reports `is_expired` based on current time but does NOT prevent access to stats
- Expired links remain in storage (they are not garbage collected)

Rate limiting:
- `shorten()` is rate-limited per `client_ip`
- Check rate limit BEFORE validating the URL
- If rate limited, raise ValueError with a message containing "rate limit"
- `resolve()` is NOT rate limited

---

## Examples

```python
service = URLShortener(default_ttl_seconds=3600, code_length=6)

# Shorten a URL
result = service.shorten("https://example.com/very/long/path")
# result = {"short_code": "000000", "url": "https://example.com/very/long/path",
#           "created_at": <timestamp>, "expires_at": <timestamp + 3600>}

# Resolve it
url = service.resolve(result["short_code"])
# url = "https://example.com/very/long/path"

# Check stats
stats = service.get_stats(result["short_code"])
# stats["clicks"] = 1

# Invalid URL
service.shorten("not-a-url")  # raises ValueError

# Custom code
result2 = service.shorten("https://example.com", custom_code="mysite")
service.resolve("mysite")  # returns "https://example.com"
```

---

## Output format

Return complete file contents in labeled code blocks:

```urlshort/validators.py
# complete file contents here
```

```urlshort/service.py
# complete file contents here
```
