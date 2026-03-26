"""Hidden evaluation tests for the URL shortener task.

These tests are NOT visible to the benchmarked agent.
They cover edge cases, validation details, rate limiting,
deterministic code generation, and expiration semantics.
"""

import pytest
from urlshort import URLShortener
from urlshort.validators import is_valid_url, RateLimiter


# ---------------------------------------------------------------------------
# URL validation edge cases
# ---------------------------------------------------------------------------


def test_url_missing_dot_in_domain():
    """Domain without a dot should be rejected."""
    assert is_valid_url("http://localhost/path") is False


def test_url_domain_starts_with_hyphen():
    """Domain starting with a hyphen should be rejected."""
    assert is_valid_url("http://-example.com") is False


def test_url_domain_ends_with_hyphen():
    """Domain ending with a hyphen should be rejected."""
    assert is_valid_url("http://example-.com") is False


def test_url_with_port():
    """URL with a port number should be accepted."""
    assert is_valid_url("http://example.com:8080/path") is True


def test_url_whitespace():
    """URL containing whitespace should be rejected."""
    assert is_valid_url("http://example .com") is False


def test_url_scheme_only():
    """URL with scheme but empty domain should be rejected."""
    assert is_valid_url("http://") is False


def test_url_long_domain_segment():
    """Domain segment longer than 63 characters should be rejected."""
    long_segment = "a" * 64
    assert is_valid_url(f"http://{long_segment}.com") is False


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limit_allows_up_to_max():
    """Rate limiter should allow exactly max_requests, then deny."""
    limiter = RateLimiter(max_requests=3, window_seconds=10)
    key = "user1"

    for _ in range(3):
        assert limiter.is_allowed(key, current_time=1.0) is True
        limiter.record(key, current_time=1.0)

    assert limiter.is_allowed(key, current_time=1.0) is False


def test_rate_limit_window_reset():
    """After the window expires, requests should be allowed again."""
    limiter = RateLimiter(max_requests=3, window_seconds=10)
    key = "user1"

    for _ in range(3):
        assert limiter.is_allowed(key, current_time=0.0) is True
        limiter.record(key, current_time=0.0)

    assert limiter.is_allowed(key, current_time=0.0) is False
    # After the window resets
    assert limiter.is_allowed(key, current_time=15.0) is True


# ---------------------------------------------------------------------------
# Deterministic code generation
# ---------------------------------------------------------------------------


def test_deterministic_code_generation():
    """First code should be '000000', second should be '000001'."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    r1 = svc.shorten("https://example.com/a", current_time=1000.0)
    r2 = svc.shorten("https://example.com/b", current_time=1000.0)
    assert r1["short_code"] == "000000"
    assert r2["short_code"] == "000001"


def test_custom_code_collision():
    """Auto-generated code should skip codes already taken by custom codes."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    # Take "000000" with a custom code
    svc.shorten(
        "https://example.com/first",
        custom_code="000000",
        current_time=1000.0,
    )
    # Next auto-generated should skip "000000" and use "000001"
    r2 = svc.shorten("https://example.com/second", current_time=1000.0)
    assert r2["short_code"] == "000001"


def test_custom_code_case_sensitive():
    """Custom codes are case-sensitive: 'MyCode' and 'mycode' are distinct."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    svc.shorten(
        "https://example.com/upper",
        custom_code="MyCode",
        current_time=1000.0,
    )
    svc.shorten(
        "https://example.com/lower",
        custom_code="mycode",
        current_time=1000.0,
    )
    assert svc.resolve("MyCode", current_time=1000.0) == "https://example.com/upper"
    assert svc.resolve("mycode", current_time=1000.0) == "https://example.com/lower"


# ---------------------------------------------------------------------------
# Expiration and stats
# ---------------------------------------------------------------------------


def test_stats_after_expiration():
    """get_stats should still work after expiration, with is_expired=True."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten(
        "https://example.com/expstats",
        ttl_seconds=100,
        current_time=1000.0,
    )
    code = result["short_code"]
    # Resolve once before expiration
    svc.resolve(code, current_time=1050.0)
    # Now expired — get_stats should still work
    stats = svc.get_stats(code)
    assert stats["is_expired"] is True
    assert stats["clicks"] == 1


# ---------------------------------------------------------------------------
# Multiple shortens and uniqueness
# ---------------------------------------------------------------------------


def test_multiple_shortens_same_url():
    """Shortening the same URL twice should produce different short codes."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    r1 = svc.shorten("https://example.com/same", current_time=1000.0)
    r2 = svc.shorten("https://example.com/same", current_time=1000.0)
    assert r1["short_code"] != r2["short_code"]


# ---------------------------------------------------------------------------
# Custom code validation
# ---------------------------------------------------------------------------


def test_custom_code_non_alphanumeric_rejected():
    """Custom codes with non-alphanumeric characters should raise ValueError."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    with pytest.raises(ValueError):
        svc.shorten(
            "https://example.com/bad",
            custom_code="my-code",
            current_time=1000.0,
        )


# ---------------------------------------------------------------------------
# Click tracking
# ---------------------------------------------------------------------------


def test_resolve_increments_clicks():
    """Each resolve call should increment the click counter."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten("https://example.com/clicks", current_time=1000.0)
    code = result["short_code"]
    svc.resolve(code, current_time=1000.0)
    svc.resolve(code, current_time=1000.0)
    svc.resolve(code, current_time=1000.0)
    stats = svc.get_stats(code)
    assert stats["clicks"] == 3


# ---------------------------------------------------------------------------
# TTL override
# ---------------------------------------------------------------------------


def test_shorten_with_ttl_override():
    """Custom ttl_seconds should override the default TTL."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten(
        "https://example.com/ttl",
        ttl_seconds=200,
        current_time=1000.0,
    )
    assert result["created_at"] == 1000.0
    assert result["expires_at"] == 1200.0
