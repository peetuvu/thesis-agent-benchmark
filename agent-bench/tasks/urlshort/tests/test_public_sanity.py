"""Public sanity tests for the URL shortener task."""

import pytest
from urlshort import URLShortener


def test_shorten_valid_url():
    """Shorten a valid URL and verify the result dict structure."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten("https://example.com/path", current_time=1000.0)
    assert isinstance(result, dict)
    assert "short_code" in result
    assert "url" in result
    assert "created_at" in result
    assert "expires_at" in result
    assert result["url"] == "https://example.com/path"


def test_resolve_returns_original():
    """Resolve a short code and verify it returns the original URL."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten("https://example.com/long/path", current_time=1000.0)
    original = svc.resolve(result["short_code"], current_time=1000.0)
    assert original == "https://example.com/long/path"


def test_clicks_counted():
    """Resolve twice and verify click count is 2."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten("https://example.com/click", current_time=1000.0)
    code = result["short_code"]
    svc.resolve(code, current_time=1000.0)
    svc.resolve(code, current_time=1000.0)
    stats = svc.get_stats(code)
    assert stats["clicks"] == 2


def test_invalid_url_rejected():
    """Shorten an invalid URL and expect ValueError."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    with pytest.raises(ValueError):
        svc.shorten("not-a-url", current_time=1000.0)


def test_custom_code_works():
    """Shorten with a custom code and resolve it."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten(
        "https://example.com/custom",
        custom_code="mycode",
        current_time=1000.0,
    )
    assert result["short_code"] == "mycode"
    original = svc.resolve("mycode", current_time=1000.0)
    assert original == "https://example.com/custom"


def test_expired_link_raises():
    """Resolve an expired link and expect ValueError containing 'expired'."""
    svc = URLShortener(default_ttl_seconds=3600, code_length=6)
    result = svc.shorten(
        "https://example.com/expire",
        ttl_seconds=100,
        current_time=1000.0,
    )
    code = result["short_code"]
    with pytest.raises(ValueError, match="expired"):
        svc.resolve(code, current_time=1200.0)
