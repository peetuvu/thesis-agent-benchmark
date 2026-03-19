"""Tests for the model registry."""

from __future__ import annotations

import pytest

from providers.models import MODELS, get_model, list_models, resolve_model


class TestGetModel:
    def test_known_alias(self) -> None:
        info = get_model("sonnet-4.6")
        assert info["api_string"] == "claude-sonnet-4-6"
        assert info["provider"] == "anthropic"
        assert info["tier"] == "mid"

    def test_openai_alias(self) -> None:
        info = get_model("gpt-4o")
        assert info["api_string"] == "gpt-4o"
        assert info["provider"] == "openai"

    def test_unknown_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model alias"):
            get_model("nonexistent-model")

    def test_error_lists_available(self) -> None:
        with pytest.raises(KeyError, match="sonnet-4.6"):
            get_model("bogus")


class TestResolveModel:
    def test_registry_alias(self) -> None:
        api_string, provider = resolve_model("opus-4.6")
        assert api_string == "claude-opus-4-6"
        assert provider == "anthropic"

    def test_auto_provider_openai(self) -> None:
        api_string, provider = resolve_model("o3")
        assert api_string == "o3"
        assert provider == "openai"

    def test_raw_string_fallback(self) -> None:
        api_string, provider = resolve_model("some-custom-model-v2")
        assert api_string == "some-custom-model-v2"
        assert provider is None

    def test_all_registry_entries_resolve(self) -> None:
        for alias, info in MODELS.items():
            api_string, provider = resolve_model(alias)
            assert api_string == info["api_string"]
            assert provider == info["provider"]


class TestListModels:
    def test_contains_providers(self) -> None:
        output = list_models()
        assert "anthropic:" in output
        assert "openai:" in output

    def test_contains_model_aliases(self) -> None:
        output = list_models()
        assert "sonnet-4.6" in output
        assert "gpt-4o" in output
        assert "o3" in output

    def test_contains_tier_labels(self) -> None:
        output = list_models()
        assert "flagship" in output
        assert "mid" in output
        assert "fast" in output
