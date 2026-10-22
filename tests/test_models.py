"""Tests for the model registry."""

from __future__ import annotations

import pytest

from providers.models import (
    MODELS,
    get_model,
    get_model_by_api_string,
    list_models,
    resolve_model,
)


class TestGetModel:
    def test_known_alias(self) -> None:
        info = get_model("sonnet-4.6")
        assert info["api_string"] == "claude-sonnet-4-6"
        assert info["provider"] == "anthropic"
        assert info["tier"] == "mid"

    def test_openai_alias(self) -> None:
        info = get_model("gpt-5.6-terra")
        assert info["api_string"] == "gpt-5.6-terra"
        assert info["provider"] == "openai"
        assert info["tier"] == "mid"

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
        api_string, provider = resolve_model("gpt-5.6-luna")
        assert api_string == "gpt-5.6-luna"
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
        assert "gpt-5.6-sol" in output

    def test_contains_tier_labels(self) -> None:
        output = list_models()
        assert "flagship" in output
        assert "mid" in output
        assert "fast" in output


class TestGetModelByApiString:
    def test_registered(self) -> None:
        info = get_model_by_api_string("gpt-5.6-sol")
        assert info is not None
        assert info["tier"] == "flagship"

    def test_unregistered_returns_none(self) -> None:
        assert get_model_by_api_string("gpt-4o") is None


class TestRegistryConsistency:
    def test_api_strings_are_lowercase(self) -> None:
        for alias, info in MODELS.items():
            assert info["api_string"] == info["api_string"].lower(), alias

    def test_tiers_use_shared_vocabulary(self) -> None:
        for alias, info in MODELS.items():
            assert info["tier"] in {"fast", "mid", "flagship"}, alias

    def test_each_provider_has_one_model_per_tier(self) -> None:
        for provider in ("anthropic", "openai"):
            tiers = sorted(i["tier"] for i in MODELS.values() if i["provider"] == provider)
            assert tiers == ["fast", "flagship", "mid"], provider

    def test_openai_models_declare_min_reasoning_effort(self) -> None:
        for alias, info in MODELS.items():
            if info["provider"] == "openai":
                assert info.get("min_reasoning_effort") == "none", alias
