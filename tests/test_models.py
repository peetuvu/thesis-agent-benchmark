"""Tests for the model registry."""

from __future__ import annotations

import pytest

from providers.models import (
    LEGACY_ALIASES,
    MODELS,
    REASONING_EFFORTS,
    check_reasoning_effort,
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

    @pytest.mark.parametrize("legacy", [False, True], ids=["current", "legacy"])
    def test_each_provider_has_one_model_per_tier(self, legacy: bool) -> None:
        for provider in ("anthropic", "openai"):
            tiers = sorted(
                i["tier"] for a, i in MODELS.items()
                if i["provider"] == provider and (a in LEGACY_ALIASES) == legacy
            )
            assert tiers == ["fast", "flagship", "mid"], provider

    def test_legacy_openai_models_declare_min_reasoning_effort(self) -> None:
        for alias, info in MODELS.items():
            if info["provider"] == "openai" and alias in LEGACY_ALIASES:
                assert info.get("min_reasoning_effort") == "none", alias

    def test_legacy_aliases_exist(self) -> None:
        assert LEGACY_ALIASES <= set(MODELS)

    def test_current_models_require_reasoning_effort_and_accept_low(self) -> None:
        for alias, info in MODELS.items():
            if alias not in LEGACY_ALIASES:
                assert info.get("requires_reasoning_effort") is True, alias
                assert "low" in info["reasoning_efforts"], alias
                assert set(info["reasoning_efforts"]) <= set(REASONING_EFFORTS), alias


class TestLegacyEntriesFrozen:
    """Existing run logs reference these entries; they must never change."""

    FROZEN = {
        "opus-4.6": {"api_string": "claude-opus-4-6", "provider": "anthropic", "tier": "flagship", "description": "The most intelligent model for building agents and coding"},
        "sonnet-4.6": {"api_string": "claude-sonnet-4-6", "provider": "anthropic", "tier": "mid", "description": "The best combination of speed and intelligence"},
        "haiku-4.5": {"api_string": "claude-haiku-4-5-20251001", "provider": "anthropic", "tier": "fast", "description": "The fastest model with near-frontier intelligence"},
        "gpt-5.6-sol": {"api_string": "gpt-5.6-sol", "provider": "openai", "tier": "flagship", "min_reasoning_effort": "none", "description": "Most capable GPT-5.6 model ($4/$20 per MTok)"},
        "gpt-5.6-terra": {"api_string": "gpt-5.6-terra", "provider": "openai", "tier": "mid", "min_reasoning_effort": "none", "description": "Balanced GPT-5.6 model ($2/$12 per MTok)"},
        "gpt-5.6-luna": {"api_string": "gpt-5.6-luna", "provider": "openai", "tier": "fast", "min_reasoning_effort": "none", "description": "Fast, low-cost GPT-5.6 model ($0.2/$1.2 per MTok)"},
    }

    @pytest.mark.parametrize("alias", sorted(FROZEN))
    def test_entry_unchanged(self, alias: str) -> None:
        assert MODELS[alias] == self.FROZEN[alias]

    def test_frozen_set_is_the_legacy_set(self) -> None:
        assert set(self.FROZEN) == LEGACY_ALIASES


class TestCurrentLineup:
    @pytest.mark.parametrize("alias,api_string,provider,tier", [
        ("haiku-5.5", "claude-haiku-5-5", "anthropic", "fast"),
        ("sonnet-5.5", "claude-sonnet-5-5", "anthropic", "mid"),
        ("opus-5.5", "claude-opus-5-5", "anthropic", "flagship"),
        ("gpt-6-luna", "gpt-6-luna", "openai", "fast"),
        ("gpt-6.1-sol", "gpt-6.1-sol", "openai", "mid"),
        ("gpt-6-astra", "gpt-6-astra", "openai", "flagship"),
    ])
    def test_alias_resolves(self, alias: str, api_string: str, provider: str, tier: str) -> None:
        assert resolve_model(alias) == (api_string, provider)
        assert get_model(alias)["tier"] == tier

    def test_list_models_marks_legacy(self) -> None:
        lines = list_models().splitlines()
        assert any("opus-4.6" in l and "(legacy)" in l for l in lines)
        assert not any("opus-5.5" in l and "(legacy)" in l for l in lines)


class TestCheckReasoningEffort:
    def test_current_model_accepts_low(self) -> None:
        for alias in ("haiku-5.5", "sonnet-5.5", "opus-5.5", "gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"):
            check_reasoning_effort(get_model(alias)["api_string"], "low")

    def test_current_model_requires_effort(self) -> None:
        with pytest.raises(ValueError, match="requires an explicit reasoning effort"):
            check_reasoning_effort("claude-opus-5-5", None)

    @pytest.mark.parametrize("model,effort", [
        ("gpt-6-astra", "none"), ("gpt-6.1-sol", "none"), ("claude-opus-5-5", "none"),
        ("claude-haiku-5-5", "minimal"),
    ])
    def test_unsupported_effort_rejected(self, model: str, effort: str) -> None:
        with pytest.raises(ValueError, match="does not accept reasoning effort"):
            check_reasoning_effort(model, effort)

    def test_legacy_model_without_effort_ok(self) -> None:
        check_reasoning_effort("claude-sonnet-4-6", None)
        check_reasoning_effort("gpt-5.6-terra", None, thinking_enabled=True)

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gpt-5.6-terra", "unregistered-x"])
    def test_effort_rejected_for_models_without_declared_support(self, model: str) -> None:
        with pytest.raises(ValueError, match="no verified reasoning-effort support"):
            check_reasoning_effort(model, "low")

    def test_effort_and_thinking_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            check_reasoning_effort("claude-opus-5-5", "low", thinking_enabled=True)
