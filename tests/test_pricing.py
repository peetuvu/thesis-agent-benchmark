"""Tests for the pricing table and per-call cost estimation."""

from __future__ import annotations

import pytest

from providers.models import LEGACY_ALIASES, MODELS
from providers.pricing import PRICES, estimate_call_cost, get_price


class TestPriceTable:
    def test_every_current_model_is_priced(self) -> None:
        for alias, info in MODELS.items():
            if alias not in LEGACY_ALIASES:
                assert get_price(info["api_string"]) is not None, alias

    def test_legacy_anthropic_models_priced_by_registry_api_string(self) -> None:
        for alias in ("opus-4.6", "sonnet-4.6", "haiku-4.5"):
            assert get_price(MODELS[alias]["api_string"]) is not None, alias

    def test_legacy_openai_models_not_priced(self) -> None:
        for alias in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert get_price(MODELS[alias]["api_string"]) is None, alias

    def test_every_price_cites_a_source(self) -> None:
        for model, price in PRICES.items():
            assert price.source.startswith("https://"), model
            assert "retrieved" in price.source, model

    @pytest.mark.parametrize("model,inp,out", [
        ("claude-haiku-5-5", 0.10, 0.50),
        ("claude-sonnet-5-5", 2.00, 10.00),
        ("claude-opus-5-5", 4.00, 20.00),
        ("gpt-6-luna", 0.10, 0.50),
        ("gpt-6.1-sol", 2.00, 10.00),
        ("gpt-6-astra", 10.00, 50.00),
        ("claude-opus-4-6", 5.00, 25.00),
        ("claude-sonnet-4-6", 3.00, 15.00),
        ("claude-haiku-4-5-20251001", 1.00, 5.00),
    ])
    def test_list_prices(self, model: str, inp: float, out: float) -> None:
        price = get_price(model)
        assert (price.input_per_mtok, price.output_per_mtok) == (inp, out)


class TestEstimateCallCost:
    def test_basic(self) -> None:
        # 1M in @ $4 + 0.5M out @ $20
        assert estimate_call_cost("claude-opus-5-5", 1_000_000, 500_000) == pytest.approx(14.0)

    def test_unpriced_model_returns_none(self) -> None:
        assert estimate_call_cost("gpt-5.6-terra", 1000, 1000) is None

    def test_haiku_short_prompt_rate(self) -> None:
        cost = estimate_call_cost("claude-haiku-5-5", 100_000, 10_000)
        assert cost == pytest.approx((100_000 * 0.10 + 10_000 * 0.50) / 1e6)

    def test_haiku_long_prompt_rate(self) -> None:
        cost = estimate_call_cost("claude-haiku-5-5", 100_001, 10_000)
        assert cost == pytest.approx((100_001 * 0.50 + 10_000 * 2.50) / 1e6)

    def test_openai_cached_input_discounted(self) -> None:
        cost = estimate_call_cost("gpt-6-astra", 10_000, 0, cached_input_tokens=4_000)
        assert cost == pytest.approx((6_000 * 10.00 + 4_000 * 1.00) / 1e6)

    def test_cached_ignored_without_cached_rate(self) -> None:
        a = estimate_call_cost("claude-opus-5-5", 10_000, 0, cached_input_tokens=4_000)
        assert a == pytest.approx(10_000 * 4.00 / 1e6)
