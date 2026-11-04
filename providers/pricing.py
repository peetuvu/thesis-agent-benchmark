"""Per-token prices for cost estimation (USD per million tokens).

Keyed by API model string, so the model registry entries stay unchanged.
Each price records where it came from and when, because list prices change
and a run log's estimated_cost_usd is only as good as the table it used.

Prices are standard (non-batch, non-priority, no regional/US-only multiplier)
first-party API list prices. Models without an entry are not priced: their
runs log estimated_cost_usd=0.0 with price_source=None.
"""

from __future__ import annotations

from dataclasses import dataclass

_ANTHROPIC_SRC = "https://platform.claude.com/docs/en/about-claude/pricing (retrieved 2026-11-04)"
_OPENAI_SRC = "https://developers.openai.com/api/docs/models/{model} (retrieved 2026-11-04)"


@dataclass(frozen=True)
class Price:
    """List price for one model, USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    source: str
    cached_input_per_mtok: float | None = None
    """Price for cached input tokens (OpenAI automatic caching). None = bill at input rate."""
    long_prompt_threshold: int | None = None
    """Prompts with MORE input tokens than this are billed at the long_* rates."""
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None


PRICES: dict[str, Price] = {
    # Anthropic. The harness sends no cache_control, so no cache read/write rates.
    "claude-opus-5-5": Price(4.00, 20.00, _ANTHROPIC_SRC),
    "claude-sonnet-5-5": Price(2.00, 10.00, _ANTHROPIC_SRC),
    # Haiku 5.5 is priced by prompt length per request: "a request whose
    # prompt is over 100,000 tokens pays higher prices".
    "claude-haiku-5-5": Price(
        0.10, 0.50, _ANTHROPIC_SRC,
        long_prompt_threshold=100_000, long_input_per_mtok=0.50, long_output_per_mtok=2.50,
    ),
    # Anthropic legacy 4.x (registry aliases opus-4.6, sonnet-4.6, haiku-4.5),
    # for costing existing runs in the generation comparison. Flat rates: the
    # 4.6 models bill the full 1M context at standard pricing, and Haiku 4.5
    # has no prompt-length tier. Keys match the API strings in the run logs.
    "claude-opus-4-6": Price(5.00, 25.00, _ANTHROPIC_SRC),
    "claude-sonnet-4-6": Price(3.00, 15.00, _ANTHROPIC_SRC),
    "claude-haiku-4-5-20251001": Price(1.00, 5.00, _ANTHROPIC_SRC),
    # OpenAI. Chat Completions caches prompt prefixes automatically, and
    # usage.prompt_tokens_details.cached_tokens is billed at the cached rate.
    "gpt-6-luna": Price(0.10, 0.50, _OPENAI_SRC.format(model="gpt-6-luna"), cached_input_per_mtok=0.01),
    "gpt-6.1-sol": Price(2.00, 10.00, _OPENAI_SRC.format(model="gpt-6.1-sol"), cached_input_per_mtok=0.10),
    "gpt-6-astra": Price(10.00, 50.00, _OPENAI_SRC.format(model="gpt-6-astra"), cached_input_per_mtok=1.00),
}


def get_price(api_string: str) -> Price | None:
    """Return the price entry for a model, or None if it is not priced."""
    return PRICES.get(api_string)


def estimate_call_cost(
    api_string: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Estimated USD cost of one API call, or None if the model is not priced.

    input_tokens is the provider's total prompt count; cached_input_tokens is
    the subset of it served from cache (billed at the cached rate if the model
    has one). output_tokens includes thinking/reasoning tokens on both
    providers, which are billed as output.
    """
    price = PRICES.get(api_string)
    if price is None:
        return None
    in_rate, out_rate = price.input_per_mtok, price.output_per_mtok
    if price.long_prompt_threshold is not None and input_tokens > price.long_prompt_threshold:
        in_rate = price.long_input_per_mtok or in_rate
        out_rate = price.long_output_per_mtok or out_rate
    cached = min(cached_input_tokens, input_tokens) if price.cached_input_per_mtok is not None else 0
    cost = (
        (input_tokens - cached) * in_rate
        + cached * (price.cached_input_per_mtok or 0.0)
        + output_tokens * out_rate
    )
    return cost / 1_000_000
