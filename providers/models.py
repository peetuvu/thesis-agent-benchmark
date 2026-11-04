"""Model registry for cross-provider model comparison.

Maps human-friendly aliases to API model strings and metadata.
"""

from __future__ import annotations

from typing import Any


ModelInfo = dict[str, Any]

MODELS: dict[str, ModelInfo] = {
    # Anthropic
    "opus-4.6": {"api_string": "claude-opus-4-6", "provider": "anthropic", "tier": "flagship", "description": "The most intelligent model for building agents and coding"},
    "sonnet-4.6": {"api_string": "claude-sonnet-4-6", "provider": "anthropic", "tier": "mid", "description": "The best combination of speed and intelligence"},
    "haiku-4.5": {"api_string": "claude-haiku-4-5-20251001", "provider": "anthropic", "tier": "fast", "description": "The fastest model with near-frontier intelligence"},
    # OpenAI — GPT-5.6 family (verified against developers.openai.com, 2026-10-22).
    # These are reasoning models; "min_reasoning_effort" is the lowest effort the
    # model accepts. With "none" they run effectively non-reasoning and the
    # provider sends temperature (see providers/openai.py). OpenAI documents
    # temperature-at-"none" for GPT-6; for GPT-5.6 it is confirmed by smoke test.
    "gpt-5.6-sol": {"api_string": "gpt-5.6-sol", "provider": "openai", "tier": "flagship", "min_reasoning_effort": "none", "description": "Most capable GPT-5.6 model ($4/$20 per MTok)"},
    "gpt-5.6-terra": {"api_string": "gpt-5.6-terra", "provider": "openai", "tier": "mid", "min_reasoning_effort": "none", "description": "Balanced GPT-5.6 model ($2/$12 per MTok)"},
    "gpt-5.6-luna": {"api_string": "gpt-5.6-luna", "provider": "openai", "tier": "fast", "min_reasoning_effort": "none", "description": "Fast, low-cost GPT-5.6 model ($0.2/$1.2 per MTok)"},
    # ------------------------------------------------------------------
    # Current lineup (added 2026-11-04). Entries above are kept unchanged
    # because existing run logs reference them; see LEGACY_ALIASES.
    #
    # These models are run only with an explicit reasoning effort
    # (RunConfig.reasoning_effort / `bench run --reasoning-effort`):
    #   "reasoning_efforts": efforts the model accepts (verified per model page)
    #   "requires_reasoning_effort": the harness refuses to run the model
    #       without one, because the legacy request shape is rejected (400).
    # No temperature is ever sent on the reasoning-effort path.
    # ------------------------------------------------------------------
    # Anthropic 5.5 generation — verified 2026-11-04 against
    # platform.claude.com/docs/en/about-claude/models/overview and the
    # per-model migration guides. All three reject temperature/top_p/top_k
    # and thinking.budget_tokens (HTTP 400); thinking is adaptive, depth is
    # set with output_config.effort.
    "haiku-5.5": {"api_string": "claude-haiku-5-5", "provider": "anthropic", "tier": "fast", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Fastest current Claude model; adaptive thinking ($0.10/$0.50 per MTok, prompts <=100K)"},
    "sonnet-5.5": {"api_string": "claude-sonnet-5-5", "provider": "anthropic", "tier": "mid", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Current Sonnet: speed/intelligence balance; adaptive thinking ($2/$10 per MTok)"},
    "opus-5.5": {"api_string": "claude-opus-5-5", "provider": "anthropic", "tier": "flagship", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Current Opus: agentic coding; thinking always on ($4/$20 per MTok)"},
    # OpenAI GPT-6 family — verified 2026-11-04 against
    # developers.openai.com/api/docs/models/<id> and guides/latest-model.
    # Temperature is accepted only at reasoning_effort "none", which
    # gpt-6-astra and gpt-6.1-sol do not support.
    "gpt-6-luna": {"api_string": "gpt-6-luna", "provider": "openai", "tier": "fast", "reasoning_efforts": ("none", "low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Most efficient GPT-6 model ($0.10/$0.50 per MTok)"},
    "gpt-6.1-sol": {"api_string": "gpt-6.1-sol", "provider": "openai", "tier": "mid", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Near-Astra performance at lower cost; no 'none' effort ($2/$10 per MTok)"},
    "gpt-6-astra": {"api_string": "gpt-6-astra", "provider": "openai", "tier": "flagship", "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"), "requires_reasoning_effort": True, "description": "Most capable GPT-6 model; no 'none' effort ($10/$50 per MTok)"},
}

# Aliases kept for reproducibility of existing run logs but superseded by the
# current lineup. Listed separately so the entries themselves stay unchanged.
LEGACY_ALIASES: frozenset[str] = frozenset({
    "opus-4.6", "sonnet-4.6", "haiku-4.5",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
})

# Every reasoning-effort value any provider SDK accepts; the CLI's choices.
# Which ones a given model accepts is its registry "reasoning_efforts".
REASONING_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)


def check_reasoning_effort(
    api_string: str, reasoning_effort: str | None, thinking_enabled: bool = False,
) -> None:
    """Validate a requested reasoning effort for a model before any API call.

    Raises ValueError if:
      - reasoning_effort is combined with thinking_enabled (two thinking controls);
      - reasoning_effort is set for a model that does not declare
        "reasoning_efforts" (legacy or unregistered), or is not in that list;
      - the model declares "requires_reasoning_effort" and none was given.
    """
    info = get_model_by_api_string(api_string)
    if reasoning_effort is None:
        if info and info.get("requires_reasoning_effort"):
            raise ValueError(
                f"Model '{api_string}' requires an explicit reasoning effort "
                f"(--reasoning-effort; accepted: {', '.join(info['reasoning_efforts'])})"
            )
        return
    if thinking_enabled:
        raise ValueError(
            "reasoning_effort and thinking_enabled are mutually exclusive; "
            "use --reasoning-effort alone"
        )
    accepted = info.get("reasoning_efforts") if info else None
    if not accepted:
        raise ValueError(
            f"Model '{api_string}' has no verified reasoning-effort support in the "
            "registry; --reasoning-effort applies only to models that declare it"
        )
    if reasoning_effort not in accepted:
        raise ValueError(
            f"Model '{api_string}' does not accept reasoning effort "
            f"'{reasoning_effort}' (accepted: {', '.join(accepted)})"
        )


def get_model(alias: str) -> ModelInfo:
    """Look up a model by alias. Raises KeyError if not found."""
    if alias not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise KeyError(f"Unknown model alias '{alias}'. Available: {available}")
    return MODELS[alias]


def get_model_by_api_string(api_string: str) -> ModelInfo | None:
    """Look up registry metadata by API model string. Returns None if unregistered."""
    for info in MODELS.values():
        if info["api_string"] == api_string:
            return info
    return None


def resolve_model(alias_or_raw: str) -> tuple[str, str | None]:
    """Resolve an alias or raw model string to (api_string, provider).

    If the string matches a registry alias, returns the registered api_string
    and provider. Otherwise returns the raw string as-is with provider=None
    (caller must supply provider separately).
    """
    if alias_or_raw in MODELS:
        info = MODELS[alias_or_raw]
        return info["api_string"], info["provider"]
    return alias_or_raw, None


def list_models() -> str:
    """Return a formatted string of available models grouped by provider."""
    by_provider: dict[str, list[tuple[str, ModelInfo]]] = {}
    for alias, info in MODELS.items():
        by_provider.setdefault(info["provider"], []).append((alias, info))

    lines: list[str] = ["Available models:", ""]
    for provider in sorted(by_provider):
        lines.append(f"  {provider}:")
        for alias, info in by_provider[provider]:
            desc = info.get("description", "")
            if alias in LEGACY_ALIASES:
                desc = f"(legacy) {desc}"
            lines.append(f"    {alias:<16} -> {info['api_string']:<35} [{info['tier']}]  {desc:<100}")
        lines.append("")

    return "\n".join(lines)
