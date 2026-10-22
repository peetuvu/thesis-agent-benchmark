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
}


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
            lines.append(f"    {alias:<16} -> {info['api_string']:<35} [{info['tier']}]  {desc:<100}")
        lines.append("")

    return "\n".join(lines)
