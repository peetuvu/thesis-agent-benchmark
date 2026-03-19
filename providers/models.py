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
    # OpenAI
    "gpt-5.4": {"api_string": "gpt-5.4", "provider": "openai", "tier": "Flagship (plus-subscription)", "description": "Best intelligence at scale for agentic, coding, and professional workflows"} ,
    "GPT-5.4 pro": {"api_string": "GPT-5.4-pro", "provider": "openai", "tier": "Flagship (pro-subscription)", "description": "Version of GPT-5.4 that produces smarter and more precise responses."},
    "gpt-4o": {"api_string": "gpt-4o", "provider": "openai", "tier": "mid", "description": "Fast, intelligent, flexible GPT model"},
    "gpt-4.1": {"api_string": "gpt-4.1", "provider": "openai", "tier": "mid", "description": "Smartest non-reasoning model"},
    "gpt-4.1-mini": {"api_string": "gpt-4.1-mini", "provider": "openai", "tier": "fast", "description": "Smaller, faster version of GPT-4.1"},
    "o3": {"api_string": "o3", "provider": "openai", "tier": "good", "description": "Reasoning model for complex tasks, succeeded by GPT-5"},
    "o3-pro": {"api_string": "o3-pro", "provider": "openai", "tier": "good", "description": "Version of o3 with more compute for better responses"},
    "o4-mini": {"api_string": "o4-mini", "provider": "openai", "tier": "fast", "description": "Fast, cost-efficient reasoning model, succeeded by GPT-5 mini"},
}


def get_model(alias: str) -> ModelInfo:
    """Look up a model by alias. Raises KeyError if not found."""
    if alias not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise KeyError(f"Unknown model alias '{alias}'. Available: {available}")
    return MODELS[alias]


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
