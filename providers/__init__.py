"""LLM provider abstraction layer.

Providers are loaded dynamically by name so that architectures never
import SDK-specific code directly.
"""

from __future__ import annotations

from providers.base import LLMProvider, LLMResponse, ProviderError

_PROVIDER_REGISTRY: dict[str, str] = {
    "anthropic": "providers.anthropic.AnthropicProvider",
    "openai": "providers.openai.OpenAIProvider",
}


def get_provider(provider_name: str) -> LLMProvider:
    """Instantiate and return an LLMProvider by name.

    Args:
        provider_name: One of the registered provider names (e.g. 'anthropic', 'openai').

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
        ProviderError: If the SDK is not installed or API key is missing.
    """
    if provider_name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider '{provider_name}'. Available providers: {available}"
        )

    qualified_name = _PROVIDER_REGISTRY[provider_name]
    module_path, class_name = qualified_name.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    return provider_class()


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    "get_provider",
]
