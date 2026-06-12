"""
services/llm/factory.py

LLM provider factory with automatic fallback.

Priority order:
  1. Ollama + Qwen3 (primary — fully local, no API key required)
  2. Groq (fallback #1 — free tier, fast, requires GROQ_API_KEY)
  3. Gemini Flash (fallback #2 — free tier, requires GEMINI_API_KEY)

The factory checks availability in priority order and returns the
first available provider. Provider selection is logged at startup.

Usage:
    provider = get_provider()
    text, tokens = provider.complete(messages)
"""

from __future__ import annotations

import logging
from typing import Optional

from services.llm.base import LLMProvider, LLMProviderError
from services.llm.ollama_provider import OllamaProvider
from services.llm.groq_provider import GroqProvider
from services.llm.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

# Module-level singleton — selected once at first call
_selected_provider: Optional[LLMProvider] = None


def get_provider(force_refresh: bool = False) -> LLMProvider:
    """
    Return the first available LLM provider in priority order.

    Args:
        force_refresh: If True, re-check availability even if a provider
                       was previously selected. Useful after config changes.

    Returns:
        An LLMProvider instance that is confirmed available.

    Raises:
        RuntimeError if no provider is available.
    """
    global _selected_provider

    if _selected_provider is not None and not force_refresh:
        return _selected_provider

    candidates: list[LLMProvider] = [
        GroqProvider(),
        OllamaProvider(),
        GeminiProvider(),
    ]

    for provider in candidates:
        try:
            if provider.is_available():
                logger.info(
                    "LLM provider selected: %s",
                    provider.name,
                )
                _selected_provider = provider
                return provider
            else:
                logger.debug(
                    "LLM provider not available: %s",
                    provider.name,
                )
        except Exception as e:
            logger.debug(
                "LLM provider availability check failed for %s: %s",
                provider.name,
                e,
            )

    raise RuntimeError(
        "No LLM provider is available. Please ensure one of the following:\n"
        "  1. Ollama is running with qwen3 pulled: "
        "ollama pull qwen3 && ollama serve\n"
        "  2. GROQ_API_KEY is set in your .env file "
        "(free at console.groq.com)\n"
        "  3. GEMINI_API_KEY is set in your .env file "
        "(free at aistudio.google.com)\n"
    )


def get_provider_info() -> dict:
    """
    Return a summary of provider availability for logging/debugging.
    Does not select a provider.
    """
    candidates: list[LLMProvider] = [
        OllamaProvider(),
        GroqProvider(),
        GeminiProvider(),
    ]
    return {
        provider.name: provider.is_available()
        for provider in candidates
    }


def reset_provider() -> None:
    """Reset the cached provider selection. Useful in tests."""
    global _selected_provider
    _selected_provider = None
