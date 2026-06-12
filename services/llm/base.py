"""
services/llm/base.py

Abstract base class for all LLM providers.

Every provider must implement:
  is_available()  — check if the provider can be used right now
  complete()      — send messages and return the response text

This interface ensures that the workflow nodes are completely
decoupled from the specific LLM implementation. The factory
selects the provider; nodes only see this interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether this provider is currently available.
        Should be fast (no actual LLM call) — just check if the
        required config (API key, running service) is present.
        """
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,  # "json" to request JSON output
    ) -> tuple[str, int]:
        """
        Send a completion request and return (response_text, tokens_used).

        Args:
            messages: List of {role: str, content: str} dicts.
                      Roles: "system", "user", "assistant"
            temperature: Sampling temperature. Lower = more deterministic.
            max_tokens: Maximum tokens in the response.
            response_format: If "json", instructs the model to return JSON.

        Returns:
            (response_text, tokens_used) tuple.
            tokens_used is approximate if the provider doesn't report it.

        Raises:
            LLMProviderError on non-recoverable failures.
            LLMRateLimitError on rate limit errors (for retry logic).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        """
        Convenience wrapper for JSON responses.
        Uses lower temperature by default for more consistent JSON output.
        """
        return self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
        )


class LLMProviderError(Exception):
    """Non-recoverable LLM provider error."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Rate limit error — caller may retry with backoff."""
    pass
