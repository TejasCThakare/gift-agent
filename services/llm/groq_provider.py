"""
services/llm/groq_provider.py

Groq LLM provider — fallback #1.
Uses the official groq Python SDK.
Default model: llama-3.3-70b-versatile (current as of 2025)
Fallback model: llama-3.1-8b-instant
"""

from __future__ import annotations

import os
from typing import Optional

from services.llm.base import LLMProvider, LLMProviderError, LLMRateLimitError


class GroqProvider(LLMProvider):

    PRIMARY_MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = self.PRIMARY_MODEL
        self._client = None

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key.strip())

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError as e:
                raise LLMProviderError("groq package not installed. Run: pip install groq") from e
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> tuple[str, int]:
        client = self._get_client()

        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()

            if "rate_limit" in error_str or "429" in error_str:
                if self.model == self.PRIMARY_MODEL:
                    self.model = self.FALLBACK_MODEL
                    try:
                        kwargs["model"] = self.model
                        response = client.chat.completions.create(**kwargs)
                    except Exception as e2:
                        raise LLMRateLimitError(f"Groq rate limited on both models: {e2}") from e2
                else:
                    raise LLMRateLimitError(f"Groq rate limited: {e}") from e

            elif "decommissioned" in error_str or "model_decommissioned" in error_str:
                # Try fallback model
                self.model = self.FALLBACK_MODEL
                try:
                    kwargs["model"] = self.model
                    response = client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMProviderError(f"Groq model error: {e2}") from e2

            elif "authentication" in error_str or "401" in error_str:
                raise LLMProviderError("Groq authentication failed. Check GROQ_API_KEY.") from e
            else:
                raise LLMProviderError(f"Groq API error: {e}") from e

        content = response.choices[0].message.content or ""
        tokens_used = 0
        if response.usage:
            tokens_used = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

        return content, tokens_used