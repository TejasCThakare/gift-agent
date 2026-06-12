"""
services/llm/gemini_provider.py

Gemini LLM provider — fallback #2.
Uses the official google-generativeai Python SDK.
Default model: gemini-1.5-flash (free tier via Google AI Studio)

Get a free API key at: https://aistudio.google.com
Add GEMINI_API_KEY to your .env file.

This provider is only used when both Ollama and Groq are unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from services.llm.base import LLMProvider, LLMProviderError, LLMRateLimitError


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider using the google-generativeai SDK.
    Only activated when GEMINI_API_KEY is available.
    """

    MODEL = "gemini-1.5-flash"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self._client = None

    @property
    def name(self) -> str:
        return f"Gemini ({self.MODEL})"

    def is_available(self) -> bool:
        """Available if GEMINI_API_KEY environment variable is set and non-empty."""
        return bool(self.api_key.strip())

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.MODEL)
            except ImportError as e:
                raise LLMProviderError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                ) from e
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> tuple[str, int]:
        """
        Send a completion request to Gemini.
        Converts OpenAI-style messages to Gemini's format.
        """
        client = self._get_client()
        import google.generativeai as genai

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles, not "user" and "assistant"
        # System messages are handled differently (as system_instruction)
        system_parts = []
        gemini_history = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                gemini_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})

        # Re-initialize client with system instruction if present
        if system_parts:
            import google.generativeai as genai
            system_text = "\n\n".join(system_parts)
            if response_format == "json":
                system_text += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no backticks, no preamble."
            client = genai.GenerativeModel(
                self.MODEL,
                system_instruction=system_text,
            )

        # Build the generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"

        # Extract the last user message as the prompt
        # and any prior messages as history
        if not gemini_history:
            raise LLMProviderError("Gemini: no user messages found in message list")

        last_user_msg = None
        chat_history = []
        for msg in gemini_history:
            if msg["role"] == "user":
                last_user_msg = msg
                chat_history.append(msg)
            else:
                chat_history.append(msg)

        # Pop the last user message from history — it becomes the prompt
        if last_user_msg in chat_history:
            chat_history = [m for m in chat_history if m is not last_user_msg]

        try:
            chat = client.start_chat(history=chat_history)
            response = chat.send_message(
                last_user_msg["parts"][0],
                generation_config=generation_config,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "rate" in error_str:
                raise LLMRateLimitError(f"Gemini rate limited: {e}") from e
            elif "api_key" in error_str or "401" in error_str:
                raise LLMProviderError(
                    "Gemini authentication failed. Check GEMINI_API_KEY."
                ) from e
            else:
                raise LLMProviderError(f"Gemini API error: {e}") from e

        content = response.text or ""

        # Estimate token usage from usage_metadata if available
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = (
                getattr(response.usage_metadata, "prompt_token_count", 0)
                + getattr(response.usage_metadata, "candidates_token_count", 0)
            )

        return content, tokens_used
