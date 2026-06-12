"""
services/llm/ollama_provider.py

Ollama LLM provider — the primary provider.
Uses Ollama's REST API directly via httpx. No SDK dependency.
Default model: qwen3 (pulled via `ollama pull qwen3`)

Ollama runs locally with no API key required. This is the default
and the only provider needed to run the project without any external accounts.

Setup:
  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
  2. Pull model: ollama pull qwen3
  3. Ollama starts automatically or: ollama serve
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

from services.llm.base import LLMProvider, LLMProviderError, LLMRateLimitError


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider using the /api/chat endpoint.
    Supports any model loaded in Ollama; defaults to qwen3.
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3")
        self._available: Optional[bool] = None  # cached after first check

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the target model is loaded.
        Caches result after first successful check.
        """
        if self._available is True:
            return True
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=3.0,
            )
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match by prefix: "qwen3" matches "qwen3:latest", "qwen3:4b", etc.
            model_base = self.model.split(":")[0]
            available = any(m.split(":")[0] == model_base for m in models)
            if available:
                self._available = True
            return available
        except Exception:
            return False

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> tuple[str, int]:
        """
        Send a chat completion request to Ollama.

        For JSON responses, adds explicit JSON instruction to the system
        message and sets format="json" in the request body.
        """
        # Prepare messages — add JSON instruction if needed
        prepared_messages = list(messages)
        if response_format == "json":
            prepared_messages = self._inject_json_instruction(prepared_messages)

        payload = {
            "model": self.model,
            "messages": prepared_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Request JSON format from Ollama if needed
        if response_format == "json":
            payload["format"] = "json"

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,  # Ollama can be slow on first load
            )
        except httpx.TimeoutException as e:
            raise LLMProviderError(f"Ollama request timed out: {e}") from e
        except httpx.ConnectError as e:
            self._available = False
            raise LLMProviderError(f"Cannot connect to Ollama at {self.base_url}: {e}") from e

        if response.status_code != 200:
            raise LLMProviderError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Ollama returned invalid JSON: {e}") from e

        content = data.get("message", {}).get("content", "")
        if not content:
            raise LLMProviderError("Ollama returned empty content")

        # Estimate token usage from eval_count if available
        tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return content, tokens_used

    def _inject_json_instruction(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Ensure the system message includes a JSON instruction.
        If a system message exists, appends to it.
        If not, prepends a new system message.
        """
        result = list(messages)
        json_note = "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no backticks, no preamble."

        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                result[i] = {**msg, "content": msg["content"] + json_note}
                return result

        # No system message — prepend one
        result.insert(0, {"role": "system", "content": json_note.strip()})
        return result
