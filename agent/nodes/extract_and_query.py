"""
agent/nodes/extract_and_query.py

Node 2: extract_and_query (LLM #1)

Responsibilities:
  - Single LLM call that produces evidence_map, signals, and search queries
  - Validates LLM output against ExtractedSignals Pydantic model
  - Enforces JSON output — retries on parse failure (max 2 retries)
  - Logs token usage and latency

The LLM is instructed to ground every signal in exact profile quotes.
The evidence_map is the audit trail for all downstream reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from agent.state import GiftAgentState
from models.signals import ExtractedSignals, SearchQuery
from prompts.extract_and_query import SYSTEM_PROMPT, build_user_prompt
from services.llm.factory import get_provider
from services.llm.base import LLMProviderError
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.extract_and_query")

MAX_RETRIES = 2


def extract_and_query(state: GiftAgentState) -> dict:
    """
    Run LLM #1 to extract evidence-grounded signals and search queries.

    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "extract_and_query")

    profile_text = state.get("profile_text", "")
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})

    if not profile_text:
        error_msg = "No profile_text in state — ingest node must run first"
        logs = log_node_end(logs, "extract_and_query", error=error_msg)
        raise ValueError(error_msg)

    provider = get_provider()
    user_prompt = build_user_prompt(profile_text, gift_context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    extracted = None
    total_tokens = 0
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_response, tokens = provider.complete_json(
                messages=messages,
                temperature=0.2,
                max_tokens=3000,
            )
            total_tokens += tokens

            # Strip markdown fences if present (some models add them)
            cleaned = _strip_markdown_json(raw_response)

            # Parse JSON
            data = json.loads(cleaned)

            # Validate against Pydantic model
            extracted = ExtractedSignals.model_validate(data)
            logger.info(
                "Extracted %d strong, %d weak signals, %d queries (attempt %d)",
                len(extracted.strong_signals),
                len(extracted.weak_signals),
                len(extracted.queries),
                attempt + 1,
            )
            break

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error (attempt {attempt + 1}): {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                # Add correction message
                messages.append({
                    "role": "assistant",
                    "content": raw_response if "raw_response" in dir() else "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. "
                        "Please respond with valid JSON only, "
                        "matching the schema exactly. No markdown, no backticks."
                    ),
                })

        except (ValidationError, KeyError, TypeError) as e:
            last_error = f"Schema validation error (attempt {attempt + 1}): {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your JSON did not match the required schema. Error: {e}. "
                        "Please try again with the exact schema structure."
                    ),
                })

        except LLMProviderError as e:
            last_error = f"LLM error: {e}"
            logger.error(last_error)
            break

    if extracted is None:
        # Use safe empty defaults rather than crashing the whole workflow
        logger.error(
            "extract_and_query failed after %d attempts: %s. Using empty signals.",
            MAX_RETRIES + 1,
            last_error,
        )
        extracted = ExtractedSignals(
            evidence_map={},
            strong_signals=["professional gift for " + contact.get("role", "executive")],
            weak_signals=["may appreciate premium professional gift"],
            queries=[
                SearchQuery(
                    query=(
                        f"premium professional gift India "
                        f"under {gift_context.get('budget_max', 5000)} rupees"
                    ),
                    type="fallback",
                    signal_used="professional role",
                )
            ],
        )
        last_error = f"Signal extraction failed — using fallback signals. Last error: {last_error}"

    logs = log_node_end(
        logs, "extract_and_query",
        tokens_used=total_tokens,
        llm_calls=min(MAX_RETRIES + 1, 1),
        error=last_error if extracted.evidence_map == {} and last_error else "",
    )

    return {
        "evidence_map": extracted.evidence_map,
        "strong_signals": extracted.strong_signals,
        "weak_signals": extracted.weak_signals,
        "queries": [q.model_dump() for q in extracted.queries],
        "logs": logs,
    }


def _strip_markdown_json(text: str) -> str:
    """Remove markdown code fences and extract JSON content."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()
