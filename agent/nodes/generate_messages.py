"""
agent/nodes/generate_messages.py

Node 8: generate_messages (LLM #3)

Generates a short personalised message for each ranked gift.
Messages are grounded in evidence_map — no new signals introduced.
Tone is derived from gift_context.occasion and relationship_type.

After this node, final_recommendations is built matching the
assignment output schema exactly.
"""

from __future__ import annotations

import json
import re

from agent.state import GiftAgentState
from models.recommendations import (
    FinalRecommendation, RankedGift, ProfileSignalsOutput,
    SearchTraceOutput, HumanReviewOutput,
)
from prompts.generate_messages import SYSTEM_PROMPT, build_user_prompt
from services.llm.factory import get_provider
from services.llm.base import LLMProviderError
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.generate_messages")

MAX_RETRIES = 1


def generate_messages(state: GiftAgentState) -> dict:
    """
    Generate personalised messages and assemble final_recommendations.
    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "generate_messages")

    ranked_gifts_raw = state.get("ranked_gifts", [])
    contact = state.get("contact", {})
    evidence_map = state.get("evidence_map", {})
    relationship_type = state.get("relationship_type", "unknown")
    gift_context = contact.get("gift_context", {})

    if not ranked_gifts_raw:
        logger.error("No ranked gifts to generate messages for")
        logs = log_node_end(logs, "generate_messages", error="No ranked gifts")
        return {"final_recommendations": {}, "logs": logs}

    # Determine message tone from relationship type and occasion
    tone = _determine_tone(relationship_type, gift_context.get("occasion", ""))

    provider = get_provider()

    user_prompt = build_user_prompt(
        ranked_gifts=ranked_gifts_raw,
        contact=contact,
        evidence_map=evidence_map,
        tone=tone,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    message_map: dict[int, str] = {}
    total_tokens = 0
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_response, tokens = provider.complete_json(
                messages=messages,
                temperature=0.4,
                max_tokens=1500,
            )
            total_tokens += tokens

            cleaned = _strip_markdown_json(raw_response)
            data = json.loads(cleaned)

            for msg_entry in data.get("messages", []):
                rank = msg_entry.get("rank")
                message_text = msg_entry.get("personalised_message", "")
                if rank and message_text:
                    message_map[int(rank)] = message_text

            if message_map:
                logger.info(
                    "Generated %d personalised messages (tone: %s)",
                    len(message_map),
                    tone,
                )
                break

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = f"Message parse error (attempt {attempt + 1}): {e}"
            logger.warning(last_error)

        except LLMProviderError as e:
            last_error = f"LLM error: {e}"
            logger.error(last_error)
            break

    # Fallback: generate simple messages if LLM failed
    if not message_map:
        logger.warning("Message generation failed — using fallback messages")
        for gift in ranked_gifts_raw:
            rank = gift.get("rank", 1)
            name = contact.get("name", "")
            gift_name = gift.get("gift_name", "this gift")
            message_map[rank] = (
                f"Dear {name}, we thought you might enjoy {gift_name}. "
                f"We hope it reflects our appreciation for your partnership."
            )
        last_error = f"LLM message generation failed — using fallback. {last_error}"

    # Attach messages to ranked gifts
    ranked_gifts_with_messages: list[RankedGift] = []
    for gift_dict in ranked_gifts_raw:
        rank = gift_dict.get("rank", 1)
        message = message_map.get(rank, "")
        gift = RankedGift.model_validate({**gift_dict, "personalised_message": message})
        ranked_gifts_with_messages.append(gift)

    # Build final_recommendations matching assignment output schema
    final_recommendations = _build_final_recommendations(
        state=state,
        ranked_gifts=ranked_gifts_with_messages,
    )

    logs = log_node_end(
        logs, "generate_messages",
        tokens_used=total_tokens,
        llm_calls=1,
        error=last_error if last_error else "",
    )

    return {
        "ranked_gifts": [g.model_dump() for g in ranked_gifts_with_messages],
        "final_recommendations": final_recommendations.to_output_schema(),
        "logs": logs,
    }


def _determine_tone(relationship_type: str, occasion: str) -> str:
    """Determine the appropriate message tone."""
    occasion_lower = occasion.lower() if occasion else ""

    # Casual occasions
    if any(word in occasion_lower for word in ["birthday", "celebration", "team", "fun"]):
        if relationship_type == "colleague":
            return "casual"

    # Professional occasions
    if any(word in occasion_lower for word in ["onboarding", "deal", "partnership", "business"]):
        return "professional"

    # Relationship-based defaults
    tone_by_relationship = {
        "existing_customer": "warm",
        "colleague": "warm",
        "prospective_customer": "professional",
        "executive": "professional",
        "founder": "warm",
        "partner": "warm",
        "unknown": "professional",
    }
    return tone_by_relationship.get(relationship_type, "warm")


def _build_final_recommendations(
    state: dict,
    ranked_gifts: list[RankedGift],
) -> FinalRecommendation:
    """Assemble the FinalRecommendation object from state."""
    contact = state.get("contact", {})
    safe_signals = state.get("safe_signals", {})
    signals_to_avoid = state.get("signals_to_avoid", [])
    queries = state.get("queries", [])
    raw_results = state.get("raw_results", [])
    validated_products = state.get("validated_products", [])
    retry_count = state.get("retry_count", 0)
    escalation_flag = state.get("escalation_flag", False)
    escalation_notes = state.get("escalation_notes", "")
    thread_id = state.get("thread_id", "")

    profile_signals = ProfileSignalsOutput(
        strong_signals=safe_signals.get("strong", []),
        weak_signals=safe_signals.get("weak", []),
        signals_to_avoid=signals_to_avoid,
    )

    usable_validated = [
        p for p in validated_products
        if p.get("validation_tier") in ("pass", "partial")
    ]

    search_trace = SearchTraceOutput(
        queries_used=[q.get("query", "") for q in queries],
        products_considered_count=len(raw_results),
        validated_count=len(usable_validated),
        escalation_triggered=escalation_flag,
        retry_count=retry_count,
    )

    return FinalRecommendation(
        contact_name=contact.get("name", ""),
        profile_signals=profile_signals,
        search_trace=search_trace,
        recommended_gifts=ranked_gifts,
        human_review=HumanReviewOutput(
            status="pending_review",
            available_actions=["approve", "reject", "edit", "regenerate"],
        ),
        escalation_flag=escalation_flag,
        escalation_notes=escalation_notes,
        thread_id=thread_id,
    )


def _strip_markdown_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()
