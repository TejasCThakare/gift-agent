"""
agent/nodes/ingest.py

Node 1: ingest

Responsibilities:
  - Parse and validate the contact JSON against Pydantic models
  - Normalize budget (ensure min <= max, handle missing values)
  - Extract relationship_type to state top-level for score_products
  - Initialize retry_count, escalation_flag, review_history
  - Generate thread_id
  - Compute flattened profile_text for LLM consumption
  - Log node execution

Fails loudly if the contact is missing required fields (name).
All other fields have sensible defaults.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError

from agent.state import GiftAgentState
from models.contact import ContactInput
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.ingest")


def ingest(state: GiftAgentState) -> dict:
    """
    Ingest and validate the contact. Initialize all state fields.

    Returns partial state update dict for LangGraph.
    """
    logs = log_node_start(state.get("logs", {}), "ingest")

    contact_raw = state.get("contact", {})
    if not contact_raw:
        logs = log_node_end(logs, "ingest", error="No contact data provided")
        raise ValueError("No contact data provided in state")

    # Validate against Pydantic model
    try:
        contact_model = ContactInput.model_validate(contact_raw)
    except ValidationError as e:
        error_msg = f"Contact validation failed: {e}"
        logger.error(error_msg)
        logs = log_node_end(logs, "ingest", error=error_msg)
        raise ValueError(error_msg) from e

    # Normalize: ensure budget_min <= budget_max
    gc = contact_model.gift_context
    if gc.budget_min > gc.budget_max:
        logger.warning(
            "budget_min (%s) > budget_max (%s) — swapping",
            gc.budget_min,
            gc.budget_max,
        )
        gc.budget_min, gc.budget_max = gc.budget_max, gc.budget_min

    # Generate thread_id (stable per contact if not already set)
    existing_thread_id = state.get("thread_id", "")
    thread_id = existing_thread_id or f"run_{uuid.uuid4().hex[:12]}"

    # Build flattened profile text for LLM
    profile_text = contact_model.to_profile_text()

    # Extract relationship type to top-level state
    relationship_type = contact_model.relationship_context.relationship_type

    # Convert back to dict for state (use model_dump for clean serialization)
    contact_dict = contact_model.model_dump()

    logger.info(
        "Ingested contact: %s | relationship: %s | budget: %s %s–%s | thread: %s",
        contact_model.name,
        relationship_type,
        gc.currency,
        gc.budget_min,
        gc.budget_max,
        thread_id,
    )

    logs = log_node_end(logs, "ingest")

    return {
        "contact": contact_dict,
        "profile_text": profile_text,
        "relationship_type": relationship_type,
        "thread_id": thread_id,
        # Initialize all state fields that may not exist yet
        "evidence_map": state.get("evidence_map", {}),
        "strong_signals": state.get("strong_signals", []),
        "weak_signals": state.get("weak_signals", []),
        "queries": state.get("queries", []),
        "safe_signals": state.get("safe_signals", {}),
        "signals_to_avoid": state.get("signals_to_avoid", []),
        "raw_results": state.get("raw_results", []),
        "validated_products": state.get("validated_products", []),
        "scored_products": state.get("scored_products", []),
        "ranked_gifts": state.get("ranked_gifts", []),
        "final_recommendations": state.get("final_recommendations", {}),
        "retry_count": state.get("retry_count", 0),
        "widened_budget_max": state.get("widened_budget_max", None),
        "escalation_flag": state.get("escalation_flag", False),
        "escalation_notes": state.get("escalation_notes", ""),
        "review_history": state.get("review_history", []),
        "review_action": state.get("review_action", ""),
        "review_notes": state.get("review_notes", ""),
        "edit_payload": state.get("edit_payload", None),
        "logs": logs,
    }
