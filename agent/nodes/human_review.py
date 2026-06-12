"""
agent/nodes/human_review.py

Node 9: human_review

Uses LangGraph's interrupt() to pause execution and wait for human input.

When interrupted, the current state is available via GET /status/{thread_id}.
The reviewer takes an action via POST /review/{thread_id}.
The graph resumes with the action stored in state.

Actions and routing:
  approve    → done (write final output)
  reject     → store rejection in review_history → rank_gifts (re-rank)
  edit       → apply inline edits to state → done
  regenerate → store notes in review_history → score_products (re-score + re-rank)

Design:
  - Rejection routes to rank_gifts (NOT ingest) — profile unchanged
  - Regeneration routes to score_products — re-scores + re-ranks
  - review_history persists across all review cycles
  - Each cycle adds one ReviewEntry to review_history
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import interrupt

from agent.state import GiftAgentState
from models.recommendations import ReviewEntry
from storage.artifact_store import save_review_history, save_trace, save_logs
from utils.logging import get_logger, log_node_start, log_node_end, summarize_logs

logger = get_logger("nodes.human_review")


def human_review(state: GiftAgentState) -> dict:
    """
    Pause graph execution for human review.

    On first entry: calls interrupt() to pause and surface the state.
    On resume: reads review_action and routes accordingly.

    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "human_review")

    final_recommendations = state.get("final_recommendations", {})
    ranked_gifts = state.get("ranked_gifts", [])
    thread_id = state.get("thread_id", "")

    # ── INTERRUPT: pause and surface state for reviewer ────────────────────
    # interrupt() sends the review payload to the waiting client
    # and suspends execution until graph.invoke() is called again
    # with the updated state containing review_action.
    review_payload = _build_review_payload(state)

    # This call pauses execution. Execution resumes when:
    # 1. graph.update_state(thread_id, {review_action, review_notes}) is called
    # 2. graph.invoke(None, config) is called to resume
    interrupt(value=review_payload)

    # ── RESUME: read the action the reviewer took ──────────────────────────
    review_action = state.get("review_action", "approve")
    review_notes = state.get("review_notes", "")
    edit_payload = state.get("edit_payload", None)

    logger.info("Review action: %s | thread: %s", review_action, thread_id)

    # Build a ReviewEntry for this review cycle
    gifts_flagged = []
    if review_action in ("reject", "regenerate") and ranked_gifts:
        # Flag all current gifts unless specific ones are noted
        gifts_flagged = [g.get("rank", i + 1) for i, g in enumerate(ranked_gifts)]

    entry = ReviewEntry(
        action=review_action,
        reason=review_notes if review_action != "edit" else "",
        notes=review_notes,
        gifts_flagged=gifts_flagged,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    review_history = list(state.get("review_history", []))
    review_history.append(entry.model_dump())

    # Persist review history to disk
    if thread_id:
        save_review_history(thread_id, review_history)
        save_trace(thread_id, state)
        log_summary = summarize_logs(state.get("logs", {}))
        save_logs(thread_id, log_summary)

    logs = log_node_end(logs, "human_review")

    # ── ROUTING: determine where to go next ────────────────────────────────
    # The graph's conditional edge reads review_action from state
    # to determine the next node. We set it here.

    if review_action == "edit" and edit_payload:
        # Apply inline edits to the ranked gifts
        updated_gifts = _apply_edit(
            ranked_gifts=ranked_gifts,
            edit_payload=edit_payload,
        )
        # Also update the final_recommendations with edited gifts
        updated_recommendations = dict(final_recommendations)
        updated_recommendations["recommended_gifts"] = [
            _gift_to_output(g) for g in updated_gifts
        ]
        updated_recommendations["human_review"] = {
            "status": "edited",
            "available_actions": [],
        }
        return {
            "review_action": review_action,
            "review_history": review_history,
            "ranked_gifts": updated_gifts,
            "final_recommendations": updated_recommendations,
            "logs": logs,
        }

    # For approve, reject, regenerate — just update state and let
    # the conditional edge route to the right node
    return {
        "review_action": review_action,
        "review_notes": review_notes,
        "review_history": review_history,
        "logs": logs,
    }


def _build_review_payload(state: GiftAgentState) -> dict:
    """
    Build the payload shown to the reviewer.
    This is what the client sees when the graph is interrupted.
    """
    contact = state.get("contact", {})
    return {
        "thread_id": state.get("thread_id", ""),
        "contact_name": contact.get("name", ""),
        "role": contact.get("role", ""),
        "company": contact.get("company", ""),
        # Evidence and signals
        "evidence_map": state.get("evidence_map", {}),
        "safe_signals": state.get("safe_signals", {}),
        "signals_to_avoid": state.get("signals_to_avoid", []),
        # Search and validation trace
        "queries_used": [q.get("query", "") for q in state.get("queries", [])],
        "validated_products_count": len(state.get("validated_products", [])),
        "scored_products": state.get("scored_products", []),
        # Recommendations
        "ranked_gifts": state.get("ranked_gifts", []),
        "final_recommendations": state.get("final_recommendations", {}),
        # Status flags
        "escalation_flag": state.get("escalation_flag", False),
        "escalation_notes": state.get("escalation_notes", ""),
        "retry_count": state.get("retry_count", 0),
        # Review history (for context on regeneration)
        "review_history": state.get("review_history", []),
        # Available actions
        "available_actions": ["approve", "reject", "edit", "regenerate"],
    }


def _apply_edit(
    ranked_gifts: list[dict],
    edit_payload: dict,
) -> list[dict]:
    """
    Apply inline edits to a ranked gift.

    edit_payload format:
    {
        "gift_index": 0,        # 0-based index into ranked_gifts
        "field": "gift_name",   # field to edit
        "new_value": "..."      # new value
    }
    """
    gifts = list(ranked_gifts)
    gift_index = edit_payload.get("gift_index", 0)
    field = edit_payload.get("field", "")
    new_value = edit_payload.get("new_value", "")

    # Fields the reviewer is allowed to edit
    editable_fields = {
        "gift_name", "why_this_gift", "personalisation_reasoning",
        "personalised_message", "assumptions", "estimated_price",
    }

    if 0 <= gift_index < len(gifts) and field in editable_fields:
        gift = dict(gifts[gift_index])
        gift[field] = new_value
        gifts[gift_index] = gift
        logger.info(
            "Applied edit: gift[%d].%s = %s",
            gift_index, field, str(new_value)[:50],
        )
    else:
        logger.warning(
            "Edit rejected: index=%d, field=%s not editable",
            gift_index, field,
        )

    return gifts


def _gift_to_output(gift: dict) -> dict:
    """Convert a ranked gift dict to the output schema format."""
    return {
        "rank": gift.get("rank"),
        "gift_name": gift.get("gift_name", ""),
        "product_url": gift.get("product_url", ""),
        "store": gift.get("store", ""),
        "estimated_price": gift.get("estimated_price", ""),
        "why_this_gift": gift.get("why_this_gift", ""),
        "personalisation_reasoning": gift.get("personalisation_reasoning", ""),
        "evidence_citations": gift.get("evidence_citations", []),
        "personalised_message": gift.get("personalised_message", ""),
        "confidence_score": gift.get("confidence", 0.0),
        "risk_level": gift.get("risk_level", "high"),
        "assumptions": gift.get("assumptions", []),
    }
