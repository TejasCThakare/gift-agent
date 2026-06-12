"""
api/routes/run.py

POST /run

Starts the gift recommendation workflow for a contact.
Runs the graph until it hits the human_review interrupt.
Returns the thread_id and the review payload for the UI.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph
from storage.artifact_store import save_trace

logger = logging.getLogger("api.run")
router = APIRouter()


class RunRequest(BaseModel):
    contact: dict


class RunResponse(BaseModel):
    thread_id: str
    status: str
    review_payload: dict
    message: str


@router.post("/run", response_model=RunResponse)
async def run_workflow(request: RunRequest) -> RunResponse:
    """
    Start the gift recommendation workflow for a given contact.

    The workflow runs until it hits the human_review interrupt,
    then returns the review payload for the UI to display.

    Request body:
        contact: The contact JSON (see data/sample_input.json for schema)

    Returns:
        thread_id: Unique identifier for this run (use for /review and /status)
        status: "awaiting_review"
        review_payload: Full state for the reviewer UI
    """
    graph = get_graph()

    initial_state: dict[str, Any] = {
        "contact": request.contact,
    }

    # Generate a thread_id — will be finalized by the ingest node
    # We use a temporary one here; the graph will assign the real one
    import uuid
    thread_id = f"run_{uuid.uuid4().hex[:12]}"

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run graph until interrupt
        result = graph.invoke(initial_state, config=config)

        # After ingest, the actual thread_id may have been set
        actual_thread_id = result.get("thread_id", thread_id)

        # If the thread_id was updated by the ingest node, re-run with the correct one
        if actual_thread_id != thread_id:
            thread_id = actual_thread_id

        # Check if we're waiting for review
        # LangGraph sets __interrupt__ in the result when interrupt() is called
        review_payload = _extract_review_payload(result)

        # Persist trace
        try:
            save_trace(thread_id, result)
        except Exception as e:
            logger.warning("Failed to save trace: %s", e)

        return RunResponse(
            thread_id=thread_id,
            status="awaiting_review",
            review_payload=review_payload,
            message=(
                f"Workflow complete. Recommendations ready for review. "
                f"Use thread_id '{thread_id}' to submit your review."
            ),
        )

    except Exception as e:
        logger.error("Workflow failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Workflow error: {str(e)}",
        )


def _extract_review_payload(result: dict) -> dict:
    """Extract the review payload from the graph result."""
    # LangGraph interrupt() stores the interrupt value in __interrupt__
    # When the graph is paused at interrupt(), the result contains
    # the last state snapshot
    contact = result.get("contact", {})
    return {
        "thread_id": result.get("thread_id", ""),
        "contact_name": contact.get("name", ""),
        "role": contact.get("role", ""),
        "company": contact.get("company", ""),
        "evidence_map": result.get("evidence_map", {}),
        "safe_signals": result.get("safe_signals", {}),
        "signals_to_avoid": result.get("signals_to_avoid", []),
        "queries_used": [q.get("query", "") for q in result.get("queries", [])],
        "scored_products": result.get("scored_products", []),
        "ranked_gifts": result.get("ranked_gifts", []),
        "final_recommendations": result.get("final_recommendations", {}),
        "escalation_flag": result.get("escalation_flag", False),
        "escalation_notes": result.get("escalation_notes", ""),
        "retry_count": result.get("retry_count", 0),
        "review_history": result.get("review_history", []),
        "available_actions": ["approve", "reject", "edit", "regenerate"],
    }
