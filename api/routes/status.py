"""
api/routes/status.py

GET /status/{thread_id}

Returns the current state of a workflow run.
Used by the UI to poll for results when the graph is running.

GET /runs

Returns a list of all past runs with summary info.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph
from storage.artifact_store import load_recommendations, load_trace, list_runs

logger = logging.getLogger("api.status")
router = APIRouter()


class RunSummary(BaseModel):
    thread_id: str
    contact_name: str
    status: str
    created_at: str


@router.get("/status/{thread_id}")
async def get_status(thread_id: str) -> dict:
    """
    Get the current state of a workflow run.

    Returns the latest state snapshot from LangGraph's MemorySaver,
    or falls back to persisted artifacts if the graph session expired.

    Path param:
        thread_id: From the /run response

    Returns:
        Full state dict including ranked_gifts, evidence_map,
        validation reports, escalation status, and review history.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Try to get state from LangGraph's checkpointer
        state = graph.get_state(config)
        if state and state.values:
            return {
                "thread_id": thread_id,
                "status": _determine_status(state.values),
                "state": state.values,
                "next_nodes": list(state.next),
            }
    except Exception as e:
        logger.debug("Could not get live state for %s: %s", thread_id, e)

    # Fallback: read from persisted artifacts
    recommendations = load_recommendations(thread_id)
    if recommendations:
        return {
            "thread_id": thread_id,
            "status": recommendations.get("human_review", {}).get("status", "unknown"),
            "final_recommendations": recommendations,
        }

    trace = load_trace(thread_id)
    if trace:
        return {
            "thread_id": thread_id,
            "status": "in_progress",
            "trace": trace,
        }

    raise HTTPException(
        status_code=404,
        detail=f"No workflow found for thread_id '{thread_id}'",
    )


@router.get("/runs")
async def list_all_runs() -> dict:
    """
    List all past workflow runs.

    Returns:
        List of run summaries with thread_id, contact_name, status, created_at.
    """
    runs = list_runs()
    return {
        "runs": runs,
        "total": len(runs),
    }


def _determine_status(state_values: dict) -> str:
    """Determine current workflow status from state."""
    if state_values.get("final_recommendations"):
        review_status = (
            state_values["final_recommendations"]
            .get("human_review", {})
            .get("status", "")
        )
        if review_status:
            return review_status
        return "awaiting_review"

    if state_values.get("ranked_gifts"):
        return "awaiting_review"

    if state_values.get("scored_products"):
        return "ranking"

    if state_values.get("validated_products"):
        return "scoring"

    if state_values.get("raw_results"):
        return "validating"

    return "processing"
