"""
api/routes/review.py

POST /review/{thread_id}

Submit a human review action to resume the workflow.

Actions:
  approve    — accept recommendations as-is → workflow proceeds to done
  reject     — reject with reason → workflow re-ranks with history
  edit       — apply inline edit to a specific gift field → done
  regenerate — provide notes for re-generation → re-scores + re-ranks
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph

logger = logging.getLogger("api.review")
router = APIRouter()


class EditPayload(BaseModel):
    gift_index: int = 0        # 0-based index into ranked_gifts
    field: str = ""            # field to edit (e.g. "personalised_message")
    new_value: str = ""        # new value for the field


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject", "edit", "regenerate"]
    reason: str = ""           # required for reject
    notes: str = ""            # optional notes for any action
    edit_payload: Optional[EditPayload] = None  # required for edit


class ReviewResponse(BaseModel):
    thread_id: str
    status: str
    message: str
    final_recommendations: dict = {}


@router.post("/review/{thread_id}", response_model=ReviewResponse)
async def submit_review(thread_id: str, request: ReviewRequest) -> ReviewResponse:
    """
    Submit a review action to resume the paused workflow.

    The workflow was paused at human_review waiting for this action.
    After receiving it, the graph resumes and runs to completion
    (or to the next interrupt if another review cycle is triggered).

    Path param:
        thread_id: From the /run response

    Request body:
        action: approve | reject | edit | regenerate
        reason: Rejection or regeneration reason (required for reject)
        notes: Additional notes for the reviewer
        edit_payload: For edit action — which gift field to change
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Validate action-specific requirements
    if request.action == "reject" and not request.reason.strip():
        raise HTTPException(
            status_code=400,
            detail="Rejection requires a reason. Please provide 'reason' in request body.",
        )

    if request.action == "edit" and not request.edit_payload:
        raise HTTPException(
            status_code=400,
            detail="Edit action requires 'edit_payload' with gift_index, field, and new_value.",
        )

    # Build state update with review action
    state_update: dict[str, Any] = {
        "review_action": request.action,
        "review_notes": request.notes or request.reason,
    }

    if request.edit_payload:
        state_update["edit_payload"] = request.edit_payload.model_dump()

    try:
        # Update state with the review action
        graph.update_state(config, state_update)

        # Resume graph execution
        result = graph.invoke(None, config=config)

        final_recommendations = result.get("final_recommendations", {})
        status = final_recommendations.get("human_review", {}).get("status", "unknown")

        # Determine response message
        messages = {
            "approve": "Recommendations approved. Final output saved.",
            "edit": "Edits applied and saved.",
            "reject": f"Recommendations rejected. Re-ranking with your feedback: '{request.reason}'",
            "regenerate": "Re-generating recommendations with your notes.",
        }

        return ReviewResponse(
            thread_id=thread_id,
            status=status,
            message=messages.get(request.action, "Action processed."),
            final_recommendations=final_recommendations,
        )

    except Exception as e:
        logger.error("Review failed for thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Review processing error: {str(e)}",
        )
