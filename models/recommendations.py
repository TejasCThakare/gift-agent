"""
models/recommendations.py

Pydantic models for ranked gifts, final recommendations, and review history.

RankedGift maps to the assignment output schema for a single gift.
FinalRecommendation maps to the full assignment output schema.
ReviewEntry records each human review action for auditability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class RankedGift(BaseModel):
    """
    A single ranked gift recommendation.
    Matches the assignment output schema exactly.

    Important: confidence and risk_level are COPIED from ScoredProduct.
    They are never generated or modified by the ranking LLM.
    evidence_citations must reference exact quotes from evidence_map.
    """
    rank: int
    gift_name: str
    product_url: str        # Copied from ScoredProduct.url — never LLM-generated
    store: str
    estimated_price: str    # Human-readable, e.g. "₹3,999"
    why_this_gift: str      # LLM-generated reasoning, grounded in signals
    personalisation_reasoning: str  # Specific signals used, must cite evidence
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Exact quotes from evidence_map that support this recommendation."
    )
    personalised_message: str = ""  # Added by generate_messages node
    confidence: float       # Copied from ScoredProduct.confidence — never LLM-generated
    risk_level: str         # Copied from ScoredProduct.risk_level
    assumptions: list[str] = Field(default_factory=list)

    def to_output_schema(self) -> dict:
        """Return dict matching the assignment output schema."""
        return {
            "rank": self.rank,
            "gift_name": self.gift_name,
            "product_url": self.product_url,
            "store": self.store,
            "estimated_price": self.estimated_price,
            "why_this_gift": self.why_this_gift,
            "personalisation_reasoning": self.personalisation_reasoning,
            "evidence_citations": self.evidence_citations,
            "personalised_message": self.personalised_message,
            "confidence_score": self.confidence,
            "risk_level": self.risk_level,
            "assumptions": self.assumptions,
        }


class ProfileSignalsOutput(BaseModel):
    """Maps to the profile_signals section of the assignment output schema."""
    strong_signals: list[str] = Field(default_factory=list)
    weak_signals: list[str] = Field(default_factory=list)
    signals_to_avoid: list[Any] = Field(default_factory=list)


class SearchTraceOutput(BaseModel):
    """Maps to the search_trace section of the assignment output schema."""
    queries_used: list[str] = Field(default_factory=list)
    products_considered_count: int = 0
    validated_count: int = 0
    escalation_triggered: bool = False
    retry_count: int = 0


class HumanReviewOutput(BaseModel):
    """Maps to the human_review section of the assignment output schema."""
    status: str = "pending_review"  # pending_review | approved | rejected | edited
    available_actions: list[str] = Field(
        default_factory=lambda: ["approve", "reject", "edit", "regenerate"]
    )


class FinalRecommendation(BaseModel):
    """
    Complete output for a single contact.
    Matches the assignment output schema exactly.
    """
    contact_name: str
    profile_signals: ProfileSignalsOutput
    search_trace: SearchTraceOutput
    recommended_gifts: list[RankedGift] = Field(default_factory=list)
    human_review: HumanReviewOutput = Field(default_factory=HumanReviewOutput)

    # Additional fields for auditability (beyond assignment schema)
    escalation_flag: bool = False
    escalation_notes: str = ""
    thread_id: str = ""
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_output_schema(self) -> dict:
        """Return dict matching the assignment output schema."""
        return {
            "contact_name": self.contact_name,
            "profile_signals": self.profile_signals.model_dump(),
            "search_trace": self.search_trace.model_dump(),
            "recommended_gifts": [g.to_output_schema() for g in self.recommended_gifts],
            "human_review": self.human_review.model_dump(),
            "escalation_flag": self.escalation_flag,
            "escalation_notes": self.escalation_notes,
            "thread_id": self.thread_id,
            "generated_at": self.generated_at,
        }


class ReviewEntry(BaseModel):
    """
    A single entry in the review_history list.
    Persisted across regenerations so the LLM can read rejection reasons
    and avoid previously flagged products.
    """
    action: str             # approve | reject | edit | regenerate
    reason: str = ""
    notes: str = ""
    gifts_flagged: list[int] = Field(
        default_factory=list,
        description="Ranks of gifts the reviewer flagged (1-indexed)."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_llm_context(self) -> str:
        """Format this review entry for inclusion in LLM prompts."""
        parts = [f"[{self.action.upper()} at {self.timestamp}]"]
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        if self.gifts_flagged:
            parts.append(f"Flagged gift ranks: {self.gifts_flagged}")
        return "\n".join(parts)
