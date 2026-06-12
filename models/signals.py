"""
models/signals.py

Pydantic models for signal extraction, filtering, and evidence tracking.

EvidenceMap ensures every signal is traceable to exact profile quotes.
FilteredSignal records why a signal was dropped, for auditability.
SafeSignals is the cleaned signal set passed to search and ranking.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A single search query with its metadata."""
    query: str
    type: str = "strong"  # strong | weak | fallback
    signal_used: str = ""


class ExtractedSignals(BaseModel):
    """
    Output of the extract_and_query LLM call.
    The evidence_map enforces grounding: every signal must have
    at least one supporting quote from the actual profile text.
    """
    evidence_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Maps each signal to a list of exact quotes from the profile "
            "that support it. Keys are signal strings. Values are lists of "
            "verbatim text extracted from posts, comments, headline, about, "
            "experience descriptions, or engaged_topics."
        )
    )
    strong_signals: list[str] = Field(
        default_factory=list,
        description=(
            "High-confidence signals directly and explicitly stated in the profile. "
            "E.g. if someone posts about cricket, 'interest in cricket' is strong."
        )
    )
    weak_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Inferred signals with lower certainty. "
            "E.g. 'may appreciate leadership books' inferred from role title."
        )
    )
    queries: list[SearchQuery] = Field(
        default_factory=list,
        description=(
            "Search queries to find relevant gift products. "
            "Each query targets a specific signal and includes budget + country."
        )
    )


class FilteredSignal(BaseModel):
    """Records a signal that was removed by the safety filter."""
    signal: str
    reason: str
    category: str  # religion | health | politics | gender | family | ethnicity


class SafeSignals(BaseModel):
    """
    Output of the filter_signals node.
    Contains only signals that passed the safety blocklist.
    Borderline signals are kept but flagged for reviewer awareness.
    """
    strong: list[str] = Field(default_factory=list)
    weak: list[str] = Field(default_factory=list)
    borderline: list[str] = Field(
        default_factory=list,
        description="Signals that were borderline — kept but flagged."
    )
    dropped: list[FilteredSignal] = Field(
        default_factory=list,
        description="Signals removed by the safety filter, with reasons."
    )

    def all_safe_signals(self) -> list[str]:
        """Returns all signals that are safe to use for search and ranking."""
        return self.strong + self.weak + self.borderline

    def to_signals_to_avoid_schema(self) -> list[dict]:
        """
        Returns the signals_to_avoid list in the assignment output schema format.
        """
        result = []
        for fs in self.dropped:
            result.append(fs.model_dump())
        # Always include the standard guardrail note
        result.append(
            "Do not infer religion, politics, health, ethnicity, gender, "
            "family status, or other sensitive personal attributes"
        )
        return result
