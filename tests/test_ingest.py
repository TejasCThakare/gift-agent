"""
tests/test_ingest.py

Unit tests for the ingest node.
Tests: validation, normalization, state initialization.
"""

import json
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.nodes.ingest import ingest


VALID_CONTACT = {
    "name": "Aarav Mehta",
    "role": "VP Sales",
    "company": "TechNova Solutions",
    "location": "Mumbai",
    "linkedin_profile": {
        "headline": "VP Sales | Cricket Enthusiast",
        "about": "Love cricket and sales.",
        "experience": [],
        "recent_posts": ["What a match! MI vs CSK"],
        "recent_comments": [],
        "engaged_topics": ["Cricket", "B2B Sales"],
    },
    "relationship_context": {
        "relationship_type": "prospective_customer",
        "last_interaction": "Demo call",
        "business_goal": "Convert to customer",
    },
    "gift_context": {
        "occasion": "Post-demo gift",
        "budget_min": 2000,
        "budget_max": 5000,
        "currency": "INR",
        "country": "India",
    },
}


class TestIngest:
    def test_valid_contact_succeeds(self):
        state = {"contact": VALID_CONTACT}
        result = ingest(state)

        assert result["contact"]["name"] == "Aarav Mehta"
        assert result["relationship_type"] == "prospective_customer"
        assert result["thread_id"].startswith("run_")
        assert "profile_text" in result
        assert "Aarav Mehta" in result["profile_text"]

    def test_state_initialization(self):
        state = {"contact": VALID_CONTACT}
        result = ingest(state)

        # All required state fields should be initialized
        assert result["retry_count"] == 0
        assert result["escalation_flag"] is False
        assert result["review_history"] == []
        assert result["evidence_map"] == {}
        assert result["safe_signals"] == {}
        assert result["raw_results"] == []

    def test_budget_normalization_when_inverted(self):
        contact = dict(VALID_CONTACT)
        contact["gift_context"] = {
            **VALID_CONTACT["gift_context"],
            "budget_min": 8000,
            "budget_max": 2000,  # inverted — should be swapped
        }
        state = {"contact": contact}
        result = ingest(state)

        # Budget should be normalized (min <= max)
        gift_context = result["contact"]["gift_context"]
        assert gift_context["budget_min"] <= gift_context["budget_max"]

    def test_missing_name_raises(self):
        state = {"contact": {"role": "Manager"}}
        with pytest.raises((ValueError, Exception)):
            ingest(state)

    def test_empty_state_raises(self):
        with pytest.raises((ValueError, Exception)):
            ingest({})

    def test_profile_text_contains_signals(self):
        state = {"contact": VALID_CONTACT}
        result = ingest(state)
        profile_text = result["profile_text"]

        assert "Cricket" in profile_text
        assert "VP Sales" in profile_text
        assert "prospective_customer" in profile_text

    def test_thread_id_preserved_if_already_set(self):
        existing_thread = "run_existingthread123"
        state = {"contact": VALID_CONTACT, "thread_id": existing_thread}
        result = ingest(state)
        assert result["thread_id"] == existing_thread

    def test_relationship_type_normalized(self):
        contact = dict(VALID_CONTACT)
        contact["relationship_context"] = {
            **VALID_CONTACT["relationship_context"],
            "relationship_type": "Existing Customer",  # non-standard format
        }
        state = {"contact": contact}
        result = ingest(state)
        # Should be normalized to snake_case
        assert result["relationship_type"] == "existing_customer"

    def test_logs_recorded(self):
        state = {"contact": VALID_CONTACT}
        result = ingest(state)
        assert "logs" in result
        assert "ingest" in result["logs"]
        assert result["logs"]["ingest"].get("latency_ms") is not None
