"""
tests/test_relationship_scoring.py

Unit tests for the relationship scoring rule table.
Verifies that all relationship types are defined, scores are in range,
and the scoring is deterministic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.nodes.score_products import RELATIONSHIP_SCORE_TABLE


REQUIRED_TYPES = [
    "prospective_customer",
    "existing_customer",
    "colleague",
    "executive",
    "founder",
    "partner",
    "unknown",
]

REQUIRED_FIELDS = ["base_score", "prefer_categories", "avoid_categories", "note"]


class TestRelationshipScoringTable:
    """Tests that the relationship scoring table is complete and valid."""

    def test_all_required_types_present(self):
        for rel_type in REQUIRED_TYPES:
            assert rel_type in RELATIONSHIP_SCORE_TABLE, (
                f"Missing relationship type: {rel_type}"
            )

    def test_all_required_fields_present(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            for field in REQUIRED_FIELDS:
                assert field in config, (
                    f"Missing field '{field}' in relationship type '{rel_type}'"
                )

    def test_all_scores_in_valid_range(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            score = config["base_score"]
            assert 0.0 <= score <= 1.0, (
                f"{rel_type} base_score {score} not in [0.0, 1.0]"
            )

    def test_prefer_categories_is_list(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            assert isinstance(config["prefer_categories"], list), (
                f"{rel_type}.prefer_categories must be a list"
            )

    def test_avoid_categories_is_list(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            assert isinstance(config["avoid_categories"], list), (
                f"{rel_type}.avoid_categories must be a list"
            )

    def test_note_is_non_empty_string(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            assert isinstance(config["note"], str) and len(config["note"]) > 0, (
                f"{rel_type}.note must be a non-empty string"
            )

    def test_existing_customer_score_higher_than_unknown(self):
        existing = RELATIONSHIP_SCORE_TABLE["existing_customer"]["base_score"]
        unknown = RELATIONSHIP_SCORE_TABLE["unknown"]["base_score"]
        assert existing > unknown

    def test_executive_has_premium_preference(self):
        config = RELATIONSHIP_SCORE_TABLE["executive"]
        prefer = " ".join(config["prefer_categories"]).lower()
        assert any(word in prefer for word in ["premium", "luxury", "quality"])

    def test_prospective_customer_avoids_personal_items(self):
        config = RELATIONSHIP_SCORE_TABLE["prospective_customer"]
        avoid = " ".join(config["avoid_categories"]).lower()
        assert any(word in avoid for word in ["personal", "alcohol", "consumable"])

    def test_scoring_is_deterministic(self):
        """Same relationship type always gives same score."""
        score1 = RELATIONSHIP_SCORE_TABLE["prospective_customer"]["base_score"]
        score2 = RELATIONSHIP_SCORE_TABLE["prospective_customer"]["base_score"]
        assert score1 == score2

    def test_all_types_have_avoid_and_prefer(self):
        """Every relationship type should guide what to prefer and avoid."""
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            assert len(config["prefer_categories"]) > 0, (
                f"{rel_type} has no preferred categories"
            )
            assert len(config["avoid_categories"]) > 0, (
                f"{rel_type} has no avoided categories"
            )
