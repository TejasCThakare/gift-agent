"""
tests/test_score_products.py

Unit tests for the score_products node.
Tests: confidence formula, determinism, risk level assignment.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.nodes.score_products import score_products, RELATIONSHIP_SCORE_TABLE, _compute_scores, _extract_key_terms
from models.products import ValidatedProduct, ValidationRule, ProductScores, ScoredProduct, CONFIDENCE_WEIGHTS


def make_validated_product(
    title="Cricket Bat SG",
    url="https://www.amazon.in/dp/B09ABC12345",
    snippet="Cricket Bat ₹3499 India",
    query_type="strong",
    validation_tier="pass",
    validation_score=0.80,
    extracted_price=3499.0,
    extracted_price_str="₹3,499",
    store="Amazon.in",
) -> dict:
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "query_source": "cricket gift India",
        "query_type": query_type,
        "validation_report": {
            "url_reachable": {"passed": True, "weight": 0.30, "value": "200", "note": ""},
            "trusted_domain": {"passed": True, "weight": 0.25, "value": "Amazon.in", "note": ""},
            "product_url_pattern": {"passed": True, "weight": 0.15, "value": None, "note": ""},
            "price_detected": {"passed": True, "weight": 0.15, "value": "₹3,499", "note": ""},
            "budget_fit": {"passed": True, "weight": 0.10, "value": None, "note": ""},
            "india_fit": {"passed": True, "weight": 0.05, "value": "India", "note": ""},
        },
        "validation_score": validation_score,
        "validation_tier": validation_tier,
        "extracted_price": extracted_price,
        "extracted_price_str": extracted_price_str,
        "store": store,
    }


class TestConfidenceFormula:
    """Tests that confidence = weighted sum of sub-scores."""

    def test_confidence_is_weighted_sum(self):
        scores = ProductScores(
            signal_score=1.0,
            budget_score=1.0,
            country_score=1.0,
            search_quality_score=1.0,
            relationship_score=0.8,
        )
        product = ValidatedProduct.model_validate(make_validated_product())
        scored = ScoredProduct.from_validated(product, scores)

        expected = (
            1.0 * CONFIDENCE_WEIGHTS["signal"]
            + 1.0 * CONFIDENCE_WEIGHTS["budget"]
            + 1.0 * CONFIDENCE_WEIGHTS["country"]
            + 1.0 * CONFIDENCE_WEIGHTS["search_quality"]
            + 0.8 * CONFIDENCE_WEIGHTS["relationship"]
        )
        assert abs(scored.confidence - expected) < 0.001

    def test_weights_sum_to_one(self):
        total = sum(CONFIDENCE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_perfect_score_gives_high_confidence(self):
        scores = ProductScores(
            signal_score=1.0, budget_score=1.0, country_score=1.0,
            search_quality_score=1.0, relationship_score=1.0,
        )
        product = ValidatedProduct.model_validate(make_validated_product())
        scored = ScoredProduct.from_validated(product, scores)
        assert scored.confidence == 1.0
        assert scored.risk_level == "low"

    def test_zero_score_gives_high_risk(self):
        scores = ProductScores(
            signal_score=0.0, budget_score=0.0, country_score=0.0,
            search_quality_score=0.0, relationship_score=0.0,
        )
        product = ValidatedProduct.model_validate(make_validated_product())
        scored = ScoredProduct.from_validated(product, scores)
        assert scored.confidence == 0.0
        assert scored.risk_level == "high"

    def test_risk_level_thresholds(self):
        def make_scored(confidence):
            scores = ProductScores(signal_score=confidence, budget_score=0.0,
                                   country_score=0.0, search_quality_score=0.0,
                                   relationship_score=0.0)
            # Manually compute to get specific confidence value
            product = ValidatedProduct.model_validate(make_validated_product())
            return ScoredProduct.from_validated(product, scores)

        # Test that risk levels follow the documented thresholds
        # (exact confidence values depend on weights but we can verify ordering)
        scores_high = ProductScores(
            signal_score=0.0, budget_score=0.1, country_score=0.0,
            search_quality_score=0.0, relationship_score=0.0
        )
        scores_medium = ProductScores(
            signal_score=0.7, budget_score=0.7, country_score=0.0,
            search_quality_score=0.0, relationship_score=0.0
        )
        scores_low = ProductScores(
            signal_score=1.0, budget_score=1.0, country_score=1.0,
            search_quality_score=1.0, relationship_score=1.0
        )
        product = ValidatedProduct.model_validate(make_validated_product())

        high_risk = ScoredProduct.from_validated(product, scores_high)
        medium_risk = ScoredProduct.from_validated(product, scores_medium)
        low_risk = ScoredProduct.from_validated(product, scores_low)

        assert high_risk.risk_level == "high"
        assert medium_risk.risk_level in ("medium", "high")
        assert low_risk.risk_level == "low"

    def test_confidence_is_deterministic(self):
        """Same inputs always produce the same confidence."""
        product_dict = make_validated_product()
        state = {
            "validated_products": [product_dict],
            "safe_signals": {"strong": ["cricket"], "weak": [], "borderline": []},
            "contact": {"gift_context": {"budget_min": 2000, "budget_max": 5000}},
            "relationship_type": "prospective_customer",
            "logs": {},
        }
        result1 = score_products(state)
        result2 = score_products(state)

        conf1 = result1["scored_products"][0]["confidence"]
        conf2 = result2["scored_products"][0]["confidence"]
        assert conf1 == conf2


class TestRelationshipScoring:
    """Tests for relationship-based scoring lookup table."""

    def test_all_relationship_types_defined(self):
        required = [
            "prospective_customer", "existing_customer", "colleague",
            "executive", "founder", "partner", "unknown",
        ]
        for rel_type in required:
            assert rel_type in RELATIONSHIP_SCORE_TABLE

    def test_scores_in_valid_range(self):
        for rel_type, config in RELATIONSHIP_SCORE_TABLE.items():
            score = config["base_score"]
            assert 0.0 <= score <= 1.0, f"{rel_type} score {score} out of range"

    def test_existing_customer_higher_than_prospect(self):
        existing = RELATIONSHIP_SCORE_TABLE["existing_customer"]["base_score"]
        prospect = RELATIONSHIP_SCORE_TABLE["prospective_customer"]["base_score"]
        assert existing >= prospect

    def test_relationship_score_affects_confidence(self):
        product_dict = make_validated_product()
        base_state = {
            "validated_products": [product_dict],
            "safe_signals": {"strong": [], "weak": [], "borderline": []},
            "contact": {"gift_context": {"budget_min": 0, "budget_max": 10000}},
            "logs": {},
        }

        state_prospect = {**base_state, "relationship_type": "prospective_customer"}
        state_existing = {**base_state, "relationship_type": "existing_customer"}

        result_prospect = score_products(state_prospect)
        result_existing = score_products(state_existing)

        conf_prospect = result_prospect["scored_products"][0]["confidence"]
        conf_existing = result_existing["scored_products"][0]["confidence"]

        # Existing customer should score higher than prospect
        assert conf_existing >= conf_prospect


class TestScoreProducts:
    """Integration tests for the full score_products node."""

    def test_fail_tier_excluded(self):
        fail_product = make_validated_product(validation_tier="fail")
        state = {
            "validated_products": [fail_product],
            "safe_signals": {"strong": [], "weak": [], "borderline": []},
            "contact": {"gift_context": {"budget_min": 0, "budget_max": 10000}},
            "relationship_type": "unknown",
            "logs": {},
        }
        result = score_products(state)
        assert len(result["scored_products"]) == 0

    def test_sorted_by_confidence_descending(self):
        products = [
            make_validated_product(title="Low Confidence Product",
                                   url="https://amazon.in/dp/LOW", snippet="random item"),
            make_validated_product(title="SG Cricket Bat",
                                   url="https://amazon.in/dp/CRICKET",
                                   snippet="Cricket Bat ₹3499 India"),
        ]
        state = {
            "validated_products": products,
            "safe_signals": {"strong": ["cricket"], "weak": [], "borderline": []},
            "contact": {"gift_context": {"budget_min": 2000, "budget_max": 5000}},
            "relationship_type": "prospective_customer",
            "logs": {},
        }
        result = score_products(state)
        scored = result["scored_products"]

        if len(scored) >= 2:
            assert scored[0]["confidence"] >= scored[1]["confidence"]

    def test_extract_key_terms(self):
        terms = _extract_key_terms("interested in cricket")
        assert "cricket" in terms
        assert "interested" not in terms  # filtered as filler

        terms2 = _extract_key_terms("may appreciate business strategy books")
        assert "business" in terms2
        assert "strategy" in terms2
        assert "books" in terms2
        assert "appreciate" not in terms2
