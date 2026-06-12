"""
tests/test_validate_products.py

Unit tests for the validate_products node and utility functions.
Tests: weighted scoring, tier assignment, price extraction.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.pricing import extract_price, format_price, price_within_budget
from utils.validation import (
    check_trusted_domain,
    check_product_url_pattern,
    check_india_fit,
    TRUSTED_DOMAINS,
)
from agent.nodes.validate_products import RULE_WEIGHTS, PASS_THRESHOLD, PARTIAL_THRESHOLD


class TestPriceExtraction:
    """Tests for price extraction from text."""

    def test_rupee_symbol(self):
        assert extract_price("₹3,499") == 3499.0
        assert extract_price("₹ 3999") == 3999.0
        assert extract_price("Price: ₹4,299.00") == 4299.0

    def test_rs_format(self):
        assert extract_price("Rs. 2,199") == 2199.0
        assert extract_price("Rs 5000") == 5000.0

    def test_inr_format(self):
        assert extract_price("INR 3500") == 3500.0
        assert extract_price("INR3,999") == 3999.0

    def test_no_price_returns_none(self):
        assert extract_price("No pricing information available") is None
        assert extract_price("") is None
        assert extract_price(None) is None

    def test_lakh_format(self):
        result = extract_price("₹1,24,999")
        assert result == 124999.0

    def test_format_price(self):
        assert format_price(3499.0) == "₹3,499"
        assert format_price(124999.0) == "₹1,24,999"
        assert format_price(500.0) == "₹500"

    def test_price_within_budget(self):
        within, score = price_within_budget(3499, 2000, 5000)
        assert within is True
        assert score == 1.0

        within, score = price_within_budget(5400, 2000, 5000)
        assert within is True
        assert score == 0.6  # within 10% tolerance

        within, score = price_within_budget(8000, 2000, 5000)
        assert within is False
        assert score == 0.0


class TestDomainValidation:
    """Tests for trusted domain checking."""

    def test_amazon_in_trusted(self):
        is_trusted, name = check_trusted_domain("https://www.amazon.in/dp/B09ABC123")
        assert is_trusted is True
        assert name == "Amazon.in"

    def test_flipkart_trusted(self):
        is_trusted, name = check_trusted_domain("https://www.flipkart.com/product/p/item")
        assert is_trusted is True
        assert "Flipkart" in name

    def test_unknown_domain_not_trusted(self):
        is_trusted, name = check_trusted_domain("https://unknownshop.example.com/item/123")
        assert is_trusted is False
        assert name is None

    def test_all_trusted_domains_accessible(self):
        """Verify the trusted domain dict is properly formed."""
        assert len(TRUSTED_DOMAINS) >= 10
        for domain, name in TRUSTED_DOMAINS.items():
            assert isinstance(domain, str)
            assert isinstance(name, str)
            assert len(name) > 0


class TestProductUrlPattern:
    """Tests for product URL pattern matching."""

    def test_amazon_asin_pattern(self):
        assert check_product_url_pattern("https://www.amazon.in/dp/B09ABC12345") is True

    def test_flipkart_pid_pattern(self):
        assert check_product_url_pattern("https://www.flipkart.com/product/p/item123") is True

    def test_generic_product_path(self):
        assert check_product_url_pattern("https://example.com/products/cricket-bat") is True

    def test_search_page_excluded(self):
        assert check_product_url_pattern("https://www.amazon.in/s?k=cricket+bat") is False

    def test_category_page_excluded(self):
        assert check_product_url_pattern("https://www.flipkart.com/sports/cricket") is False


class TestIndiaFit:
    """Tests for India-specificity checking."""

    def test_dot_in_domain(self):
        is_india, score = check_india_fit("https://www.amazon.in/dp/B09", "")
        assert is_india is True
        assert score == 1.0

    def test_india_in_snippet(self):
        is_india, score = check_india_fit("https://example.com/product", "Delivers to India. ₹3999.")
        assert is_india is True
        assert score >= 0.7

    def test_not_india(self):
        is_india, score = check_india_fit("https://amazon.com/dp/B09", "Ships from US. $45.99")
        assert is_india is False
        assert score == 0.0


class TestValidationWeights:
    """Tests for validation rule weight configuration."""

    def test_weights_sum_to_one(self):
        total = sum(RULE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_url_reachable_is_highest_weight(self):
        assert RULE_WEIGHTS["url_reachable"] == max(RULE_WEIGHTS.values())

    def test_thresholds_are_ordered(self):
        assert PASS_THRESHOLD > PARTIAL_THRESHOLD
        assert PARTIAL_THRESHOLD > 0.0
        assert PASS_THRESHOLD <= 1.0
