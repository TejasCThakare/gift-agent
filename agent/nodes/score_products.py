"""
agent/nodes/score_products.py

Node 6: score_products

Deterministic confidence scoring formula. No LLM involved.

Computes 5 sub-scores per product and combines them with fixed weights
into a confidence value. This value is immutable — LLM nodes receive
it as read-only context and cannot change it.

Sub-scores and weights:
  signal_score:          0.35 — how well product matches profile signals
  budget_score:          0.25 — price vs budget fit
  country_score:         0.20 — India-specificity
  search_quality_score:  0.10 — query type (strong > weak > fallback)
  relationship_score:    0.10 — suitability for relationship type

Relationship scoring is a rule-based lookup table — deterministic,
explainable, and implemented in code (not hidden in prompts).
"""

from __future__ import annotations

from agent.state import GiftAgentState
from models.products import ScoredProduct, ProductScores, ValidatedProduct, ValidationRule
from utils.pricing import price_within_budget
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.score_products")


# ── RELATIONSHIP SCORING TABLE ────────────────────────────────────────────────
# Defines base relationship score and preference metadata for each type.
# This is the single source of truth for relationship-based scoring.

RELATIONSHIP_SCORE_TABLE: dict[str, dict] = {
    "prospective_customer": {
        "base_score": 0.80,
        "prefer_categories": [
            "premium stationery", "tech accessories", "business books",
            "experience voucher", "desk accessories", "branded merchandise",
        ],
        "avoid_categories": [
            "overly personal", "consumables", "alcohol", "clothing",
            "health and wellness", "religious items",
        ],
        "note": "Professional, low-risk, builds relationship without pressure",
    },
    "existing_customer": {
        "base_score": 0.90,
        "prefer_categories": [
            "personal interest gifts", "premium experiences", "customised gifts",
            "thank you gifts", "loyalty recognition",
        ],
        "avoid_categories": [
            "generic corporate", "overly cheap", "inappropriate consumables",
        ],
        "note": "Can be more personal — relationship is established",
    },
    "colleague": {
        "base_score": 0.70,
        "prefer_categories": [
            "team celebration", "fun accessories", "food and beverages (non-alcohol)",
            "hobby gifts", "casual professional",
        ],
        "avoid_categories": [
            "overly expensive", "too intimate", "alcohol",
        ],
        "note": "Casual professional — appropriate for team contexts",
    },
    "executive": {
        "base_score": 0.85,
        "prefer_categories": [
            "premium minimal", "luxury desk accessories", "artisan products",
            "experience vouchers", "quality branded items",
        ],
        "avoid_categories": [
            "cheap novelty", "mass-market trinkets", "overly casual",
        ],
        "note": "Premium, tasteful, minimal — quality over quantity",
    },
    "founder": {
        "base_score": 0.80,
        "prefer_categories": [
            "mission-aligned", "startup culture", "thought leadership books",
            "quality tools", "experience gifts",
        ],
        "avoid_categories": [
            "generic corporate", "low effort", "mass-market",
        ],
        "note": "Mission-aligned and thoughtful — reflects shared values",
    },
    "partner": {
        "base_score": 0.85,
        "prefer_categories": [
            "collaborative", "shared value", "premium professional",
            "co-branded opportunity", "quality business gifts",
        ],
        "avoid_categories": [
            "one-sided branding", "overly personal",
        ],
        "note": "Partnership-appropriate — mutual respect and professionalism",
    },
    "unknown": {
        "base_score": 0.65,
        "prefer_categories": [
            "safe professional", "universally appropriate", "neutral",
        ],
        "avoid_categories": [
            "anything personal or sensitive",
        ],
        "note": "Default safe professional gift",
    },
}


def score_products(state: GiftAgentState) -> dict:
    """
    Compute deterministic confidence scores for all validated products.
    Excludes "fail" tier products. Sorts by confidence descending.

    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "score_products")

    validated_products = state.get("validated_products", [])
    safe_signals = state.get("safe_signals", {})
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})
    relationship_type = state.get("relationship_type", "unknown")

    budget_min = float(gift_context.get("budget_min", 0))
    budget_max = float(gift_context.get("budget_max", 5000))
    widened_budget_max = state.get("widened_budget_max", None)
    effective_budget_max = widened_budget_max if widened_budget_max else budget_max

    strong_signals = safe_signals.get("strong", [])
    weak_signals = safe_signals.get("weak", [])
    borderline_signals = safe_signals.get("borderline", [])
    all_signals = strong_signals + weak_signals + borderline_signals

    relationship_config = RELATIONSHIP_SCORE_TABLE.get(
        relationship_type,
        RELATIONSHIP_SCORE_TABLE["unknown"],
    )
    relationship_base_score = relationship_config["base_score"]

    scored: list[dict] = []

    for product_dict in validated_products:
        # Skip fail tier products unless we're in escalation mode
        tier = product_dict.get("validation_tier", "fail")
        if tier == "fail":
            continue

        product = ValidatedProduct.model_validate(product_dict)

        scores = _compute_scores(
            product=product,
            strong_signals=strong_signals,
            weak_signals=weak_signals + borderline_signals,
            budget_min=budget_min,
            budget_max=effective_budget_max,
            relationship_base_score=relationship_base_score,
        )

        scored_product = ScoredProduct.from_validated(
            product=product,
            scores=scores,
        )

        logger.debug(
            "%s | conf=%.3f | risk=%s | signal=%.2f budget=%.2f country=%.2f",
            scored_product.title[:40],
            scored_product.confidence,
            scored_product.risk_level,
            scores.signal_score,
            scores.budget_score,
            scores.country_score,
        )

        scored.append(scored_product.model_dump())

    # Sort by confidence descending
    scored.sort(key=lambda p: p["confidence"], reverse=True)

    logger.info(
        "Scored %d products. Top confidence: %.3f",
        len(scored),
        scored[0]["confidence"] if scored else 0.0,
    )

    logs = log_node_end(logs, "score_products")

    return {
        "scored_products": scored,
        "logs": logs,
    }


def _compute_scores(
    product: ValidatedProduct,
    strong_signals: list[str],
    weak_signals: list[str],
    budget_min: float,
    budget_max: float,
    relationship_base_score: float,
) -> ProductScores:
    """Compute all 5 sub-scores for a product."""

    # ── 1. Signal score ───────────────────────────────────────────────────
    # Check if product title/snippet matches any safe signals
    search_text = f"{product.title} {product.snippet}".lower()
    signal_score = 0.0

    for signal in strong_signals:
        # Extract key terms from signal for matching
        key_terms = _extract_key_terms(signal)
        if any(term in search_text for term in key_terms):
            signal_score = max(signal_score, 1.0)
            break

    if signal_score < 1.0:
        for signal in weak_signals:
            key_terms = _extract_key_terms(signal)
            if any(term in search_text for term in key_terms):
                signal_score = max(signal_score, 0.5)
                break

    # ── 2. Budget score ───────────────────────────────────────────────────
    if product.extracted_price is not None:
        _, budget_score = price_within_budget(
            price=product.extracted_price,
            budget_min=budget_min,
            budget_max=budget_max,
        )
    else:
        # No price found — partial credit (give benefit of doubt)
        budget_score = 0.5

    # ── 3. Country score ──────────────────────────────────────────────────
    url_lower = product.url.lower()
    snippet_lower = (product.snippet or "").lower()

    if url_lower.endswith(".in") or any(
        d in url_lower for d in [".in/", "amazon.in", "flipkart.com",
                                  "myntra.com", "nykaa.com", "tatacliq.com"]
    ):
        country_score = 1.0
    elif any(k in snippet_lower or k in url_lower
             for k in ["india", "₹", "inr", "rupee", "rs."]):
        country_score = 0.7
    else:
        country_score = 0.0

    # ── 4. Search quality score ───────────────────────────────────────────
    query_type = product.query_type or "fallback"
    search_quality_score = {
        "strong": 1.0,
        "weak": 0.7,
        "fallback": 0.5,
    }.get(query_type, 0.5)

    # ── 5. Relationship score ─────────────────────────────────────────────
    # The base relationship score already encodes appropriateness.
    # Adjust slightly based on validation tier.
    tier_adjustment = {
        "pass": 0.0,
        "partial": -0.1,
        "fail": -0.3,
    }.get(product.validation_tier, 0.0)

    relationship_score = max(0.0, relationship_base_score + tier_adjustment)

    return ProductScores(
        signal_score=round(signal_score, 4),
        budget_score=round(budget_score, 4),
        country_score=round(country_score, 4),
        search_quality_score=round(search_quality_score, 4),
        relationship_score=round(relationship_score, 4),
    )


def _extract_key_terms(signal: str) -> list[str]:
    """
    Extract searchable key terms from a signal string.
    E.g. "interested in cricket" → ["cricket"]
         "works in SaaS sales" → ["saas", "sales"]
    """
    # Remove common filler words
    filler = {
        "interested", "in", "the", "a", "an", "is", "are", "was", "for",
        "of", "to", "and", "or", "with", "may", "appreciate", "enjoys",
        "engages", "works", "has", "been", "that", "this", "about",
    }
    terms = [
        word.lower()
        for word in signal.replace("-", " ").split()
        if word.lower() not in filler and len(word) > 3
    ]
    return terms
