"""
agent/nodes/validate_products.py

Node 5: validate_products

Rejects articles, blog posts, category pages, and search result pages.
Only accepts URLs that look like specific product pages.
Trusted domains skip HTTP reachability check but still must pass
product URL pattern check.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from agent.state import GiftAgentState
from models.products import RawProduct, ValidatedProduct, ValidationRule
from utils.pricing import extract_price, format_price, price_within_budget
from utils.validation import (
    check_url_reachable,
    check_trusted_domain,
    check_product_url_pattern,
    check_india_fit,
)
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.validate_products")

PASS_THRESHOLD = float(os.getenv("VALIDATION_PASS_THRESHOLD", "0.75"))
PARTIAL_THRESHOLD = float(os.getenv("VALIDATION_PARTIAL_THRESHOLD", "0.40"))

RULE_WEIGHTS = {
    "url_reachable": 0.25,
    "trusted_domain": 0.30,
    "product_url_pattern": 0.20,
    "price_detected": 0.10,
    "budget_fit": 0.10,
    "india_fit": 0.05,
}

# URL patterns that indicate a page is NOT a product page
NON_PRODUCT_URL_PATTERNS = [
    "/search", "?q=", "/browse", "/category", "/categories",
    "/b?", "node=", "/collections/", "/blogs/", "/blog/",
    "best-gift", "gift-ideas", "gift-guide", "gift-list",
    "under-5000", "under-3000", "under-1000",
    "best-pens", "best-books", "top-10", "top-5",
    "/tag/", "/tags/", "/author/", "/page/",
    "wikipedia.org", "youtube.com", "instagram.com",
    "facebook.com", "twitter.com", "linkedin.com",
    "indiatimes.com", "ndtv.com", "hindustantimes.com",
    "thehindu.com", "livemint.com", "economictimes",
    "amarujala.com", "navbharattimes", "jagran.com",
    "interestingarticlestoread", "giftony.com",
    "betmockers.com", "pitchhigh.com", "bel-in.com",
    "almanac.com", "calendarr.com", "vedantu.com",
    "grokipedia.com", "startpage.com", "yandex.com",
    "yahoo.com/search", "brave.com/search",
    "knightsandwalker.com/en-in/blogs",
    "/gifts-under", "/gift-under",
]


def validate_products(state: GiftAgentState) -> dict:
    logs = log_node_start(state.get("logs", {}), "validate_products")

    raw_results = state.get("raw_results", [])
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})

    budget_min = float(gift_context.get("budget_min", 0))
    budget_max = float(gift_context.get("budget_max", 5000))
    widened_budget_max = state.get("widened_budget_max", None)
    effective_budget_max = widened_budget_max if widened_budget_max else budget_max

    logger.info(
        "Validating %d raw results (budget: %.0f–%.0f)",
        len(raw_results), budget_min, effective_budget_max,
    )

    validated: list[dict] = []

    for raw in raw_results:
        product = RawProduct.model_validate(raw) if isinstance(raw, dict) else raw

        # Hard reject non-product URLs before any scoring
        if _is_non_product_url(product.url, product.title):
            logger.debug("Hard reject (non-product): %s", product.url[:80])
            continue

        validated_product = _validate_single(
            product=product,
            budget_min=budget_min,
            budget_max=effective_budget_max,
        )
        validated.append(validated_product.model_dump())

    pass_count = sum(1 for p in validated if p["validation_tier"] == "pass")
    partial_count = sum(1 for p in validated if p["validation_tier"] == "partial")
    fail_count = sum(1 for p in validated if p["validation_tier"] == "fail")

    logger.info(
        "Validation results: %d pass, %d partial, %d fail (hard rejected: %d)",
        pass_count, partial_count, fail_count,
        len(raw_results) - len(validated),
    )

    logs = log_node_end(logs, "validate_products")

    return {
        "validated_products": validated,
        "logs": logs,
    }


def _is_non_product_url(url: str, title: str) -> bool:
    """
    Hard reject URLs that are clearly not product pages.
    Returns True if the URL should be rejected.
    """
    url_lower = url.lower()
    title_lower = (title or "").lower()

    # Check URL against non-product patterns
    for pattern in NON_PRODUCT_URL_PATTERNS:
        if pattern in url_lower:
            return True

    # Reject titles that are clearly articles/lists not products
    article_title_signals = [
        "best gift for", "best gifts for", "gift ideas", "gift guide",
        "top 10", "top 5", "best pens", "best books",
        "online shopping from a great selection",  # Amazon category page
        "great indian", "buy mobile phones online",  # Amazon homepage/category
        "cheapest places", "budget destination",
        "best girlfriend", "best boyfriend",
    ]
    for signal in article_title_signals:
        if signal in title_lower:
            return True

    # Reject Amazon category/browse pages specifically
    if "amazon.in" in url_lower:
        # Allow: /dp/ (product), /gp/product/ (product)
        # Reject: /b? (browse), /s? (search), /Best-Sellers (category)
        if any(x in url_lower for x in ["/b?", "/s?", "best-sellers",
                                          "node=", "/stores/", "great-selection"]):
            return True

    return False


def _validate_single(
    product: RawProduct,
    budget_min: float,
    budget_max: float,
) -> ValidatedProduct:
    report: dict[str, ValidationRule] = {}
    validation_score = 0.0

    # Rule 2: Trusted domain
    is_trusted, store_name = check_trusted_domain(product.url)
    rule2 = ValidationRule(
        passed=is_trusted,
        weight=RULE_WEIGHTS["trusted_domain"],
        value=store_name,
        note=f"Store: {store_name}" if store_name else "Not a trusted Indian e-commerce domain",
    )
    report["trusted_domain"] = rule2
    if is_trusted:
        validation_score += RULE_WEIGHTS["trusted_domain"]

    # Rule 1: URL reachable — skip for trusted domains
    if is_trusted:
        rule1 = ValidationRule(
            passed=True,
            weight=RULE_WEIGHTS["url_reachable"],
            value="trusted domain — reachability assumed",
            note="Trusted domain — HTTP check skipped",
        )
        validation_score += RULE_WEIGHTS["url_reachable"]
    else:
        reachable, note = check_url_reachable(product.url)
        rule1 = ValidationRule(
            passed=reachable,
            weight=RULE_WEIGHTS["url_reachable"],
            value=note,
            note=note,
        )
        if reachable:
            validation_score += RULE_WEIGHTS["url_reachable"]
    report["url_reachable"] = rule1

    # Rule 3: Product URL pattern
    is_product_url = check_product_url_pattern(product.url)
    rule3 = ValidationRule(
        passed=is_product_url,
        weight=RULE_WEIGHTS["product_url_pattern"],
        value=product.url,
        note="Product page URL" if is_product_url else "Not a specific product page",
    )
    report["product_url_pattern"] = rule3
    if is_product_url:
        validation_score += RULE_WEIGHTS["product_url_pattern"]

    # Rule 4: Price detected
    search_text = f"{product.title} {product.snippet}"
    extracted_price = extract_price(search_text)
    price_str = format_price(extracted_price) if extracted_price else None

    if extracted_price is not None:
        price_passed = True
        price_credit = RULE_WEIGHTS["price_detected"]
    elif is_trusted and is_product_url:
        price_passed = True
        price_credit = RULE_WEIGHTS["price_detected"] * 0.5
        price_str = "Price on product page"
    else:
        price_passed = False
        price_credit = 0.0

    rule4 = ValidationRule(
        passed=price_passed,
        weight=RULE_WEIGHTS["price_detected"],
        value=price_str,
        note=f"Price: {price_str}" if price_str else "No price in snippet",
    )
    report["price_detected"] = rule4
    validation_score += price_credit

    # Rule 5: Budget fit
    within_budget = False
    budget_score_value = 0.0
    budget_note = "Price unknown"

    if extracted_price is not None:
        within_budget, budget_score_value = price_within_budget(
            price=extracted_price,
            budget_min=budget_min,
            budget_max=budget_max,
        )
        if within_budget and extracted_price <= budget_max:
            budget_note = f"Within budget (₹{extracted_price:.0f} ≤ ₹{budget_max:.0f})"
        elif within_budget:
            budget_note = f"Slightly over (₹{extracted_price:.0f} vs ₹{budget_max:.0f})"
        else:
            budget_note = f"Over budget (₹{extracted_price:.0f} > ₹{budget_max:.0f})"
    elif is_trusted and is_product_url:
        budget_score_value = 0.5
        within_budget = True
        budget_note = "Price on product page — assumed within budget"

    rule5 = ValidationRule(
        passed=within_budget,
        weight=RULE_WEIGHTS["budget_fit"],
        value=price_str,
        note=budget_note,
    )
    report["budget_fit"] = rule5
    validation_score += RULE_WEIGHTS["budget_fit"] * budget_score_value

    # Rule 6: India fit
    is_india, country_score_value = check_india_fit(product.url, product.snippet)
    if is_trusted:
        is_india = True
        country_score_value = 1.0

    rule6 = ValidationRule(
        passed=is_india,
        weight=RULE_WEIGHTS["india_fit"],
        value="India" if is_india else None,
        note="India-specific" if is_india else "India fit not confirmed",
    )
    report["india_fit"] = rule6
    validation_score += RULE_WEIGHTS["india_fit"] * country_score_value

    validation_score = round(min(validation_score, 1.0), 4)

    if validation_score >= PASS_THRESHOLD:
        tier = "pass"
    elif validation_score >= PARTIAL_THRESHOLD:
        tier = "partial"
    else:
        tier = "fail"

    logger.debug(
        "%s | score=%.3f | tier=%s | trusted=%s | product_url=%s",
        product.title[:50], validation_score, tier, is_trusted, is_product_url,
    )

    return ValidatedProduct(
        title=product.title,
        url=product.url,
        snippet=product.snippet,
        query_source=product.query_source,
        query_type=product.query_type,
        validation_report=report,
        validation_score=validation_score,
        validation_tier=tier,
        extracted_price=extracted_price,
        extracted_price_str=price_str,
        store=store_name or _infer_store_from_url(product.url),
    )


def _infer_store_from_url(url: str) -> str:
    url_lower = url.lower()
    if "amazon" in url_lower:
        return "Amazon.in"
    if "flipkart" in url_lower:
        return "Flipkart"
    if "myntra" in url_lower:
        return "Myntra"
    if "nykaa" in url_lower:
        return "Nykaa"
    try:
        hostname = urlparse(url).hostname or ""
        parts = hostname.replace("www.", "").split(".")
        if parts:
            return parts[0].title()
    except Exception:
        pass
    return "Online Store"