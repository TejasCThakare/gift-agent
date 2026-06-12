"""
utils/pricing.py

Price extraction from product snippets and page titles.

Handles Indian price formats:
  ₹3,999  ₹ 3999  Rs. 3,999  INR 3999  Rs 3,999
  ₹3999.00  ₹ 3,99,999 (lakh format)

Returns the price as a float and as a formatted display string.
Returns None if no price is found.
"""

from __future__ import annotations

import re
from typing import Optional


# Patterns ordered by specificity.
# Use simple greedy digit+comma match then strip commas manually.
_PRICE_PATTERNS = [
    # ₹ symbol
    r"₹\s*([\d,]+(?:\.\d{1,2})?)",
    # Rs. or Rs
    r"(?:Rs\.?\s*)([\d,]+(?:\.\d{1,2})?)",
    # INR prefix
    r"INR\s*([\d,]+(?:\.\d{1,2})?)",
    # Suffix ₹
    r"([\d,]+(?:\.\d{1,2})?)\s*₹",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _PRICE_PATTERNS]


def extract_price(text: str) -> Optional[float]:
    """
    Extract the first price found in the text.

    Args:
        text: Any string containing a price — snippet, title, etc.

    Returns:
        Price as a float (e.g. 3999.0), or None if not found.
    """
    if not text:
        return None

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            price_str = match.group(1)
            # Remove commas from Indian number format
            price_str = price_str.replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                continue

    return None


def format_price(price: float, currency: str = "INR") -> str:
    """
    Format a price float for display in Indian format.

    Args:
        price: Price as float.
        currency: Currency code (default INR).

    Returns:
        Formatted string like "₹3,999" or "₹1,24,999"
    """
    if currency == "INR":
        return "₹" + _format_indian_number(int(round(price)))
    return f"{currency} {price:,.2f}"


def _format_indian_number(n: int) -> str:
    """
    Format an integer in Indian number system (with lakh separators).
    1234567 → 12,34,567
    """
    s = str(n)
    if len(s) <= 3:
        return s
    # Last 3 digits
    result = s[-3:]
    s = s[:-3]
    # Remaining digits in groups of 2
    while len(s) > 2:
        result = s[-2:] + "," + result
        s = s[:-2]
    if s:
        result = s + "," + result
    return result


def price_within_budget(
    price: float,
    budget_min: float,
    budget_max: float,
    tolerance: float = 0.10,
) -> tuple[bool, float]:
    """
    Check if a price is within budget, with optional tolerance.

    Args:
        price: The product price.
        budget_min: Minimum acceptable price.
        budget_max: Maximum acceptable price.
        tolerance: Fractional tolerance above budget_max (default 0.10 = 10%).

    Returns:
        (within_budget, budget_score) tuple.
        budget_score: 1.0 if within range, 0.6 if within tolerance, 0.0 if over.
    """
    if budget_min <= price <= budget_max:
        return True, 1.0
    if price <= budget_max * (1 + tolerance):
        return True, 0.6
    return False, 0.0
