"""
utils/validation.py

Utilities for validating product URLs.

Checks:
  1. URL is reachable (HTTP HEAD or GET, timeout=5s)
  2. URL is from a trusted domain (Indian e-commerce allowlist)
  3. URL looks like a product page (path pattern matching)
  4. URL is from a .in domain or India-specific subdomain

All checks are deterministic — no LLM involved.
"""

from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

URL_VALIDATION_TIMEOUT = float(os.getenv("URL_VALIDATION_TIMEOUT", "5"))

# Trusted Indian e-commerce domains
TRUSTED_DOMAINS = {
    "amazon.in": "Amazon.in",
    "www.amazon.in": "Amazon.in",
    "flipkart.com": "Flipkart",
    "www.flipkart.com": "Flipkart",
    "myntra.com": "Myntra",
    "www.myntra.com": "Myntra",
    "nykaa.com": "Nykaa",
    "www.nykaa.com": "Nykaa",
    "tatacliq.com": "Tata CLiQ",
    "www.tatacliq.com": "Tata CLiQ",
    "reliancedigital.in": "Reliance Digital",
    "www.reliancedigital.in": "Reliance Digital",
    "croma.com": "Croma",
    "www.croma.com": "Croma",
    "meesho.com": "Meesho",
    "www.meesho.com": "Meesho",
    "snapdeal.com": "Snapdeal",
    "www.snapdeal.com": "Snapdeal",
    "ajio.com": "AJIO",
    "www.ajio.com": "AJIO",
    "bewakoof.com": "Bewakoof",
    "www.bewakoof.com": "Bewakoof",
    "thesouledstore.com": "The Souled Store",
    "www.thesouledstore.com": "The Souled Store",
    "firstcry.com": "FirstCry",
    "www.firstcry.com": "FirstCry",
    "pepperfry.com": "Pepperfry",
    "www.pepperfry.com": "Pepperfry",
    "urbanladder.com": "Urban Ladder",
    "www.urbanladder.com": "Urban Ladder",
    "bigbasket.com": "BigBasket",
    "www.bigbasket.com": "BigBasket",
    "blinkit.com": "Blinkit",
    "swiggyinstamart.com": "Swiggy Instamart",
    "purplle.com": "Purplle",
    "www.purplle.com": "Purplle",
    "boat-lifestyle.com": "boAt",
    "www.boat-lifestyle.com": "boAt",
    "noise.com": "Noise",
    "www.noise.com": "Noise",
    "firestoneindia.in": "Firestone India",
    "crossword.in": "Crossword",
    "www.crossword.in": "Crossword",
    "indiagift.in": "India Gift",
    "www.indiagift.in": "India Gift",
    "archiesonline.com": "Archies",
    "www.archiesonline.com": "Archies",
}

# Product URL patterns — indicate a specific product page vs a category page
_PRODUCT_URL_PATTERNS = [
    r"/dp/[A-Z0-9]{10}",          # Amazon ASIN
    r"/p/[a-zA-Z0-9]+",           # Flipkart, Myntra
    r"/product/",                   # Generic
    r"/products/",                  # Generic plural
    r"/item/",                      # Snapdeal, Meesho
    r"/buy/",                       # Various
    r"/[a-z0-9-]+-pid-\d+",       # Flipkart PID pattern
    r"/[a-z0-9-]+-\d+\.html",     # Various Indian e-commerce
    r"/itm/",                       # eBay
    r"/ip/",                        # Walmart
    r"-[A-Z0-9]{10}$",             # Amazon URL ending
]

_COMPILED_PRODUCT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _PRODUCT_URL_PATTERNS
]


def check_url_reachable(url: str) -> tuple[bool, str]:
    """
    Check if a URL is reachable via HTTP.
    Uses HEAD first, falls back to GET if HEAD is blocked.

    Returns:
        (reachable, note) tuple.
    """
    if not url or not url.startswith("http"):
        return False, "Invalid URL format"

    try:
        with httpx.Client(
            timeout=URL_VALIDATION_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; GiftAgentBot/1.0; "
                    "+https://github.com/gift-agent)"
                )
            },
        ) as client:
            # Try HEAD first (faster, less bandwidth)
            try:
                response = client.head(url)
                if response.status_code < 400:
                    return True, f"HTTP {response.status_code}"
                # Some sites block HEAD — try GET
                if response.status_code in (405, 403):
                    response = client.get(url)
                    if response.status_code < 400:
                        return True, f"HTTP {response.status_code} (GET fallback)"
                    return False, f"HTTP {response.status_code}"
                return False, f"HTTP {response.status_code}"
            except httpx.TooManyRedirects:
                return False, "Too many redirects"

    except httpx.TimeoutException:
        return False, "Request timed out"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def check_trusted_domain(url: str) -> tuple[bool, Optional[str]]:
    """
    Check if a URL is from a trusted Indian e-commerce domain.

    Returns:
        (is_trusted, store_name) tuple.
        store_name is the human-readable store name if trusted, else None.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname_lower = hostname.lower()

        # Direct lookup
        if hostname_lower in TRUSTED_DOMAINS:
            return True, TRUSTED_DOMAINS[hostname_lower]

        # Check if any trusted domain is a suffix
        for domain, name in TRUSTED_DOMAINS.items():
            if hostname_lower.endswith("." + domain) or hostname_lower == domain:
                return True, name

        return False, None
    except Exception:
        return False, None


def check_product_url_pattern(url: str) -> bool:
    """
    Check if a URL looks like a specific product page
    rather than a category or search results page.
    """
    if not url:
        return False

    # Exclude obvious non-product URLs
    exclude_patterns = [
        r"/search\?",
        r"/s\?",
        r"/category/",
        r"/browse/",
        r"/c/",
        r"\?q=",
        r"/collections/",
    ]
    for pattern in exclude_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False

    # Check for product indicators
    for pattern in _COMPILED_PRODUCT_PATTERNS:
        if pattern.search(url):
            return True

    # If URL path ends in a segment that looks like a product slug
    # (contains digits or is long enough to be a product name), give benefit of doubt.
    # Short single-word segments like "cricket" or "sports" are category pages.
    try:
        parsed = urlparse(url)
        path_segments = [s for s in parsed.path.split("/") if s]
        if len(path_segments) >= 3:
            last_segment = path_segments[-1]
            # Must contain at least one digit OR be a long hyphenated slug (>=4 words)
            has_digit = bool(re.search(r"\d", last_segment))
            is_long_slug = len(last_segment.split("-")) >= 4
            if has_digit or is_long_slug:
                if re.match(r"[a-z0-9][a-z0-9-]+[a-z0-9]$", last_segment, re.IGNORECASE):
                    return True
    except Exception:
        pass

    return False


def check_india_fit(url: str, snippet: str) -> tuple[bool, float]:
    """
    Check if a product is India-specific.

    Returns:
        (is_india_fit, country_score) where country_score is 0.0–1.0.
    """
    url_lower = url.lower()
    snippet_lower = snippet.lower() if snippet else ""

    # .in domain is strongest signal
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname.endswith(".in"):
            return True, 1.0
    except Exception:
        pass

    # India-specific domain patterns
    india_domains = ["amazon.in", "flipkart.com", "myntra.com", "nykaa.com",
                     "tatacliq.com", "meesho.com"]
    for domain in india_domains:
        if domain in url_lower:
            return True, 1.0

    # India in snippet
    india_keywords = ["india", "indian", "₹", "inr", "rupee", "rs.", "deliver to india",
                      "ships to india", "india delivery"]
    for keyword in india_keywords:
        if keyword in snippet_lower or keyword in url_lower:
            return True, 0.7

    return False, 0.0
