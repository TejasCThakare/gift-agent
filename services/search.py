"""
services/search.py

DuckDuckGo search wrapper for product discovery.

Uses the duckduckgo-search library (DDGS class) which requires no API key.
Implements exponential backoff with jitter for rate limit handling.
Returns structured RawProduct objects with source attribution.

Design notes:
- All URLs come from actual search results — never constructed by an LLM.
- Results include the query that found them (query_source) for traceability.
- Preferred domains are used to bias result selection toward Indian e-commerce.
- Max results per query is configurable via environment variable.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    before_sleep_log,
    RetryError,
)

from models.products import RawProduct

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "5"))
MAX_TOTAL_RESULTS = 15

PREFERRED_DOMAINS = [
    "amazon.in",
    "flipkart.com",
    "myntra.com",
    "nykaa.com",
    "tatacliq.com",
    "reliancedigital.in",
    "croma.com",
    "meesho.com",
    "snapdeal.com",
    "ajio.com",
    "firstcry.com",
    "bewakoof.com",
    "thesouledstore.com",
]


class SearchRateLimitError(Exception):
    pass


class SearchError(Exception):
    pass


def search_products(
    queries: list[dict],
    max_results_per_query: int = MAX_RESULTS_PER_QUERY,
) -> list[RawProduct]:
    all_results: list[RawProduct] = []
    seen_urls: set[str] = set()

    for query_dict in queries:
        query_text = query_dict.get("query", "")
        query_type = query_dict.get("type", "strong")

        if not query_text:
            continue

        logger.info("Searching: %s [%s]", query_text, query_type)

        try:
            results = _search_with_retry(
                query=query_text,
                max_results=max_results_per_query,
            )
        except RetryError as e:
            logger.warning("Search exhausted retries for query '%s': %s", query_text, e)
            results = []
        except SearchError as e:
            logger.warning("Search error for query '%s': %s", query_text, e)
            results = []

        for result in results:
            url = result.get("href", "") or result.get("url", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            all_results.append(
                RawProduct(
                    title=result.get("title", ""),
                    url=url,
                    snippet=result.get("body", "") or result.get("snippet", ""),
                    query_source=query_text,
                    query_type=query_type,
                )
            )

        if len(queries) > 1:
            time.sleep(random.uniform(1.0, 2.5))

    all_results = _prioritize_by_domain(all_results)
    return all_results[:MAX_TOTAL_RESULTS]


def _search_with_retry(query: str, max_results: int) -> list[dict]:
    @retry(
        retry=retry_if_exception_type(SearchRateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=15) + wait_random(0, 2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _do_search():
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch="moderate",
                ))
                return results
        except Exception as e:
            error_str = str(e).lower()
            if "ratelimit" in error_str or "429" in error_str or "blocked" in error_str:
                raise SearchRateLimitError(f"DuckDuckGo rate limited: {e}") from e
            elif "timeout" in error_str:
                raise SearchRateLimitError(f"DuckDuckGo timeout (will retry): {e}") from e
            else:
                raise SearchError(f"DuckDuckGo search error: {e}") from e

    return _do_search()


def _prioritize_by_domain(results: list[RawProduct]) -> list[RawProduct]:
    preferred = []
    others = []

    for result in results:
        url_lower = result.url.lower()
        is_preferred = any(domain in url_lower for domain in PREFERRED_DOMAINS)
        if is_preferred:
            preferred.append(result)
        else:
            others.append(result)

    return preferred + others