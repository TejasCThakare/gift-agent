"""
agent/nodes/search_products.py

Node 4: search_products

Uses the deterministic signal_to_product mapping layer to translate
signals into concrete product queries before searching.
"""

from __future__ import annotations

from agent.state import GiftAgentState
from services.search import search_products as _search
from services.signal_to_product import get_product_queries
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.search_products")


def search_products(state: GiftAgentState) -> dict:
    logs = log_node_start(state.get("logs", {}), "search_products")

    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})
    safe_signals = state.get("safe_signals", {})
    retry_count = state.get("retry_count", 0)
    widened_budget_max = state.get("widened_budget_max", None)

    budget_max = widened_budget_max or float(gift_context.get("budget_max", 5000))
    role = contact.get("role", "")
    occasion = gift_context.get("occasion", "")
    relationship_type = state.get("relationship_type", "unknown")

    strong_signals = safe_signals.get("strong", [])
    weak_signals = safe_signals.get("weak", []) + safe_signals.get("borderline", [])

    product_queries = get_product_queries(
        strong_signals=strong_signals,
        weak_signals=weak_signals,
        role=role,
        occasion=occasion,
        budget_max=budget_max,
        relationship_type=relationship_type,
    )

    if retry_count > 0:
        active_queries = [
            q for q in product_queries
            if q.get("type") in ("weak", "fallback")
        ]
        if not active_queries:
            active_queries = product_queries
        logger.info(
            "Retry %d: using %d weak/fallback queries (budget: %.0f)",
            retry_count, len(active_queries), budget_max,
        )
    else:
        active_queries = product_queries

    logger.info("Searching with %d product queries", len(active_queries))
    for q in active_queries[:5]:
        logger.info("  Query [%s]: %s", q.get("type"), q.get("query", "")[:80])

    raw_results = _search(queries=active_queries)

    logger.info("Found %d raw results after deduplication", len(raw_results))

    logs = log_node_end(logs, "search_products")

    return {
        "queries": active_queries,
        "raw_results": [r.model_dump() for r in raw_results],
        "logs": logs,
    }