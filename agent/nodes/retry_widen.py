"""
agent/nodes/retry_widen.py

Retry branch: triggered when validate_products finds < 3 usable products.

Actions:
  1. Increment retry_count
  2. Widen budget by 10% (budget_max * 1.10)
  3. Add new queries based on weak signals with widened budget
  4. Clear raw_results and validated_products for fresh search

The graph conditional edge checks retry_count after this node:
  retry_count < 2 → route back to search_products
  retry_count == 2 → route to escalate
"""

from __future__ import annotations

from agent.state import GiftAgentState
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.retry_widen")

BUDGET_WIDEN_FACTOR = 1.10


def retry_widen(state: GiftAgentState) -> dict:
    """
    Widen search parameters for retry.
    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "retry_widen")

    retry_count = state.get("retry_count", 0) + 1
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})
    current_budget_max = state.get("widened_budget_max") or float(
        gift_context.get("budget_max", 5000)
    )

    # Widen budget
    widened_budget_max = round(current_budget_max * BUDGET_WIDEN_FACTOR, 2)

    logger.info(
        "Retry %d: widening budget from %.0f to %.0f",
        retry_count,
        current_budget_max,
        widened_budget_max,
    )

    # Generate new fallback queries with widened budget
    safe_signals = state.get("safe_signals", {})
    weak_signals = safe_signals.get("weak", [])
    role = contact.get("role", "professional")
    country = gift_context.get("country", "India")
    currency = gift_context.get("currency", "INR")

    new_queries = []

    # Add widened-budget versions of existing fallback/weak queries
    for signal in (weak_signals or [])[:2]:
        key_terms = signal.replace("may appreciate", "").replace(
            "interested in", ""
        ).strip()
        if key_terms:
            new_queries.append({
                "query": (
                    f"{key_terms} gift {country} "
                    f"under {int(widened_budget_max)} rupees"
                ),
                "type": "weak",
                "signal_used": signal,
            })

    # Always add a role-based fallback
    new_queries.append({
        "query": (
            f"premium gift for {role} {country} "
            f"under {int(widened_budget_max)} rupees amazon.in"
        ),
        "type": "fallback",
        "signal_used": "professional role",
    })

    # Merge with existing queries, avoiding exact duplicates
    existing_queries = state.get("queries", [])
    existing_texts = {q.get("query", "") for q in existing_queries}
    merged_queries = list(existing_queries)
    for q in new_queries:
        if q["query"] not in existing_texts:
            merged_queries.append(q)
            existing_texts.add(q["query"])

    logger.info("Retry %d queries: %d total", retry_count, len(merged_queries))

    logs = log_node_end(logs, "retry_widen")

    return {
        "retry_count": retry_count,
        "widened_budget_max": widened_budget_max,
        "queries": merged_queries,
        # Clear previous search results to force fresh search
        "raw_results": [],
        "validated_products": [],
        "logs": logs,
    }
