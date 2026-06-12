"""
agent/nodes/escalate.py

Escalation branch: triggered after retry_count == 2 and still < 3 usable products.

Actions:
  1. Take the best available products regardless of validation tier
     (including "fail" tier as last resort)
  2. Set escalation_flag = True
  3. Write detailed escalation_notes explaining what failed and why
  4. Mark missing recommendation slots explicitly
  5. Set all escalated products' confidence to "low" indication in notes
     (actual confidence still computed by score_products — this is metadata)

The workflow continues to score_products with escalated products.
Human review will see the escalation flag and notes prominently.
"""

from __future__ import annotations

from agent.state import GiftAgentState
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.escalate")


def escalate(state: GiftAgentState) -> dict:
    """
    Handle escalation when insufficient products found after retries.
    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "escalate")

    validated_products = state.get("validated_products", [])
    raw_results = state.get("raw_results", [])
    queries = state.get("queries", [])
    retry_count = state.get("retry_count", 0)
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})

    # Count usable products (pass + partial)
    usable = [
        p for p in validated_products
        if p.get("validation_tier") in ("pass", "partial")
    ]
    fail_only = [
        p for p in validated_products
        if p.get("validation_tier") == "fail"
    ]

    logger.warning(
        "ESCALATION: %d retries exhausted. %d usable products, %d fail-only",
        retry_count,
        len(usable),
        len(fail_only),
    )

    # Build escalation notes
    notes_parts = [
        f"Escalation triggered after {retry_count} retry attempts.",
        f"Found {len(usable)} usable product(s) out of {len(validated_products)} validated ({len(raw_results)} raw results).",
        f"Used {len(queries)} search queries.",
        "",
    ]

    if len(usable) < 3:
        missing = 3 - len(usable)
        notes_parts.append(f"Missing {missing} recommendation slot(s).")

    # Diagnose what failed
    failure_reasons = _diagnose_failures(validated_products)
    if failure_reasons:
        notes_parts.append("Common validation failures:")
        for reason, count in failure_reasons.items():
            notes_parts.append(f"  - {reason}: {count} product(s)")

    notes_parts.append("")
    notes_parts.append(
        "Recommendations with low confidence require human review before use. "
        "Consider providing alternative product URLs or adjusting the budget."
    )

    escalation_notes = "\n".join(notes_parts)

    # If we have fewer than 3 usable products, supplement with best fail-only products
    # (ordered by how many rules they passed)
    all_products_for_scoring = list(usable)
    if len(all_products_for_scoring) < 3 and fail_only:
        fail_only_sorted = sorted(
            fail_only,
            key=lambda p: p.get("validation_score", 0.0),
            reverse=True,
        )
        # Take enough fail-only products to reach 3 total (or all of them)
        needed = 3 - len(all_products_for_scoring)
        all_products_for_scoring.extend(fail_only_sorted[:needed])
        logger.warning(
            "Using %d fail-tier products to supplement recommendations",
            min(needed, len(fail_only)),
        )

    logs = log_node_end(logs, "escalate")

    return {
        "escalation_flag": True,
        "escalation_notes": escalation_notes,
        "validated_products": all_products_for_scoring,
        "logs": logs,
    }


def _diagnose_failures(validated_products: list[dict]) -> dict[str, int]:
    """
    Count how many products failed each validation rule.
    Returns a dict of {failure_reason: count}.
    """
    failure_counts: dict[str, int] = {}

    for product in validated_products:
        report = product.get("validation_report", {})
        for rule_name, rule_data in report.items():
            if isinstance(rule_data, dict):
                passed = rule_data.get("passed", True)
            else:
                passed = getattr(rule_data, "passed", True)
            if not passed:
                readable = rule_name.replace("_", " ").title()
                failure_counts[readable] = failure_counts.get(readable, 0) + 1

    # Sort by frequency
    return dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True))
