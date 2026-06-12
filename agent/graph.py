"""
agent/graph.py

LangGraph StateGraph definition for the Gift Agent workflow.

Registers all nodes, conditional edges, retry/escalation branches,
and the human-review interrupt with MemorySaver checkpointing.

Graph topology:
  ingest
    → extract_and_query
    → filter_signals
    → search_products
    → validate_products
    → [conditional: enough_products?]
        yes → score_products
        no  → retry_widen
                → [conditional: retry_exhausted?]
                    no  → search_products (loop)
                    yes → escalate → score_products
    → rank_gifts
    → generate_messages
    → human_review (interrupt)
    → [conditional: review_action?]
        approve    → done
        edit       → done
        reject     → rank_gifts (re-rank with history)
        regenerate → score_products (re-score + re-rank)

State persistence:
  MemorySaver checkpoints state at every node boundary.
  This enables resume after interrupt() without re-running prior nodes.
"""

from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import GiftAgentState
from agent.nodes.ingest import ingest
from agent.nodes.extract_and_query import extract_and_query
from agent.nodes.filter_signals import filter_signals
from agent.nodes.search_products import search_products
from agent.nodes.validate_products import validate_products
from agent.nodes.score_products import score_products
from agent.nodes.rank_gifts import rank_gifts
from agent.nodes.generate_messages import generate_messages
from agent.nodes.human_review import human_review
from agent.nodes.retry_widen import retry_widen
from agent.nodes.escalate import escalate
from storage.artifact_store import save_state, save_recommendations
from utils.logging import get_logger, log_node_start, log_node_end, summarize_logs

logger = get_logger("agent.graph")

# Minimum number of usable products required before scoring
MIN_PRODUCTS_REQUIRED = int(os.getenv("MIN_PRODUCTS_REQUIRED", "3"))
MAX_RETRIES = 2


# ── CONDITIONAL EDGE FUNCTIONS ────────────────────────────────────────────────

def route_after_validation(
    state: GiftAgentState,
) -> Literal["score_products", "retry_widen"]:
    """
    After validate_products: check if we have enough usable products.
    Usable = validation_tier in ("pass", "partial")
    """
    validated = state.get("validated_products", [])
    usable = [
        p for p in validated
        if p.get("validation_tier") in ("pass", "partial")
    ]
    if len(usable) >= MIN_PRODUCTS_REQUIRED:
        return "score_products"
    return "retry_widen"


def route_after_retry(
    state: GiftAgentState,
) -> Literal["search_products", "escalate"]:
    """
    After retry_widen: check if max retries exceeded.
    """
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return "escalate"
    return "search_products"


def route_after_review(
    state: GiftAgentState,
) -> Literal["done", "rank_gifts", "score_products"]:
    """
    After human_review: route based on review_action.

    approve    → done
    edit       → done (edits already applied in human_review node)
    reject     → rank_gifts (re-rank with rejection in history)
    regenerate → score_products (re-score + re-rank with notes)
    """
    action = state.get("review_action", "approve")
    if action in ("approve", "edit"):
        return "done"
    elif action == "reject":
        return "rank_gifts"
    elif action == "regenerate":
        return "score_products"
    else:
        logger.warning("Unknown review_action '%s' — defaulting to done", action)
        return "done"


# ── DONE NODE ─────────────────────────────────────────────────────────────────

def done(state: GiftAgentState) -> dict:
    """
    Terminal node: write final output, update review status, persist artifacts.
    """
    logs = log_node_start(state.get("logs", {}), "done")

    thread_id = state.get("thread_id", "")
    final_recommendations = dict(state.get("final_recommendations", {}))
    review_action = state.get("review_action", "approve")

    # Update human_review status in final output
    status_map = {
        "approve": "approved",
        "edit": "edited",
        "reject": "rejected",
        "regenerate": "regenerated",
    }
    final_recommendations["human_review"] = {
        "status": status_map.get(review_action, "approved"),
        "available_actions": [],
        "review_history_count": len(state.get("review_history", [])),
    }

    # Attach review history to output for full audit trail
    final_recommendations["review_history"] = state.get("review_history", [])

    # Compute log summary
    logs = log_node_end(logs, "done")
    log_summary = summarize_logs(logs)
    final_recommendations["workflow_logs"] = log_summary

    logger.info(
        "Done: thread=%s | action=%s | gifts=%d | escalated=%s",
        thread_id,
        review_action,
        len(final_recommendations.get("recommended_gifts", [])),
        final_recommendations.get("escalation_flag", False),
    )

    # Persist final artifacts
    if thread_id:
        try:
            save_recommendations(thread_id, final_recommendations)
            save_state(thread_id, {
                k: v for k, v in state.items()
                if k != "logs"  # logs saved separately
            })
        except Exception as e:
            logger.error("Failed to persist artifacts: %s", e)

    return {
        "final_recommendations": final_recommendations,
        "logs": logs,
    }


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Returns a compiled graph ready to invoke.
    Uses MemorySaver for state persistence across interrupt/resume cycles.
    """
    builder = StateGraph(GiftAgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("ingest", ingest)
    builder.add_node("extract_and_query", extract_and_query)
    builder.add_node("filter_signals", filter_signals)
    builder.add_node("search_products", search_products)
    builder.add_node("validate_products", validate_products)
    builder.add_node("score_products", score_products)
    builder.add_node("rank_gifts", rank_gifts)
    builder.add_node("generate_messages", generate_messages)
    builder.add_node("human_review", human_review)
    builder.add_node("retry_widen", retry_widen)
    builder.add_node("escalate", escalate)
    builder.add_node("done", done)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("ingest")

    # ── Main flow (linear) ────────────────────────────────────────────────
    builder.add_edge("ingest", "extract_and_query")
    builder.add_edge("extract_and_query", "filter_signals")
    builder.add_edge("filter_signals", "search_products")
    builder.add_edge("search_products", "validate_products")

    # ── Conditional: enough products? ─────────────────────────────────────
    builder.add_conditional_edges(
        "validate_products",
        route_after_validation,
        {
            "score_products": "score_products",
            "retry_widen": "retry_widen",
        },
    )

    # ── Retry branch ──────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "retry_widen",
        route_after_retry,
        {
            "search_products": "search_products",
            "escalate": "escalate",
        },
    )

    # ── Escalation merges into score_products ─────────────────────────────
    builder.add_edge("escalate", "score_products")

    # ── Scoring → ranking → messages → review ────────────────────────────
    builder.add_edge("score_products", "rank_gifts")
    builder.add_edge("rank_gifts", "generate_messages")
    builder.add_edge("generate_messages", "human_review")

    # ── Human review routing ──────────────────────────────────────────────
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "done": "done",
            "rank_gifts": "rank_gifts",        # reject
            "score_products": "score_products", # regenerate
        },
    )

    # ── Terminal ──────────────────────────────────────────────────────────
    builder.add_edge("done", END)

    # ── Compile with MemorySaver ──────────────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("LangGraph compiled successfully")
    return graph


# ── Module-level graph singleton ──────────────────────────────────────────────
# Build once at import time. Re-used across all API requests.
_graph = None


def get_graph() -> StateGraph:
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
