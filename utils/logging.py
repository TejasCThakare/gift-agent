"""
utils/logging.py

Structured logging helpers for workflow observability.

Each workflow node calls log_node_start() and log_node_end() to record
timing, token usage, and any errors. These entries are accumulated in
state["logs"] and written to the artifact store at the end of the run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with the configured level."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def log_node_start(logs: dict, node_name: str) -> dict:
    """
    Record the start of a node execution.
    Returns the updated logs dict.
    """
    logs = dict(logs) if logs else {}
    logs[node_name] = {
        "start_time": time.time(),
        "end_time": None,
        "latency_ms": None,
        "tokens_used": 0,
        "llm_calls": 0,
        "errors": [],
    }
    return logs


def log_node_end(
    logs: dict,
    node_name: str,
    tokens_used: int = 0,
    llm_calls: int = 0,
    error: Optional[str] = None,
) -> dict:
    """
    Record the end of a node execution.
    Returns the updated logs dict.
    """
    logs = dict(logs) if logs else {}
    if node_name not in logs:
        logs[node_name] = {
            "start_time": None,
            "end_time": None,
            "latency_ms": None,
            "tokens_used": 0,
            "llm_calls": 0,
            "errors": [],
        }

    entry = dict(logs[node_name])
    end_time = time.time()
    entry["end_time"] = end_time

    start_time = entry.get("start_time")
    if start_time:
        entry["latency_ms"] = round((end_time - start_time) * 1000, 1)

    entry["tokens_used"] = tokens_used
    entry["llm_calls"] = llm_calls

    if error:
        errors = list(entry.get("errors", []))
        errors.append(error)
        entry["errors"] = errors

    logs[node_name] = entry
    return logs


def summarize_logs(logs: dict) -> dict:
    """
    Produce a summary of all node logs.
    Used in the final output for observability.
    """
    total_latency_ms = 0.0
    total_tokens = 0
    total_llm_calls = 0
    all_errors = []
    node_summaries = {}

    for node_name, entry in logs.items():
        if not isinstance(entry, dict):
            continue

        latency = entry.get("latency_ms") or 0
        tokens = entry.get("tokens_used") or 0
        llm_calls = entry.get("llm_calls") or 0
        errors = entry.get("errors") or []

        total_latency_ms += latency
        total_tokens += tokens
        total_llm_calls += llm_calls
        all_errors.extend(errors)

        node_summaries[node_name] = {
            "latency_ms": latency,
            "tokens_used": tokens,
            "llm_calls": llm_calls,
            "errors": errors,
        }

    return {
        "nodes": node_summaries,
        "totals": {
            "total_latency_ms": round(total_latency_ms, 1),
            "total_tokens_used": total_tokens,
            "total_llm_calls": total_llm_calls,
            "total_errors": len(all_errors),
            "errors": all_errors,
        },
    }
