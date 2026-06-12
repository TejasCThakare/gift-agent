"""
storage/artifact_store.py

JSON artifact persistence for workflow runs.

Writes and reads structured JSON files for:
  - Full workflow state snapshots
  - Review history
  - Final recommendations
  - Escalation events
  - Validation reports
  - Run summaries (latency, tokens, LLM calls)

Files are organized under ARTIFACTS_DIR/{thread_id}/:
  state.json            — complete workflow state at end of run
  recommendations.json  — final output matching assignment schema
  review_history.json   — all review actions taken
  trace.json            — intermediate outputs (signals, queries, products)
  logs.json             — timing and token usage per node

All writes are atomic (write to temp file, then rename).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


def _ensure_thread_dir(thread_id: str) -> Path:
    """Create and return the directory for a given thread."""
    thread_dir = ARTIFACTS_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    return thread_dir


def _write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data to a file."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp_path.rename(path)


def _read_json(path: Path) -> Any:
    """Read JSON data from a file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(thread_id: str, state: dict) -> None:
    """Save the complete workflow state."""
    thread_dir = _ensure_thread_dir(thread_id)
    _write_json(thread_dir / "state.json", state)


def save_recommendations(thread_id: str, recommendations: dict) -> None:
    """Save the final recommendations in assignment output schema format."""
    thread_dir = _ensure_thread_dir(thread_id)
    _write_json(thread_dir / "recommendations.json", recommendations)


def save_review_history(thread_id: str, review_history: list) -> None:
    """Save the complete review history."""
    thread_dir = _ensure_thread_dir(thread_id)
    _write_json(thread_dir / "review_history.json", {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "entries": review_history,
    })


def save_trace(thread_id: str, state: dict) -> None:
    """
    Save intermediate outputs (trace) for debugging and auditability.
    Includes: evidence_map, signals, safe_signals, queries,
              raw_results, validated_products, scored_products.
    """
    thread_dir = _ensure_thread_dir(thread_id)
    trace = {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "contact_name": state.get("contact", {}).get("name", ""),
        "evidence_map": state.get("evidence_map", {}),
        "strong_signals": state.get("strong_signals", []),
        "weak_signals": state.get("weak_signals", []),
        "safe_signals": state.get("safe_signals", {}),
        "signals_to_avoid": state.get("signals_to_avoid", []),
        "queries": state.get("queries", []),
        "raw_results_count": len(state.get("raw_results", [])),
        "raw_results": state.get("raw_results", []),
        "validated_products": state.get("validated_products", []),
        "scored_products": state.get("scored_products", []),
        "retry_count": state.get("retry_count", 0),
        "escalation_flag": state.get("escalation_flag", False),
        "escalation_notes": state.get("escalation_notes", ""),
    }
    _write_json(thread_dir / "trace.json", trace)


def save_logs(thread_id: str, logs: dict) -> None:
    """Save the node execution logs."""
    thread_dir = _ensure_thread_dir(thread_id)
    _write_json(thread_dir / "logs.json", {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "nodes": logs,
    })


def load_state(thread_id: str) -> dict | None:
    """Load the workflow state for a given thread."""
    thread_dir = ARTIFACTS_DIR / thread_id
    return _read_json(thread_dir / "state.json")


def load_recommendations(thread_id: str) -> dict | None:
    """Load final recommendations for a given thread."""
    thread_dir = ARTIFACTS_DIR / thread_id
    return _read_json(thread_dir / "recommendations.json")


def load_trace(thread_id: str) -> dict | None:
    """Load the trace for a given thread."""
    thread_dir = ARTIFACTS_DIR / thread_id
    return _read_json(thread_dir / "trace.json")


def list_runs() -> list[dict]:
    """
    List all completed runs with summary info.
    Returns a list of {thread_id, contact_name, created_at, status}.
    """
    if not ARTIFACTS_DIR.exists():
        return []

    runs = []
    for thread_dir in sorted(ARTIFACTS_DIR.iterdir(), reverse=True):
        if not thread_dir.is_dir():
            continue
        thread_id = thread_dir.name

        recommendations = _read_json(thread_dir / "recommendations.json")
        contact_name = ""
        status = "unknown"
        created_at = ""

        if recommendations:
            contact_name = recommendations.get("contact_name", "")
            review = recommendations.get("human_review", {})
            status = review.get("status", "pending_review")
            created_at = recommendations.get("generated_at", "")

        runs.append({
            "thread_id": thread_id,
            "contact_name": contact_name,
            "status": status,
            "created_at": created_at,
        })

    return runs
