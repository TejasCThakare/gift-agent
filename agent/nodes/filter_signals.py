"""
agent/nodes/filter_signals.py

Node 3: filter_signals

Pure Python safety filter — no LLM involved.

Checks each signal string against a blocklist of sensitive category
keywords. Signals touching blocked categories are removed and logged
in signals_to_avoid with the category and reason.

Borderline signals (ambiguous but not clearly blocked) are kept but
flagged in safe_signals.borderline for reviewer awareness.

This is a rule-based system by design — it's deterministic, auditable,
and cannot be tricked by LLM prompt injection.
"""

from __future__ import annotations

from models.signals import SafeSignals, FilteredSignal
from agent.state import GiftAgentState
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.filter_signals")


# ── BLOCKLIST ─────────────────────────────────────────────────────────────────
# Maps category name to lists of keywords that indicate a signal touches
# that sensitive category. Matching is case-insensitive substring search.

BLOCKLIST: dict[str, list[str]] = {
    "religion": [
        "temple", "mosque", "church", "gurudwara", "mandir", "masjid",
        "hindu", "muslim", "christian", "sikh", "jain", "buddhist", "parsi",
        "jewish", "catholic", "protestant", "religion", "religious", "faith",
        "prayer", "puja", "namaz", "sabbath", "halal", "kosher", "vegetarian",
        "vegan", "fasting", "festival", "diwali", "eid", "christmas", "navratri",
        "ram", "krishna", "allah", "god", "deity", "worship",
    ],
    "health": [
        "cancer", "diabetes", "heart disease", "blood pressure", "hypertension",
        "obesity", "overweight", "diet", "weight loss", "weight gain", "therapy",
        "medication", "prescription", "hospital", "doctor", "surgery", "illness",
        "sick", "disease", "disorder", "mental health", "depression", "anxiety",
        "stress management", "wellness", "rehab", "recovery", "injury", "chronic",
        "disability", "allergy", "intolerance",
    ],
    "politics": [
        "bjp", "congress", "aap", "shiv sena", "bsp", "ncp", "election",
        "vote", "voting", "political party", "politician", "prime minister",
        "chief minister", "parliament", "democracy", "liberal", "conservative",
        "left wing", "right wing", "protest", "rally", "manifesto",
    ],
    "ethnicity": [
        "caste", "brahmin", "dalit", "kshatriya", "vaishya", "shudra",
        "obc", "sc", "st", "tribe", "tribal", "ethnic", "race", "racial",
        "community", "ancestry", "origin", "heritage", "marathi", "tamil",
        "bengali", "gujarati", "punjabi", "bihari", "based on background",
    ],
    "gender": [
        "wife", "husband", "girlfriend", "boyfriend", "partner", "spouse",
        "feminine", "masculine", "gender", "female", "male", "trans",
        "non-binary", "she prefers", "he prefers", "her style", "his style",
        "for a woman", "for a man",
    ],
    "family_status": [
        "married", "divorced", "single", "widowed", "children", "kids",
        "son", "daughter", "parent", "mother", "father", "pregnant",
        "expecting", "newborn", "family man", "family woman", "dad", "mom",
        "grandparent", "grandmother", "grandfather",
    ],
}

# Borderline keywords — keep but flag for reviewer awareness
BORDERLINE_KEYWORDS: list[str] = [
    "vegetarian",  # food preference — borderline personal
    "fitness",     # could be health-adjacent
    "runner",      # fitness — borderline
    "workout",     # fitness — borderline
    "coffee",      # consumable — send with caution
    "tea",         # consumable — send with caution
    "alcohol",     # consumable — flag for professional context
    "wine",        # consumable — flag
    "beer",        # consumable — flag
    "whiskey",     # consumable — flag
    "liquor",      # consumable — flag
]


def filter_signals(state: GiftAgentState) -> dict:
    """
    Filter signals through the safety blocklist.
    Returns updated safe_signals and signals_to_avoid.
    """
    logs = log_node_start(state.get("logs", {}), "filter_signals")

    strong_signals = state.get("strong_signals", [])
    weak_signals = state.get("weak_signals", [])
    evidence_map = state.get("evidence_map", {})

    safe_strong: list[str] = []
    safe_weak: list[str] = []
    borderline: list[str] = []
    dropped: list[FilteredSignal] = []

    for signal in strong_signals:
        result = _classify_signal(signal)
        if result == "safe":
            safe_strong.append(signal)
        elif result == "borderline":
            borderline.append(signal)
            logger.info("Borderline signal (kept, flagged): %s", signal)
        else:
            # result is the category name
            fs = FilteredSignal(
                signal=signal,
                reason=f"Signal touches sensitive category: {result}",
                category=result,
            )
            dropped.append(fs)
            logger.info("Dropped signal [%s]: %s", result, signal)

    for signal in weak_signals:
        result = _classify_signal(signal)
        if result == "safe":
            safe_weak.append(signal)
        elif result == "borderline":
            borderline.append(signal)
        else:
            fs = FilteredSignal(
                signal=signal,
                reason=f"Signal touches sensitive category: {result}",
                category=result,
            )
            dropped.append(fs)

    # Also check evidence_map values for sensitive content
    # (in case the LLM injected sensitive quotes)
    clean_evidence_map = _filter_evidence_map(evidence_map, dropped)

    safe_signals_model = SafeSignals(
        strong=safe_strong,
        weak=safe_weak,
        borderline=borderline,
        dropped=dropped,
    )

    # Build signals_to_avoid in assignment output schema format
    signals_to_avoid = safe_signals_model.to_signals_to_avoid_schema()

    logger.info(
        "Filter: %d strong safe, %d weak safe, %d borderline, %d dropped",
        len(safe_strong),
        len(safe_weak),
        len(borderline),
        len(dropped),
    )

    logs = log_node_end(logs, "filter_signals")

    return {
        "safe_signals": safe_signals_model.model_dump(),
        "signals_to_avoid": signals_to_avoid,
        "evidence_map": clean_evidence_map,
        "logs": logs,
    }


def _classify_signal(signal: str) -> str:
    """
    Classify a signal as "safe", "borderline", or a category name (unsafe).

    Uses word-boundary matching for short keywords (<=3 chars) to prevent
    substring false positives (e.g. "st" in "interested" triggering caste block).
    Multi-word phrases and longer keywords use substring matching.

    Returns:
        "safe" — signal is clean
        "borderline" — signal needs reviewer awareness
        "<category>" — signal is blocked (category name)
    """
    import re as _re
    signal_lower = signal.lower()

    def _word_match(keyword: str, text: str) -> bool:
        if " " in keyword:
            return keyword in text
        if len(keyword) <= 3:
            return bool(_re.search(r'\b' + _re.escape(keyword) + r'\b', text))
        return keyword in text

    # Check blocklist first (unsafe)
    for category, keywords in BLOCKLIST.items():
        for keyword in keywords:
            if _word_match(keyword, signal_lower):
                return category

    # Check borderline
    for keyword in BORDERLINE_KEYWORDS:
        if _word_match(keyword, signal_lower):
            return "borderline"

    return "safe"


def _filter_evidence_map(
    evidence_map: dict,
    dropped: list[FilteredSignal],
) -> dict:
    """
    Remove entries from evidence_map that correspond to dropped signals.
    Also check evidence values for sensitive content.
    """
    dropped_signals = {fs.signal.lower() for fs in dropped}
    clean_map = {}

    for signal, quotes in evidence_map.items():
        if signal.lower() in dropped_signals:
            continue  # Remove dropped signal's evidence

        # Also check if the signal text itself is sensitive
        if _classify_signal(signal) not in ("safe", "borderline"):
            continue

        clean_map[signal] = quotes

    return clean_map
