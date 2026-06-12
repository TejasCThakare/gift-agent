"""
tests/test_filter_signals.py

Unit tests for the filter_signals node.
Tests: blocklist enforcement, safe signal passthrough, borderline flagging.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.nodes.filter_signals import filter_signals, _classify_signal


class TestClassifySignal:
    """Unit tests for the _classify_signal function."""

    def test_safe_signal_passes(self):
        assert _classify_signal("interested in cricket") == "safe"
        assert _classify_signal("reads business strategy books") == "safe"
        assert _classify_signal("VP Sales leader") == "safe"

    def test_religion_blocked(self):
        assert _classify_signal("attends temple regularly") == "religion"
        assert _classify_signal("follows hindu traditions") == "religion"
        assert _classify_signal("goes to mosque on Fridays") == "religion"

    def test_health_blocked(self):
        assert _classify_signal("has diabetes") == "health"
        assert _classify_signal("following a diet") == "health"
        assert _classify_signal("in therapy") == "health"

    def test_politics_blocked(self):
        assert _classify_signal("supports BJP") == "politics"
        assert _classify_signal("interested in election results") == "politics"

    def test_gender_blocked(self):
        assert _classify_signal("wife works in tech") == "gender"
        assert _classify_signal("gift for his wife") == "gender"

    def test_family_blocked(self):
        assert _classify_signal("married with two children") == "family_status"
        assert _classify_signal("recently divorced") == "family_status"

    def test_borderline_flagged(self):
        result = _classify_signal("enjoys coffee")
        assert result == "borderline"

        result = _classify_signal("likes wine")
        assert result == "borderline"

    def test_case_insensitive(self):
        assert _classify_signal("FOLLOWS HINDU TRADITIONS") == "religion"
        assert _classify_signal("interested in CRICKET") == "safe"


class TestFilterSignals:
    """Integration tests for the filter_signals node."""

    def test_safe_signals_pass_through(self):
        state = {
            "strong_signals": ["interested in cricket", "reads business books"],
            "weak_signals": ["may appreciate leadership content"],
            "evidence_map": {
                "interested in cricket": ["Love cricket — follow every IPL match"],
                "reads business books": ["Just finished The Challenger Sale"],
                "may appreciate leadership content": ["Leading 40-person sales team"],
            },
            "logs": {},
        }
        result = filter_signals(state)

        assert "interested in cricket" in result["safe_signals"]["strong"]
        assert "reads business books" in result["safe_signals"]["strong"]
        assert "may appreciate leadership content" in result["safe_signals"]["weak"]

    def test_sensitive_signals_blocked(self):
        state = {
            "strong_signals": ["attends temple regularly", "interested in cricket"],
            "weak_signals": ["has diabetes"],
            "evidence_map": {},
            "logs": {},
        }
        result = filter_signals(state)

        # Cricket should pass
        assert "interested in cricket" in result["safe_signals"]["strong"]
        # Temple and diabetes should be dropped
        dropped_signals = [d["signal"] for d in result["safe_signals"]["dropped"]]
        assert "attends temple regularly" in dropped_signals
        assert "has diabetes" in dropped_signals

    def test_signals_to_avoid_populated(self):
        state = {
            "strong_signals": ["attends mosque"],
            "weak_signals": [],
            "evidence_map": {},
            "logs": {},
        }
        result = filter_signals(state)

        assert len(result["signals_to_avoid"]) >= 1
        # Should include the standard guardrail note
        avoid_texts = [
            s if isinstance(s, str) else s.get("signal", "")
            for s in result["signals_to_avoid"]
        ]
        assert any("sensitive" in str(s).lower() or "religion" in str(s).lower() for s in result["signals_to_avoid"])

    def test_borderline_signals_kept_but_flagged(self):
        state = {
            "strong_signals": ["enjoys coffee"],
            "weak_signals": [],
            "evidence_map": {},
            "logs": {},
        }
        result = filter_signals(state)

        # Coffee should be in borderline, not dropped
        assert "enjoys coffee" in result["safe_signals"]["borderline"]
        dropped = [d["signal"] for d in result["safe_signals"].get("dropped", [])]
        assert "enjoys coffee" not in dropped

    def test_empty_signals_handled(self):
        state = {
            "strong_signals": [],
            "weak_signals": [],
            "evidence_map": {},
            "logs": {},
        }
        result = filter_signals(state)

        assert result["safe_signals"]["strong"] == []
        assert result["safe_signals"]["weak"] == []
        assert len(result["signals_to_avoid"]) >= 1  # At least the standard note

    def test_evidence_map_cleaned_of_dropped_signals(self):
        state = {
            "strong_signals": ["interested in cricket", "attends temple"],
            "weak_signals": [],
            "evidence_map": {
                "interested in cricket": ["MI vs CSK match!"],
                "attends temple": ["Goes to temple every Sunday"],
            },
            "logs": {},
        }
        result = filter_signals(state)

        # Temple evidence should be removed from evidence_map
        assert "attends temple" not in result["evidence_map"]
        # Cricket evidence should remain
        assert "interested in cricket" in result["evidence_map"]
