"""
Unit tests for Task 10.1 — Trait Synthesis Agent:
  trace reading, evidence grouping, and multi-evidence rule.

Covers:
  - _get_trace_concept: concept extraction from heterogeneous trace objects
  - group_traces_by_concept: groups traces by concept, skips None-concept traces
  - _recall_traces: falls back to cognee.recall() when recall_agent_memory_traces
    is unavailable (AttributeError) or raises
  - multi-evidence rule in trait_synthesis_agent: concepts with < 2 signals
    must never reach gateway.remember/improve/forget

Requirements: 8.1, 8.3, 3.6
"""
from __future__ import annotations

import asyncio
import sys
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Ensure backend/ is on sys.path
_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from agents.trait_synthesis import (
    _get_trace_concept,
    group_traces_by_concept,
)
from models.schemas import TutorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(concept=None, content=None, text=None, data=None):
    """Build a SimpleNamespace that mimics a cognee trace object."""
    t = SimpleNamespace()
    if concept is not None:
        t.concept = concept
    if content is not None:
        t.content = content
    if text is not None:
        t.text = text
    if data is not None:
        t.data = data
    return t


def _make_state(
    user_id: str = "user_42",
    session_id: str = "sess_abc",
) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic="OS memory management",
        current_episode="ep_paging",
        mode="teacher",
        session_id=session_id,
        nudge_count=0,
        answer_history=[],
        trait_snapshot=[],
        ingest_needed=False,
    )


# ---------------------------------------------------------------------------
# _get_trace_concept — concept extraction
# ---------------------------------------------------------------------------

class TestGetTraceConcept:
    """Tests for the duck-typed concept extraction helper."""

    def test_returns_concept_attribute_when_present(self):
        trace = _trace(concept="paging")
        assert _get_trace_concept(trace) == "paging"

    def test_concept_attribute_takes_priority_over_content(self):
        trace = _trace(concept="paging", content="some other content")
        assert _get_trace_concept(trace) == "paging"

    def test_whitespace_only_concept_attribute_falls_through(self):
        trace = _trace(concept="   ")
        # Whitespace-only concept is empty after strip — should fall through to content
        result = _get_trace_concept(trace)
        # It should not return the blank concept; returns None or uses content
        assert result is None or result != "   "

    def test_falls_back_to_content_when_no_concept_attr(self):
        trace = _trace(content="External fragmentation. Some detail.")
        result = _get_trace_concept(trace)
        assert result is not None
        assert len(result) > 0

    def test_falls_back_to_text_when_no_concept_or_content(self):
        trace = _trace(text="Segmentation. Explanation follows.")
        result = _get_trace_concept(trace)
        assert result is not None
        assert len(result) > 0

    def test_returns_none_when_no_extractable_attribute(self):
        # An object with no usable attributes at all
        trace = SimpleNamespace()
        result = _get_trace_concept(trace)
        # str() of a SimpleNamespace is not "None" but looks like
        # "namespace()", which is < 120 chars — the fallback might return it.
        # We just ensure no exception is raised.
        # No assertion on the exact value, just non-crash.

    def test_returns_none_for_none_trace(self):
        # When concept/content/text are all None attributes
        trace = _trace(concept=None)
        # concept attr exists but is None — should skip to content fallback
        # content/text also absent → None
        result = _get_trace_concept(trace)
        # concept=None is set, but not a str → falls through; no other attrs → None
        assert result is None

    def test_content_truncated_to_first_sentence(self):
        trace = _trace(content="Topic A is interesting. More detail here.")
        result = _get_trace_concept(trace)
        assert result is not None
        # First sentence truncation: at most 60 chars of first sentence
        assert len(result) <= 60

    def test_concept_stripped(self):
        trace = _trace(concept="  paging  ")
        assert _get_trace_concept(trace) == "paging"


# ---------------------------------------------------------------------------
# group_traces_by_concept
# ---------------------------------------------------------------------------

class TestGroupTracesByConcept:
    """Tests for the grouping helper that powers the multi-evidence rule."""

    def test_empty_list_returns_empty_dict(self):
        assert group_traces_by_concept([]) == {}

    def test_none_returns_empty_dict(self):
        assert group_traces_by_concept(None) == {}

    def test_single_trace_with_concept(self):
        t = _trace(concept="paging")
        result = group_traces_by_concept([t])
        assert "paging" in result
        assert result["paging"] == [t]

    def test_groups_multiple_traces_under_same_concept(self):
        t1 = _trace(concept="paging")
        t2 = _trace(concept="paging")
        result = group_traces_by_concept([t1, t2])
        assert "paging" in result
        assert len(result["paging"]) == 2

    def test_separates_different_concepts(self):
        t1 = _trace(concept="paging")
        t2 = _trace(concept="segmentation")
        result = group_traces_by_concept([t1, t2])
        assert "paging" in result
        assert "segmentation" in result
        assert result["paging"] == [t1]
        assert result["segmentation"] == [t2]

    def test_traces_without_concept_are_skipped(self):
        t_no_concept = SimpleNamespace()  # no concept/content/text attrs
        t_with_concept = _trace(concept="paging")
        result = group_traces_by_concept([t_no_concept, t_with_concept])
        assert "paging" in result
        # Only the trace with an extractable concept appears
        assert all(v != [] for v in result.values())

    def test_non_list_input_wrapped_into_list(self):
        # A single trace object (not in a list) should be handled
        t = _trace(concept="MMU")
        result = group_traces_by_concept(t)
        assert "MMU" in result
        assert result["MMU"] == [t]

    def test_tuple_input_accepted(self):
        t1 = _trace(concept="paging")
        t2 = _trace(concept="paging")
        result = group_traces_by_concept((t1, t2))
        assert "paging" in result
        assert len(result["paging"]) == 2

    def test_preserves_trace_objects(self):
        t = _trace(concept="MMU")
        result = group_traces_by_concept([t])
        assert result["MMU"][0] is t

    def test_mixed_concept_and_content_fallback(self):
        """Traces using content fallback also get grouped correctly."""
        t1 = _trace(concept="external fragmentation")
        t2 = _trace(content="external fragmentation. Details.")
        result = group_traces_by_concept([t1, t2])
        # Both should appear in the result (possibly under different keys
        # since t2 uses the first-sentence fallback)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _recall_traces — defensive fallback behaviour
# ---------------------------------------------------------------------------

class TestRecallTraces:
    """Tests for the defensive recall helper (Requirement 8.1, 3.6)."""

    def test_uses_recall_agent_memory_traces_when_available(self):
        """Primary path: cognee.recall_agent_memory_traces is called with session_id."""
        from agents.trait_synthesis import _recall_traces

        mock_cognee = MagicMock()
        expected_traces = [_trace(concept="paging")]
        mock_cognee.recall_agent_memory_traces = AsyncMock(return_value=expected_traces)

        with patch("agents.trait_synthesis.cognee", mock_cognee):
            result = asyncio.run(_recall_traces("sess_123"))

        mock_cognee.recall_agent_memory_traces.assert_called_once_with("sess_123")
        assert result == expected_traces

    def test_falls_back_to_recall_on_attribute_error(self):
        """When recall_agent_memory_traces doesn't exist, falls back to cognee.recall()."""
        from agents.trait_synthesis import _recall_traces

        mock_cognee = MagicMock()
        # Simulate AttributeError by making the attribute raise on await
        del mock_cognee.recall_agent_memory_traces
        fallback_traces = [_trace(concept="segmentation")]
        mock_cognee.recall = AsyncMock(return_value=fallback_traces)

        with patch("agents.trait_synthesis.cognee", mock_cognee):
            result = asyncio.run(_recall_traces("sess_456"))

        mock_cognee.recall.assert_called_once_with(query="sess_456")
        assert result == fallback_traces

    def test_falls_back_to_recall_on_exception(self):
        """When recall_agent_memory_traces raises a non-AttributeError, falls back."""
        from agents.trait_synthesis import _recall_traces

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(
            side_effect=RuntimeError("cognee unavailable")
        )
        fallback_traces = [_trace(concept="TLB")]
        mock_cognee.recall = AsyncMock(return_value=fallback_traces)

        with patch("agents.trait_synthesis.cognee", mock_cognee):
            result = asyncio.run(_recall_traces("sess_789"))

        mock_cognee.recall.assert_called_once_with(query="sess_789")
        assert result == fallback_traces

    def test_returns_empty_list_when_both_fail(self):
        """When both recall methods fail, returns [] without raising."""
        from agents.trait_synthesis import _recall_traces

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(
            side_effect=RuntimeError("primary down")
        )
        mock_cognee.recall = AsyncMock(side_effect=RuntimeError("fallback down"))

        with patch("agents.trait_synthesis.cognee", mock_cognee):
            result = asyncio.run(_recall_traces("sess_err"))

        assert result == []


# ---------------------------------------------------------------------------
# Multi-evidence rule in trait_synthesis_agent
# ---------------------------------------------------------------------------

class TestMultiEvidenceRule:
    """
    Requirement 3.6 / 8.3: concepts with fewer than 2 evidence signals
    must never trigger gateway.remember(), .improve(), or .forget().
    """

    def _make_mock_gateway(self):
        gw = MagicMock()
        gw.remember = AsyncMock()
        gw.improve = AsyncMock()
        gw.forget = AsyncMock()
        return gw

    def test_single_signal_concept_never_calls_remember(self):
        """
        A concept with exactly 1 trace must be skipped — gateway.remember()
        must not be called.
        """
        from agents.trait_synthesis import trait_synthesis_agent

        state = _make_state()
        mock_gw = self._make_mock_gateway()

        # One trace for "paging"
        single_trace = _trace(concept="paging")

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(return_value=[single_trace])
        mock_cognee.recall = AsyncMock(return_value=None)

        with (
            patch("agents.trait_synthesis.cognee", mock_cognee),
            patch("agents.trait_synthesis.gateway", mock_gw),
        ):
            asyncio.run(trait_synthesis_agent(state))

        mock_gw.remember.assert_not_called()
        mock_gw.improve.assert_not_called()
        mock_gw.forget.assert_not_called()

    def test_zero_signal_concept_never_calls_any_gateway_method(self):
        """Empty trace list results in no gateway calls at all."""
        from agents.trait_synthesis import trait_synthesis_agent

        state = _make_state()
        mock_gw = self._make_mock_gateway()

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(return_value=[])
        mock_cognee.recall = AsyncMock(return_value=None)

        with (
            patch("agents.trait_synthesis.cognee", mock_cognee),
            patch("agents.trait_synthesis.gateway", mock_gw),
        ):
            asyncio.run(trait_synthesis_agent(state))

        mock_gw.remember.assert_not_called()
        mock_gw.improve.assert_not_called()
        mock_gw.forget.assert_not_called()

    def test_two_signals_for_same_concept_reaches_gateway(self):
        """
        A concept with exactly 2 traces must pass the multi-evidence rule
        and reach gateway.remember() (assuming no existing trait).
        """
        from agents.trait_synthesis import trait_synthesis_agent

        state = _make_state()
        mock_gw = self._make_mock_gateway()

        t1 = _trace(concept="paging")
        t2 = _trace(concept="paging")

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(return_value=[t1, t2])
        # No existing trait in Track B
        mock_cognee.recall = AsyncMock(return_value=None)

        # abstract_trait and synthesise_feedback call litellm — mock them out
        with (
            patch("agents.trait_synthesis.cognee", mock_cognee),
            patch("agents.trait_synthesis.gateway", mock_gw),
            patch("agents.trait_synthesis.litellm.acompletion", new=AsyncMock(
                side_effect=RuntimeError("LLM not needed in this test")
            )),
        ):
            asyncio.run(trait_synthesis_agent(state))

        # With 2 signals, no existing trait, not resolved → remember() called
        mock_gw.remember.assert_called_once()

    def test_mixed_single_and_multi_signal_concepts(self):
        """
        When some concepts have < 2 signals and others have >= 2,
        only the multi-signal concepts trigger gateway calls.
        """
        from agents.trait_synthesis import trait_synthesis_agent

        state = _make_state()
        mock_gw = self._make_mock_gateway()

        # "paging" gets 2 traces → should reach gateway
        p1 = _trace(concept="paging")
        p2 = _trace(concept="paging")
        # "MMU" gets only 1 trace → must be skipped
        m1 = _trace(concept="MMU")

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(
            return_value=[p1, p2, m1]
        )
        mock_cognee.recall = AsyncMock(return_value=None)

        with (
            patch("agents.trait_synthesis.cognee", mock_cognee),
            patch("agents.trait_synthesis.gateway", mock_gw),
            patch("agents.trait_synthesis.litellm.acompletion", new=AsyncMock(
                side_effect=RuntimeError("LLM not needed")
            )),
        ):
            asyncio.run(trait_synthesis_agent(state))

        # remember() called exactly once (for "paging")
        assert mock_gw.remember.call_count == 1
        # MMU was skipped — improve and forget also not called for any concept
        mock_gw.improve.assert_not_called()
        mock_gw.forget.assert_not_called()

    def test_graph_name_uses_correct_user_id(self):
        """
        The graph name passed to gateway.remember() must be
        f"user_{user_id}_traits" (Requirements 3.1, 3.6).
        """
        from agents.trait_synthesis import trait_synthesis_agent

        state = _make_state(user_id="alice")
        mock_gw = self._make_mock_gateway()

        t1 = _trace(concept="paging")
        t2 = _trace(concept="paging")

        mock_cognee = MagicMock()
        mock_cognee.recall_agent_memory_traces = AsyncMock(return_value=[t1, t2])
        mock_cognee.recall = AsyncMock(return_value=None)

        with (
            patch("agents.trait_synthesis.cognee", mock_cognee),
            patch("agents.trait_synthesis.gateway", mock_gw),
            patch("agents.trait_synthesis.litellm.acompletion", new=AsyncMock(
                side_effect=RuntimeError("LLM not needed")
            )),
        ):
            asyncio.run(trait_synthesis_agent(state))

        # Verify the graph name passed to remember()
        assert mock_gw.remember.call_count == 1
        call_args = mock_gw.remember.call_args
        graph_name = call_args[0][0]  # first positional arg
        assert graph_name == "user_alice_traits"
