"""
Unit tests for Task 10.4 — Chain 2 trigger points.

Verifies that the Trait Synthesis Agent is triggered at exactly the three
defined points:
  1. After Teacher session end (entry_point="teacher_node")
  2. After Interviewer session end (entry_point="interviewer_node")
  3. On mode switch (POST /session/mode_switch calls invoke_chain2)

Also verifies that switching away from "digest" mode does NOT invoke Chain 2
(digest mode accumulates no agent-memory traces).

Requirements: 8.2
"""
from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Ensure backend/ is on sys.path
_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from models.schemas import TutorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    user_id: str = "user_42",
    session_id: str = "sess_abc",
    mode: str = "teacher",
) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic="OS memory management",
        current_episode="ep_paging",
        mode=mode,
        session_id=session_id,
        nudge_count=0,
        answer_history=[],
        trait_snapshot=[],
        ingest_needed=False,
    )


# ---------------------------------------------------------------------------
# Trigger 1: invoke_chain2 with entry_point="teacher_node"
#            → trait_synthesis_agent is called
# ---------------------------------------------------------------------------

class TestTrigger1TeacherSessionEnd:
    """After Teacher session ends, Chain 2 routes through teacher_node → trait_synthesis_node."""

    def test_teacher_entry_point_calls_trait_synthesis_agent(self):
        """
        Trigger 1: invoke_chain2(state, entry_point="teacher_node") must cause
        trait_synthesis_agent to be called exactly once.
        """
        from chains.langgraph_chains import invoke_chain2

        state = _make_state(mode="teacher")
        mock_trait_synthesis = AsyncMock(return_value=None)

        with patch("chains.langgraph_chains.trait_synthesis_agent", mock_trait_synthesis):
            asyncio.run(invoke_chain2(state, entry_point="teacher_node"))

        mock_trait_synthesis.assert_called_once()

    def test_teacher_entry_point_passes_state_to_trait_synthesis(self):
        """
        Trigger 1: The state passed into invoke_chain2 is forwarded to
        trait_synthesis_agent (possibly enriched by LangGraph, but user_id
        and session_id must survive).
        """
        from chains.langgraph_chains import invoke_chain2

        state = _make_state(user_id="alice", session_id="sess_teacher_001")
        captured_states: list[TutorState] = []

        async def capture_state(s: TutorState) -> None:
            captured_states.append(s)

        with patch("chains.langgraph_chains.trait_synthesis_agent", capture_state):
            asyncio.run(invoke_chain2(state, entry_point="teacher_node"))

        assert len(captured_states) == 1
        assert captured_states[0]["user_id"] == "alice"
        assert captured_states[0]["session_id"] == "sess_teacher_001"

    def test_teacher_entry_point_returns_tutor_state(self):
        """invoke_chain2 must return a TutorState (the final graph output)."""
        from chains.langgraph_chains import invoke_chain2

        state = _make_state()
        mock_trait_synthesis = AsyncMock(return_value=None)

        with patch("chains.langgraph_chains.trait_synthesis_agent", mock_trait_synthesis):
            result = asyncio.run(invoke_chain2(state, entry_point="teacher_node"))

        # Result must be a dict-like TutorState
        assert isinstance(result, dict)
        assert "user_id" in result


# ---------------------------------------------------------------------------
# Trigger 2: invoke_chain2 with entry_point="interviewer_node"
#            → trait_synthesis_agent is called
# ---------------------------------------------------------------------------

class TestTrigger2InterviewerSessionEnd:
    """After Interviewer session ends, Chain 2 routes through interviewer_node → trait_synthesis_node."""

    def test_interviewer_entry_point_calls_trait_synthesis_agent(self):
        """
        Trigger 2: invoke_chain2(state, entry_point="interviewer_node") must
        cause trait_synthesis_agent to be called exactly once.
        """
        from chains.langgraph_chains import invoke_chain2

        state = _make_state(mode="interviewer")
        mock_trait_synthesis = AsyncMock(return_value=None)

        with patch("chains.langgraph_chains.trait_synthesis_agent", mock_trait_synthesis):
            asyncio.run(invoke_chain2(state, entry_point="interviewer_node"))

        mock_trait_synthesis.assert_called_once()

    def test_interviewer_entry_point_passes_state_to_trait_synthesis(self):
        """
        Trigger 2: The state threaded through invoke_chain2 with the interviewer
        entry point preserves user_id and session_id.
        """
        from chains.langgraph_chains import invoke_chain2

        state = _make_state(user_id="bob", session_id="sess_interviewer_042")
        captured_states: list[TutorState] = []

        async def capture_state(s: TutorState) -> None:
            captured_states.append(s)

        with patch("chains.langgraph_chains.trait_synthesis_agent", capture_state):
            asyncio.run(invoke_chain2(state, entry_point="interviewer_node"))

        assert len(captured_states) == 1
        assert captured_states[0]["user_id"] == "bob"
        assert captured_states[0]["session_id"] == "sess_interviewer_042"

    def test_interviewer_entry_point_returns_tutor_state(self):
        """invoke_chain2 must return a TutorState regardless of entry point."""
        from chains.langgraph_chains import invoke_chain2

        state = _make_state(mode="interviewer")
        mock_trait_synthesis = AsyncMock(return_value=None)

        with patch("chains.langgraph_chains.trait_synthesis_agent", mock_trait_synthesis):
            result = asyncio.run(invoke_chain2(state, entry_point="interviewer_node"))

        assert isinstance(result, dict)
        assert "user_id" in result


# ---------------------------------------------------------------------------
# Both entry points call trait_synthesis_agent — symmetry check
# ---------------------------------------------------------------------------

class TestBothEntryPointsCallTraitSynthesis:
    """Both teacher_node and interviewer_node entry points must reach trait_synthesis_agent."""

    @pytest.mark.parametrize("entry_point", ["teacher_node", "interviewer_node"])
    def test_each_entry_point_invokes_trait_synthesis_once(self, entry_point: str):
        """
        Both Trigger 1 and Trigger 2 must each produce exactly one call to
        trait_synthesis_agent — no double-calls, no missed calls.
        """
        from chains.langgraph_chains import invoke_chain2

        state = _make_state()
        mock_trait_synthesis = AsyncMock(return_value=None)

        with patch("chains.langgraph_chains.trait_synthesis_agent", mock_trait_synthesis):
            asyncio.run(invoke_chain2(state, entry_point=entry_point))

        mock_trait_synthesis.assert_called_once()


# ---------------------------------------------------------------------------
# invoke_chain2 rejects invalid entry points
# ---------------------------------------------------------------------------

class TestInvalidEntryPoint:
    """invoke_chain2 must raise ValueError for any unrecognised entry point."""

    def test_invalid_entry_point_raises_value_error(self):
        from chains.langgraph_chains import invoke_chain2

        state = _make_state()
        with pytest.raises(ValueError, match="invalid entry_point"):
            asyncio.run(invoke_chain2(state, entry_point="digest_node"))

    def test_empty_string_entry_point_raises_value_error(self):
        from chains.langgraph_chains import invoke_chain2

        state = _make_state()
        with pytest.raises(ValueError, match="invalid entry_point"):
            asyncio.run(invoke_chain2(state, entry_point=""))


# ---------------------------------------------------------------------------
# Trigger 3: POST /session/mode_switch
#            → invoke_chain2 is called with correct entry_point
# ---------------------------------------------------------------------------

class TestTrigger3ModeSwitch:
    """
    On mode switch, the session router must call invoke_chain2 with the
    entry_point derived from *current_mode* (the mode being switched away from).

    Requirements: 8.2
    """

    def _make_switch_payload(
        self,
        current_mode: str = "teacher",
        new_mode: str = "interviewer",
    ) -> dict:
        return {
            "user_id": "user_switch_42",
            "session_id": "sess_switch_99",
            "current_mode": current_mode,
            "new_mode": new_mode,
            "topic": "deep learning",
            "current_episode": "ep_transformer",
            "nudge_count": 0,
            "answer_history": [],
            "trait_snapshot": [],
            "ingest_needed": False,
        }

    def test_mode_switch_from_teacher_calls_invoke_chain2_with_teacher_node(self):
        """
        Trigger 3: Switching away from 'teacher' mode must call
        invoke_chain2(..., entry_point='teacher_node').
        """
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_switch_payload(
            current_mode="teacher", new_mode="interviewer"
        ))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            asyncio.run(mode_switch(req))

        mock_invoke.assert_called_once()
        _, kwargs = mock_invoke.call_args
        assert kwargs.get("entry_point") == "teacher_node" or (
            len(mock_invoke.call_args.args) >= 2
            and mock_invoke.call_args.args[1] == "teacher_node"
        )

    def test_mode_switch_from_interviewer_calls_invoke_chain2_with_interviewer_node(self):
        """
        Trigger 3: Switching away from 'interviewer' mode must call
        invoke_chain2(..., entry_point='interviewer_node').
        """
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_switch_payload(
            current_mode="interviewer", new_mode="teacher"
        ))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            asyncio.run(mode_switch(req))

        mock_invoke.assert_called_once()
        _, kwargs = mock_invoke.call_args
        assert kwargs.get("entry_point") == "interviewer_node" or (
            len(mock_invoke.call_args.args) >= 2
            and mock_invoke.call_args.args[1] == "interviewer_node"
        )

    def test_mode_switch_builds_state_with_correct_fields(self):
        """
        Trigger 3: The TutorState built inside mode_switch must carry the
        user_id and session_id from the request.
        """
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        payload = self._make_switch_payload(current_mode="teacher", new_mode="interviewer")
        req = ModeSwitchRequest(**payload)

        captured_states: list[TutorState] = []

        async def capture(state: TutorState, *, entry_point: str) -> TutorState:
            captured_states.append(state)
            return state

        with patch("routers.session.invoke_chain2", capture):
            asyncio.run(mode_switch(req))

        assert len(captured_states) == 1
        state = captured_states[0]
        assert state["user_id"] == "user_switch_42"
        assert state["session_id"] == "sess_switch_99"

    def test_mode_switch_returns_ok_status_after_chain2(self):
        """The endpoint must return status='ok' and include new_mode in the response."""
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_switch_payload(
            current_mode="teacher", new_mode="digest"
        ))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            response = asyncio.run(mode_switch(req))

        assert response.status == "ok"
        assert response.new_mode == "digest"


# ---------------------------------------------------------------------------
# Trigger 3 — digest mode DOES NOT invoke Chain 2
# ---------------------------------------------------------------------------

class TestDigestModeDoesNotTriggerChain2:
    """
    Switching away from 'digest' mode must NOT invoke Chain 2, because
    digest mode does not accumulate agent-memory traces.

    Requirements: 8.2
    """

    def _make_digest_payload(self, new_mode: str = "teacher") -> dict:
        return {
            "user_id": "user_digest_7",
            "session_id": "sess_digest_11",
            "current_mode": "digest",
            "new_mode": new_mode,
            "topic": "OS memory management",
            "current_episode": "",
            "nudge_count": 0,
            "answer_history": [],
            "trait_snapshot": [],
            "ingest_needed": False,
        }

    def test_switching_away_from_digest_does_not_call_invoke_chain2(self):
        """
        When current_mode is 'digest', invoke_chain2 must NOT be called.
        """
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_digest_payload(new_mode="teacher"))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            asyncio.run(mode_switch(req))

        mock_invoke.assert_not_called()

    def test_digest_switch_returns_ok_without_synthesis(self):
        """
        Switching away from digest mode returns status='ok' with an
        appropriate message, even though no trait synthesis runs.
        """
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_digest_payload(new_mode="interviewer"))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            response = asyncio.run(mode_switch(req))

        assert response.status == "ok"
        assert response.new_mode == "interviewer"
        assert response.user_id == "user_digest_7"

    @pytest.mark.parametrize("new_mode", ["teacher", "interviewer"])
    def test_digest_to_any_mode_never_triggers_chain2(self, new_mode: str):
        """Digest → any other mode transition never invokes Chain 2."""
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(**self._make_digest_payload(new_mode=new_mode))
        mock_invoke = AsyncMock(return_value=_make_state())

        with patch("routers.session.invoke_chain2", mock_invoke):
            asyncio.run(mode_switch(req))

        mock_invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Edge-case: invoke_chain2 exception propagates as HTTP 503
# ---------------------------------------------------------------------------

class TestModeSwitchChain2Failure:
    """When invoke_chain2 raises, mode_switch must surface an HTTP 503."""

    def test_chain2_exception_raises_http503(self):
        from fastapi import HTTPException
        from routers.session import mode_switch
        from routers.session import ModeSwitchRequest

        req = ModeSwitchRequest(
            user_id="user_err",
            session_id="sess_err",
            current_mode="teacher",
            new_mode="interviewer",
        )
        mock_invoke = AsyncMock(side_effect=RuntimeError("LangGraph down"))

        with patch("routers.session.invoke_chain2", mock_invoke):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(mode_switch(req))

        assert exc_info.value.status_code == 503
