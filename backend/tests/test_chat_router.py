"""
Unit tests for Task 12.1 — POST /chat SSE endpoint.

Verifies:
  - ChatRequest model field validation
  - load_or_create_state creates a fresh TutorState with ingest_needed=True for new sessions
  - load_or_create_state returns the cached state for existing sessions
  - load_or_create_state sets ingest_needed=True when the topic changes
  - load_or_create_state updates mode when it changes for an existing session
  - The POST /chat endpoint returns an EventSourceResponse (SSE, not WebSockets)
  - Chain 1 is invoked when ingest_needed=True
  - Agent is invoked directly (no LangGraph) for mid-session turns

Requirements: 11.1, 11.3, 11.4
"""
from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend/ is on sys.path
_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    user_id: str = "user_1",
    session_id: str = "sess_1",
    mode: str = "teacher",
    topic: str = "OS memory management",
    ingest_needed: bool = False,
) -> dict:
    return {
        "user_id": user_id,
        "topic": topic,
        "current_episode": "",
        "mode": mode,
        "session_id": session_id,
        "nudge_count": 0,
        "answer_history": [],
        "trait_snapshot": [],
        "ingest_needed": ingest_needed,
    }


# ---------------------------------------------------------------------------
# ChatRequest model
# ---------------------------------------------------------------------------

class TestChatRequestModel:
    """ChatRequest must accept the expected fields and reject invalid modes."""

    def test_valid_teacher_request(self):
        from routers.chat import ChatRequest

        req = ChatRequest(
            user_id="u1",
            session_id="s1",
            message="What is paging?",
            mode="teacher",
            topic="OS memory management",
        )
        assert req.user_id == "u1"
        assert req.mode == "teacher"
        assert req.topic == "OS memory management"

    def test_valid_interviewer_request(self):
        from routers.chat import ChatRequest

        req = ChatRequest(
            user_id="u2",
            session_id="s2",
            message="__session_start__",
            mode="interviewer",
            topic="deep learning",
        )
        assert req.mode == "interviewer"

    def test_invalid_mode_raises_validation_error(self):
        from pydantic import ValidationError
        from routers.chat import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(
                user_id="u3",
                session_id="s3",
                message="hello",
                mode="websocket",   # invalid — Req 11.4
                topic="deep learning",
            )


# ---------------------------------------------------------------------------
# load_or_create_state
# ---------------------------------------------------------------------------

class TestLoadOrCreateState:
    """Tests for the session-state management helper."""

    def setup_method(self):
        """Clear the session store before each test to avoid cross-test pollution."""
        from routers.chat import _session_store
        _session_store.clear()

    def test_new_session_creates_state_with_ingest_needed_true(self):
        from routers.chat import load_or_create_state

        state = load_or_create_state("u1", "s1", "teacher", "OS memory management")

        assert state["user_id"] == "u1"
        assert state["session_id"] == "s1"
        assert state["mode"] == "teacher"
        assert state["topic"] == "OS memory management"
        assert state["ingest_needed"] is True    # Chain 1 must run first turn

    def test_new_session_initialises_history_to_empty_list(self):
        from routers.chat import load_or_create_state

        state = load_or_create_state("u2", "s2", "teacher", "deep learning")

        assert state["answer_history"] == []
        assert state["trait_snapshot"] == []
        assert state["nudge_count"] == 0

    def test_existing_session_returns_cached_state(self):
        from routers.chat import load_or_create_state, _session_store

        # First call creates the state
        state1 = load_or_create_state("u3", "s3", "teacher", "OS memory management")
        # Simulate a turn having occurred (ingest_needed flipped)
        state1["ingest_needed"] = False
        state1["current_episode"] = "ep_paging"
        _session_store[("u3", "s3")] = state1

        # Second call must return the same object
        state2 = load_or_create_state("u3", "s3", "teacher", "OS memory management")
        assert state2 is state1
        assert state2["ingest_needed"] is False
        assert state2["current_episode"] == "ep_paging"

    def test_mode_change_updates_existing_state(self):
        from routers.chat import load_or_create_state, _session_store

        state = load_or_create_state("u4", "s4", "teacher", "deep learning")
        state["ingest_needed"] = False
        _session_store[("u4", "s4")] = state

        # Switch mode
        updated = load_or_create_state("u4", "s4", "interviewer", "deep learning")
        assert updated["mode"] == "interviewer"

    def test_topic_change_resets_ingest_needed(self):
        from routers.chat import load_or_create_state, _session_store

        state = load_or_create_state("u5", "s5", "teacher", "OS memory management")
        state["ingest_needed"] = False
        _session_store[("u5", "s5")] = state

        # Change topic mid-session
        updated = load_or_create_state("u5", "s5", "teacher", "deep learning")
        assert updated["topic"] == "deep learning"
        assert updated["ingest_needed"] is True     # new topic needs ingestion

    def test_different_sessions_are_independent(self):
        from routers.chat import load_or_create_state

        s1 = load_or_create_state("u6", "sess_a", "teacher", "OS memory management")
        s2 = load_or_create_state("u6", "sess_b", "teacher", "deep learning")

        assert s1 is not s2
        assert s1["session_id"] == "sess_a"
        assert s2["session_id"] == "sess_b"
        assert s2["topic"] == "deep learning"


# ---------------------------------------------------------------------------
# POST /chat — endpoint returns EventSourceResponse (SSE, not WebSocket)
# ---------------------------------------------------------------------------

class TestChatEndpointSSE:
    """The /chat endpoint must return EventSourceResponse — no WebSockets."""

    def setup_method(self):
        from routers.chat import _session_store
        _session_store.clear()

    def test_chat_returns_event_source_response(self):
        """
        POST /chat must return an EventSourceResponse (SSE).
        Requirement 11.4: no WebSockets.
        """
        from sse_starlette.sse import EventSourceResponse
        from routers.chat import chat, ChatRequest

        req = ChatRequest(
            user_id="u_sse",
            session_id="s_sse",
            message="hello",
            mode="teacher",
            topic="deep learning",
        )

        mock_chain1 = AsyncMock(return_value=_make_state(ingest_needed=False))

        async def fake_teacher(state, msg):
            yield "hello "
            yield "world"

        with (
            patch("routers.chat.invoke_chain1", mock_chain1),
            patch("routers.chat.teacher_agent", fake_teacher),
        ):
            response = asyncio.run(chat(req))

        assert isinstance(response, EventSourceResponse)

    def test_chat_uses_teacher_agent_for_teacher_mode(self):
        """teacher mode must route to teacher_agent, not interviewer_agent."""
        from routers.chat import chat, ChatRequest, _session_store

        # Pre-populate state so ingest_needed is False (mid-session turn)
        state = _make_state(user_id="u_t", session_id="s_t", mode="teacher",
                            topic="deep learning", ingest_needed=False)
        _session_store[("u_t", "s_t")] = state

        req = ChatRequest(
            user_id="u_t",
            session_id="s_t",
            message="explain paging",
            mode="teacher",
            topic="deep learning",
        )

        teacher_calls: list = []
        interviewer_calls: list = []

        async def fake_teacher(s, msg):
            teacher_calls.append(msg)
            yield "tok"

        async def fake_interviewer(s, msg):
            interviewer_calls.append(msg)
            yield "tok"

        async def consume():
            with (
                patch("routers.chat.teacher_agent", fake_teacher),
                patch("routers.chat.interviewer_agent", fake_interviewer),
            ):
                response = await chat(req)
                # Consume the generator so the inner function runs
                async for _ in response.body_iterator:
                    pass

        asyncio.run(consume())

        assert teacher_calls == ["explain paging"]
        assert interviewer_calls == []

    def test_chat_uses_interviewer_agent_for_interviewer_mode(self):
        """interviewer mode must route to interviewer_agent, not teacher_agent."""
        from routers.chat import chat, ChatRequest, _session_store

        state = _make_state(user_id="u_i", session_id="s_i", mode="interviewer",
                            topic="deep learning", ingest_needed=False)
        _session_store[("u_i", "s_i")] = state

        req = ChatRequest(
            user_id="u_i",
            session_id="s_i",
            message="__session_start__",
            mode="interviewer",
            topic="deep learning",
        )

        teacher_calls: list = []
        interviewer_calls: list = []

        async def fake_teacher(s, msg):
            teacher_calls.append(msg)
            yield "tok"

        async def fake_interviewer(s, msg):
            interviewer_calls.append(msg)
            yield "tok"

        async def consume():
            with (
                patch("routers.chat.teacher_agent", fake_teacher),
                patch("routers.chat.interviewer_agent", fake_interviewer),
            ):
                response = await chat(req)
                async for _ in response.body_iterator:
                    pass

        asyncio.run(consume())

        assert interviewer_calls == ["__session_start__"]
        assert teacher_calls == []

    def test_chat_invokes_chain1_when_ingest_needed(self):
        """
        When ingest_needed=True (new topic), Chain 1 must be invoked before
        the agent is called — mid-session turns bypass LangGraph entirely.
        """
        from routers.chat import chat, ChatRequest, _session_store

        # New session — no pre-populated state (ingest_needed will default True)
        req = ChatRequest(
            user_id="u_chain",
            session_id="s_chain",
            message="first message",
            mode="teacher",
            topic="OS memory management",
        )

        chain1_calls: list = []

        async def fake_chain1(state):
            chain1_calls.append(state["topic"])
            return {**state, "ingest_needed": False}

        async def fake_teacher(s, msg):
            yield "token"

        async def consume():
            with (
                patch("routers.chat.invoke_chain1", fake_chain1),
                patch("routers.chat.teacher_agent", fake_teacher),
            ):
                response = await chat(req)
                async for _ in response.body_iterator:
                    pass

        asyncio.run(consume())

        assert chain1_calls == ["OS memory management"]

    def test_chat_skips_chain1_for_mid_session_turn(self):
        """
        Mid-session turns (ingest_needed=False) must NOT invoke Chain 1.
        """
        from routers.chat import chat, ChatRequest, _session_store

        state = _make_state(user_id="u_mid", session_id="s_mid", mode="teacher",
                            topic="OS memory management", ingest_needed=False)
        _session_store[("u_mid", "s_mid")] = state

        req = ChatRequest(
            user_id="u_mid",
            session_id="s_mid",
            message="second message",
            mode="teacher",
            topic="OS memory management",
        )

        chain1_calls: list = []

        async def fake_chain1(st):
            chain1_calls.append(st)
            return st

        async def fake_teacher(s, msg):
            yield "token"

        async def consume():
            with (
                patch("routers.chat.invoke_chain1", fake_chain1),
                patch("routers.chat.teacher_agent", fake_teacher),
            ):
                response = await chat(req)
                async for _ in response.body_iterator:
                    pass

        asyncio.run(consume())

        assert chain1_calls == []   # Chain 1 must NOT be called mid-session

    def test_chat_stream_includes_done_event(self):
        """The SSE stream must end with a 'done' event."""
        from routers.chat import chat, ChatRequest, _session_store

        state = _make_state(user_id="u_done", session_id="s_done", mode="teacher",
                            topic="deep learning", ingest_needed=False)
        _session_store[("u_done", "s_done")] = state

        req = ChatRequest(
            user_id="u_done",
            session_id="s_done",
            message="tell me",
            mode="teacher",
            topic="deep learning",
        )

        async def fake_teacher(s, msg):
            yield "some "
            yield "tokens"

        chunks: list[bytes] = []

        async def consume():
            with patch("routers.chat.teacher_agent", fake_teacher):
                response = await chat(req)
                async for chunk in response.body_iterator:
                    if isinstance(chunk, bytes):
                        chunks.append(chunk)
                    elif isinstance(chunk, str):
                        chunks.append(chunk.encode())

        asyncio.run(consume())

        full_output = b"".join(chunks).decode()
        # The stream must contain the "done" event
        assert "done" in full_output
