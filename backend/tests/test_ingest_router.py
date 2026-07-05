"""
Unit tests for Task 12.2 — POST /ingest, POST /session/start, POST /session/end.

Verifies:
  - POST /ingest returns {status: "queued", topic} immediately (async, non-blocking)
  - POST /ingest queues IngestionAgent.run via BackgroundTasks
  - POST /ingest accepts optional video_ids
  - POST /session/start creates a TutorState in the session store with ingest_needed=True
  - POST /session/start returns a unique session_id
  - POST /session/end triggers Chain 2 with the correct entry point for teacher mode
  - POST /session/end triggers Chain 2 with the correct entry point for interviewer mode
  - POST /session/end skips Chain 2 for digest mode
  - POST /session/end removes the session from the store after synthesis
  - POST /session/end returns 503 when Chain 2 raises an exception

Requirements: 11.2
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

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
# IngestRequest model
# ---------------------------------------------------------------------------

class TestIngestRequestModel:
    """IngestRequest must accept topic (required) and optional video_ids."""

    def test_valid_request_topic_only(self):
        from routers.ingest import IngestRequest

        req = IngestRequest(topic="OS memory management")
        assert req.topic == "OS memory management"
        assert req.video_ids is None

    def test_valid_request_with_video_ids(self):
        from routers.ingest import IngestRequest

        req = IngestRequest(topic="deep learning", video_ids=["abc123", "def456"])
        assert req.topic == "deep learning"
        assert req.video_ids == ["abc123", "def456"]

    def test_missing_topic_raises_validation_error(self):
        from pydantic import ValidationError
        from routers.ingest import IngestRequest

        with pytest.raises(ValidationError):
            IngestRequest()


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    """POST /ingest must return {status: queued, topic} and queue the agent."""

    def test_ingest_returns_queued_status(self):
        from routers.ingest import ingest, IngestRequest

        req = IngestRequest(topic="OS memory management")

        background_tasks = MagicMock()
        background_tasks.add_task = MagicMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()

        with patch("routers.ingest.IngestionAgent", return_value=mock_agent_instance):
            response = asyncio.run(ingest(req, background_tasks))

        assert response.status == "queued"
        assert response.topic == "OS memory management"

    def test_ingest_calls_background_add_task(self):
        from routers.ingest import ingest, IngestRequest

        req = IngestRequest(topic="deep learning")

        background_tasks = MagicMock()
        background_tasks.add_task = MagicMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()

        with patch("routers.ingest.IngestionAgent", return_value=mock_agent_instance):
            asyncio.run(ingest(req, background_tasks))

        # BackgroundTasks.add_task must be called with (agent.run, topic, [])
        background_tasks.add_task.assert_called_once_with(
            mock_agent_instance.run, "deep learning", []
        )

    def test_ingest_passes_video_ids_to_background_task(self):
        from routers.ingest import ingest, IngestRequest

        req = IngestRequest(topic="transformers", video_ids=["vid1", "vid2"])

        background_tasks = MagicMock()
        background_tasks.add_task = MagicMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()

        with patch("routers.ingest.IngestionAgent", return_value=mock_agent_instance):
            asyncio.run(ingest(req, background_tasks))

        background_tasks.add_task.assert_called_once_with(
            mock_agent_instance.run, "transformers", ["vid1", "vid2"]
        )

    def test_ingest_does_not_await_agent_run_directly(self):
        """
        The endpoint must NOT await agent.run() directly — it queues via
        BackgroundTasks so the response is returned before ingestion completes.
        """
        from routers.ingest import ingest, IngestRequest

        req = IngestRequest(topic="paging")

        background_tasks = MagicMock()
        background_tasks.add_task = MagicMock()

        mock_agent_instance = MagicMock()
        run_awaited = []
        mock_agent_instance.run = AsyncMock(side_effect=lambda *a, **kw: run_awaited.append(True))

        with patch("routers.ingest.IngestionAgent", return_value=mock_agent_instance):
            asyncio.run(ingest(req, background_tasks))

        # run() should NOT have been awaited directly
        assert run_awaited == []
        # But add_task should have been called
        assert background_tasks.add_task.call_count == 1


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------

class TestSessionStartEndpoint:
    """POST /session/start must create a session and return a session_id."""

    def setup_method(self):
        from routers.chat import _session_store
        _session_store.clear()

    def test_session_start_returns_session_id(self):
        from routers.ingest import session_start, SessionStartRequest

        req = SessionStartRequest(
            user_id="u1",
            topic="OS memory management",
            mode="teacher",
        )

        response = asyncio.run(session_start(req))

        assert response.status == "started"
        assert response.user_id == "u1"
        assert response.topic == "OS memory management"
        assert response.mode == "teacher"
        assert isinstance(response.session_id, str)
        assert len(response.session_id) > 0

    def test_session_start_creates_state_in_store(self):
        from routers.ingest import session_start, SessionStartRequest
        from routers.chat import _session_store

        req = SessionStartRequest(
            user_id="u2",
            topic="deep learning",
            mode="interviewer",
        )

        response = asyncio.run(session_start(req))

        key = ("u2", response.session_id)
        assert key in _session_store

        state = _session_store[key]
        assert state["user_id"] == "u2"
        assert state["topic"] == "deep learning"
        assert state["mode"] == "interviewer"
        assert state["ingest_needed"] is True   # must trigger Chain 1 on first chat turn
        assert state["session_id"] == response.session_id

    def test_session_start_ingest_needed_is_true(self):
        """New sessions must have ingest_needed=True so Chain 1 runs on first chat turn."""
        from routers.ingest import session_start, SessionStartRequest
        from routers.chat import _session_store

        req = SessionStartRequest(user_id="u3", topic="paging")

        response = asyncio.run(session_start(req))

        state = _session_store[("u3", response.session_id)]
        assert state["ingest_needed"] is True

    def test_session_start_generates_unique_session_ids(self):
        """Two calls for the same user must produce different session_ids."""
        from routers.ingest import session_start, SessionStartRequest

        req = SessionStartRequest(user_id="u4", topic="deep learning")

        r1 = asyncio.run(session_start(req))
        r2 = asyncio.run(session_start(req))

        assert r1.session_id != r2.session_id

    def test_session_start_default_mode_is_teacher(self):
        """Default mode should be 'teacher' when not specified."""
        from routers.ingest import session_start, SessionStartRequest

        req = SessionStartRequest(user_id="u5", topic="backpropagation")

        response = asyncio.run(session_start(req))

        assert response.mode == "teacher"


# ---------------------------------------------------------------------------
# POST /session/end
# ---------------------------------------------------------------------------

class TestSessionEndEndpoint:
    """POST /session/end must trigger Chain 2 and clean up the session store."""

    def setup_method(self):
        from routers.chat import _session_store
        _session_store.clear()

    def test_session_end_teacher_mode_triggers_chain2_with_teacher_node(self):
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        state = _make_state(user_id="u1", session_id="s1", mode="teacher")
        _session_store[("u1", "s1")] = state

        req = SessionEndRequest(user_id="u1", session_id="s1")

        chain2_calls: list = []

        async def fake_chain2(st, entry_point="teacher_node"):
            chain2_calls.append(entry_point)
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            response = asyncio.run(session_end(req))

        assert chain2_calls == ["teacher_node"]
        assert response.status == "ok"

    def test_session_end_interviewer_mode_triggers_chain2_with_interviewer_node(self):
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        state = _make_state(user_id="u2", session_id="s2", mode="interviewer")
        _session_store[("u2", "s2")] = state

        req = SessionEndRequest(user_id="u2", session_id="s2")

        chain2_calls: list = []

        async def fake_chain2(st, entry_point="teacher_node"):
            chain2_calls.append(entry_point)
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            response = asyncio.run(session_end(req))

        assert chain2_calls == ["interviewer_node"]
        assert response.status == "ok"

    def test_session_end_digest_mode_skips_chain2(self):
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        state = _make_state(user_id="u3", session_id="s3", mode="digest")
        _session_store[("u3", "s3")] = state

        req = SessionEndRequest(user_id="u3", session_id="s3")

        chain2_calls: list = []

        async def fake_chain2(st, entry_point="teacher_node"):
            chain2_calls.append(entry_point)
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            response = asyncio.run(session_end(req))

        assert chain2_calls == []   # no synthesis for digest mode
        assert response.status == "ok"
        assert "digest" in response.message.lower()

    def test_session_end_removes_session_from_store_after_synthesis(self):
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        state = _make_state(user_id="u4", session_id="s4", mode="teacher")
        _session_store[("u4", "s4")] = state

        req = SessionEndRequest(user_id="u4", session_id="s4")

        async def fake_chain2(st, entry_point="teacher_node"):
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            asyncio.run(session_end(req))

        assert ("u4", "s4") not in _session_store

    def test_session_end_raises_503_when_chain2_fails(self):
        from fastapi import HTTPException
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        state = _make_state(user_id="u5", session_id="s5", mode="teacher")
        _session_store[("u5", "s5")] = state

        req = SessionEndRequest(user_id="u5", session_id="s5")

        async def failing_chain2(st, entry_point="teacher_node"):
            raise RuntimeError("LangGraph node failed")

        with patch("routers.ingest.invoke_chain2", failing_chain2):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(session_end(req))

        assert exc_info.value.status_code == 503

    def test_session_end_mode_override_takes_precedence(self):
        """
        When a mode is supplied in the request body it should override the
        stored mode for Chain 2 entry point selection.
        """
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        # Store has teacher mode
        state = _make_state(user_id="u6", session_id="s6", mode="teacher")
        _session_store[("u6", "s6")] = state

        # Request overrides to interviewer
        req = SessionEndRequest(user_id="u6", session_id="s6", mode="interviewer")

        chain2_calls: list = []

        async def fake_chain2(st, entry_point="teacher_node"):
            chain2_calls.append(entry_point)
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            asyncio.run(session_end(req))

        assert chain2_calls == ["interviewer_node"]

    def test_session_end_works_when_session_not_in_store(self):
        """
        Session not in store (e.g. restarted server) — should build a minimal
        state from request fields and still invoke Chain 2.
        """
        from routers.ingest import session_end, SessionEndRequest
        from routers.chat import _session_store

        # Ensure the key is absent
        _session_store.pop(("u7", "s7"), None)

        req = SessionEndRequest(
            user_id="u7",
            session_id="s7",
            mode="teacher",
        )

        chain2_calls: list = []

        async def fake_chain2(st, entry_point="teacher_node"):
            chain2_calls.append(entry_point)
            return st

        with patch("routers.ingest.invoke_chain2", fake_chain2):
            response = asyncio.run(session_end(req))

        assert chain2_calls == ["teacher_node"]
        assert response.status == "ok"
