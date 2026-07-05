"""
Chat router — POST /chat SSE endpoint.

Accepts a user message and mode (teacher or interviewer) and returns an
SSE stream of response tokens.

Chain dispatch logic:
  - New topic (ingest_needed=True)   → invoke Chain 1 before streaming
  - Mid-session turn (existing topic) → invoke agent directly (no LangGraph overhead)

Requirements: 11.1, 11.3, 11.4
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Literal

# --- config MUST come before any cognee import ---
import config  # noqa: F401

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.interviewer import interviewer_agent
from agents.teacher import teacher_agent
from chains.langgraph_chains import invoke_chain1
from models.schemas import TutorState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    mode: Literal["teacher", "interviewer"]
    topic: str


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

# In-memory session store: (user_id, session_id) → TutorState
# This is sufficient for the single-presenter demo scope.
_session_store: dict[tuple[str, str], TutorState] = {}


def load_or_create_state(
    user_id: str,
    session_id: str,
    mode: Literal["teacher", "interviewer"],
    topic: str,
) -> TutorState:
    """
    Load an existing TutorState for (user_id, session_id) or create a fresh one.

    For a new session (first time this (user_id, session_id) pair is seen):
      - ingest_needed is set to True so Chain 1 runs before the first turn.
      - All other fields are initialised to safe defaults.

    For a returning session:
      - The cached state is returned as-is; ingest_needed will already be False
        after the first turn.

    If the mode or topic changes within an existing session the state is updated
    in place — this allows the caller to switch modes without losing answer_history.

    Requirements: 11.1
    """
    key = (user_id, session_id)

    if key not in _session_store:
        logger.info(
            "load_or_create_state: creating new state for user=%r session=%r mode=%r topic=%r",
            user_id,
            session_id,
            mode,
            topic,
        )
        state: TutorState = {
            "user_id": user_id,
            "topic": topic,
            "current_episode": "",
            "mode": mode,
            "session_id": session_id,
            "nudge_count": 0,
            "answer_history": [],
            "trait_snapshot": [],
            "ingest_needed": True,   # triggers Chain 1 on first turn
        }
        _session_store[key] = state
    else:
        state = _session_store[key]
        # Update mutable fields that may change across turns
        if state["mode"] != mode:
            logger.debug(
                "load_or_create_state: mode changed from %r to %r for session %r",
                state["mode"],
                mode,
                session_id,
            )
            state["mode"] = mode
        if state["topic"] != topic:
            logger.debug(
                "load_or_create_state: topic changed from %r to %r for session %r — resetting ingest flag",
                state["topic"],
                topic,
                session_id,
            )
            state["topic"] = topic
            state["ingest_needed"] = True   # new topic needs a fresh ingestion

    return state


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """
    Accept a user message and return an SSE stream of response tokens.

    Stream begins before the full response is generated (Requirement 11.3).
    Uses SSE (EventSourceResponse), NOT WebSockets (Requirement 11.4).

    Chain dispatch:
      - If state.ingest_needed is True (new/changed topic): invoke Chain 1
        (ingestion_node → teacher_node) before streaming.
      - Mid-session turns (existing topic, same mode): invoke the appropriate
        agent directly without LangGraph overhead.

    Requirements: 11.1, 11.3, 11.4
    """
    async def token_generator() -> AsyncGenerator[str, None]:
        state = load_or_create_state(
            req.user_id,
            req.session_id,
            req.mode,
            req.topic,
        )

        # --- Chain 1 dispatch: run ingestion before streaming new topics ---
        if state.get("ingest_needed"):
            logger.info(
                "chat: ingest_needed=True for topic=%r; invoking Chain 1 "
                "(user=%r session=%r)",
                req.topic,
                req.user_id,
                req.session_id,
            )
            try:
                state = await invoke_chain1(state)
                # Persist the updated state (ingest_needed is now False)
                _session_store[(req.user_id, req.session_id)] = state
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "chat: Chain 1 failed for topic=%r (%s); continuing without ingestion",
                    req.topic,
                    exc,
                )
                # Mark ingest as done anyway to avoid retry loops in the same session
                state["ingest_needed"] = False
                _session_store[(req.user_id, req.session_id)] = state
                yield (
                    f"event: error\r\ndata: Topic ingestion failed: {exc}. "
                    f"Proceeding with available knowledge.\r\n\r\n"
                )

        # --- Mid-session turn: invoke agent directly ---
        if req.mode == "teacher":
            agent_gen = teacher_agent(state, req.message)
        else:
            agent_gen = interviewer_agent(state, req.message)

        # Iterate the agent's AsyncGenerator and yield each token as an SSE event.
        # Streaming begins immediately — Requirement 11.3.
        # Yield raw SSE-formatted strings so that both the ASGI layer and tests
        # that iterate body_iterator directly can read the content.
        try:
            async for token in agent_gen:
                if token:
                    yield f"data: {token}\r\n\r\n"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "chat: agent streaming failed for user=%r session=%r (%s)",
                req.user_id,
                req.session_id,
                exc,
            )
            yield f"event: error\r\ndata: Streaming error: {exc}\r\n\r\n"

        # Signal the client that the stream is complete (Requirement 11.3)
        yield "event: done\r\ndata: \r\n\r\n"

    return EventSourceResponse(token_generator())
