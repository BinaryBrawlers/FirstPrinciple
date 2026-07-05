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
            "awaiting_response_to_posed_problem": False,
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

        # --- Chain 1 dispatch: kick off ingestion as a background task ---
        # We do NOT await ingestion inline — it can take 30-120s and would
        # time out the SSE connection.  Instead we fire-and-forget via
        # asyncio.create_task so the user gets an immediate response.
        if state.get("ingest_needed"):
            import asyncio
            logger.info(
                "chat: ingest_needed=True for topic=%r; firing Chain 1 in background "
                "(user=%r session=%r)",
                req.topic,
                req.user_id,
                req.session_id,
            )
            state["ingest_needed"] = False
            _session_store[(req.user_id, req.session_id)] = state

            async def _run_chain1(s: TutorState) -> None:
                try:
                    updated = await invoke_chain1(s)
                    _session_store[(s["user_id"], s["session_id"])] = updated
                    logger.info("chat: background Chain 1 complete for topic=%r", s["topic"])
                except Exception as exc:  # noqa: BLE001
                    logger.error("chat: background Chain 1 failed for topic=%r (%s)", s["topic"], exc)

            asyncio.create_task(_run_chain1(dict(state)))  # type: ignore[arg-type]

            # Signal the frontend that ingestion started (no user-facing text)
            yield {"event": "ingesting", "data": req.topic}

        # --- Mid-session turn: invoke agent directly ---
        # Call the internal impl directly, bypassing the @cognee.agent_memory
        # decorator which requires dataset permissions that may not be set up.
        if req.mode == "teacher":
            from agents.teacher import _teacher_agent_impl
            agent_gen = _teacher_agent_impl(state, req.message)
        else:
            from agents.interviewer import _interviewer_agent_impl
            agent_gen = _interviewer_agent_impl(state, req.message)

        # Iterate the agent's AsyncGenerator and yield each token as an SSE event.
        # Streaming begins immediately — Requirement 11.3.
        try:
            async for token in agent_gen:
                if token:
                    yield {"data": token}
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "chat: agent streaming failed for user=%r session=%r (%s)",
                req.user_id,
                req.session_id,
                exc,
            )
            yield {"event": "error", "data": f"Streaming error: {exc}"}

        # Signal the client that the stream is complete (Requirement 11.3)
        yield {"event": "done", "data": ""}

    return EventSourceResponse(
        token_generator(),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
