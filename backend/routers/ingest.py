"""
Ingest router — POST /ingest, POST /session/start, POST /session/end.

POST /ingest
  Accepts a topic name (and optional video_ids) and triggers the Ingestion
  Agent asynchronously via FastAPI BackgroundTasks. Returns immediately with
  {"status": "queued", "topic": topic} before ingestion completes.

POST /session/start
  Initialises a new session and returns a session_id. Stores the initial
  TutorState in the shared in-memory session store used by the chat router.

POST /session/end
  Triggers Chain 2 (trait synthesis) for the session being ended.
  Entry point is chosen from the session's current mode:
    "teacher"     → "teacher_node"
    "interviewer" → "interviewer_node"
    "digest"      → no synthesis (digest mode accumulates no agent traces)

Requirements: 11.2
"""
from __future__ import annotations

import logging
import uuid
from typing import Literal, Optional

# --- config MUST come before any cognee import ---
import config  # noqa: F401

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from agents.ingestion import IngestionAgent
from chains.langgraph_chains import invoke_chain2
from models.schemas import TutorState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

# ---------------------------------------------------------------------------
# Shared session store
# In-memory store keyed by (user_id, session_id).  The chat router uses
# the same store (imported from routers.chat) at runtime; we keep a
# module-level reference here that main.py can wire up, but for the
# session/start and session/end endpoints we manage sessions directly via
# the same _session_store dict that chat.py exposes.
# ---------------------------------------------------------------------------

# We import the shared store from chat.py at call time (inside functions)
# to avoid circular imports at module load time.


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    topic: str
    video_ids: Optional[list[str]] = None


class IngestResponse(BaseModel):
    status: str
    topic: str


class SessionStartRequest(BaseModel):
    user_id: str
    topic: str
    mode: Literal["teacher", "interviewer", "digest"] = "teacher"


class SessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    topic: str
    mode: str
    status: str


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str
    # Optional overrides; if not supplied the store's current values are used.
    mode: Optional[Literal["teacher", "interviewer", "digest"]] = None
    topic: Optional[str] = None
    current_episode: Optional[str] = None
    nudge_count: Optional[int] = None
    answer_history: Optional[list[dict]] = None
    trait_snapshot: Optional[list[str]] = None
    ingest_needed: Optional[bool] = None


class SessionEndResponse(BaseModel):
    status: str
    message: str
    user_id: str
    session_id: str


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Queue an ingestion job for *topic* and return immediately.

    The Ingestion Agent runs in the background via FastAPI BackgroundTasks;
    the client does NOT need to wait for ingestion to complete before sending
    chat messages (the chat router will invoke Chain 1 on the first turn if
    the topic has not yet been ingested).

    Request:  {topic: str, video_ids?: list[str]}
    Response: {status: "queued", topic: str}

    Requirements: 11.2
    """
    logger.info("ingest: queueing background ingestion for topic=%r", req.topic)

    agent = IngestionAgent()
    video_ids = req.video_ids or []

    background_tasks.add_task(agent.run, req.topic, video_ids)

    return IngestResponse(status="queued", topic=req.topic)


# ---------------------------------------------------------------------------
# POST /session/start
# ---------------------------------------------------------------------------


@router.post("/session/start", response_model=SessionStartResponse)
async def session_start(req: SessionStartRequest) -> SessionStartResponse:
    """
    Initialise a new learning session and return a freshly generated session_id.

    Creates a TutorState in the shared in-memory session store (the same store
    used by POST /chat) so that subsequent chat turns can resolve the session.

    The session is created with ``ingest_needed=True`` so Chain 1 runs on the
    first POST /chat turn for this session.

    Requirements: 11.2
    """
    from routers.chat import _session_store  # deferred to avoid circular import

    session_id = str(uuid.uuid4())
    key = (req.user_id, session_id)

    state: TutorState = {
        "user_id": req.user_id,
        "topic": req.topic,
        "current_episode": "",
        "mode": req.mode,
        "session_id": session_id,
        "nudge_count": 0,
        "answer_history": [],
        "trait_snapshot": [],
        "ingest_needed": True,  # triggers Chain 1 on first chat turn
    }
    _session_store[key] = state

    logger.info(
        "session_start: created session_id=%r for user=%r topic=%r mode=%r",
        session_id,
        req.user_id,
        req.topic,
        req.mode,
    )

    return SessionStartResponse(
        session_id=session_id,
        user_id=req.user_id,
        topic=req.topic,
        mode=req.mode,
        status="started",
    )


# ---------------------------------------------------------------------------
# POST /session/end
# ---------------------------------------------------------------------------


@router.post("/session/end", response_model=SessionEndResponse)
async def session_end(req: SessionEndRequest) -> SessionEndResponse:
    """
    End a session and trigger Chain 2 (trait synthesis).

    Looks up the TutorState from the shared session store.  Field overrides
    in the request body (mode, topic, etc.) take precedence over the stored
    values, allowing the client to send final session metadata.

    Chain 2 entry point is chosen from the effective mode:
      "teacher"     → "teacher_node"
      "interviewer" → "interviewer_node"
      "digest"      → no synthesis (digest mode doesn't accumulate traces)

    The session is removed from the store after synthesis completes so that
    subsequent requests for the same session_id start fresh.

    Raises HTTP 503 if Chain 2 fails.

    Requirements: 11.2, 8.2
    """
    from routers.chat import _session_store  # deferred to avoid circular import

    key = (req.user_id, req.session_id)
    stored_state = _session_store.get(key)

    # Build effective state — start from stored values if available, then apply
    # any overrides supplied in the request body.
    if stored_state is not None:
        effective_state: TutorState = dict(stored_state)  # type: ignore[assignment]
    else:
        # Session not in store (e.g. created outside this backend instance or
        # already cleaned up). Build a minimal state from the request fields.
        logger.warning(
            "session_end: session (%r, %r) not found in store; building from request fields",
            req.user_id,
            req.session_id,
        )
        effective_state = {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "topic": "",
            "current_episode": "",
            "mode": "teacher",
            "nudge_count": 0,
            "answer_history": [],
            "trait_snapshot": [],
            "ingest_needed": False,
        }

    # Apply request body overrides
    if req.mode is not None:
        effective_state["mode"] = req.mode
    if req.topic is not None:
        effective_state["topic"] = req.topic
    if req.current_episode is not None:
        effective_state["current_episode"] = req.current_episode
    if req.nudge_count is not None:
        effective_state["nudge_count"] = req.nudge_count
    if req.answer_history is not None:
        effective_state["answer_history"] = req.answer_history
    if req.trait_snapshot is not None:
        effective_state["trait_snapshot"] = req.trait_snapshot
    if req.ingest_needed is not None:
        effective_state["ingest_needed"] = req.ingest_needed

    effective_mode = effective_state["mode"]

    logger.info(
        "session_end: ending session_id=%r for user=%r mode=%r",
        req.session_id,
        req.user_id,
        effective_mode,
    )

    # Digest mode accumulates no agent-memory traces; skip synthesis.
    if effective_mode == "digest":
        _session_store.pop(key, None)
        logger.info(
            "session_end: digest mode — skipping trait synthesis for session=%r",
            req.session_id,
        )
        return SessionEndResponse(
            status="ok",
            message="Session ended (digest mode — no trait synthesis triggered).",
            user_id=req.user_id,
            session_id=req.session_id,
        )

    # Map mode → Chain 2 entry point
    entry_point_map: dict[str, str] = {
        "teacher": "teacher_node",
        "interviewer": "interviewer_node",
    }
    entry_point = entry_point_map[effective_mode]

    # Invoke Chain 2 (trait synthesis)
    try:
        await invoke_chain2(effective_state, entry_point=entry_point)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "session_end: Chain 2 failed for session=%r user=%r (%s)",
            req.session_id,
            req.user_id,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trait synthesis chain failed at session end: {exc}. "
                "Session data is retained — retry if synthesis is required."
            ),
        ) from exc

    # Remove session from store after successful synthesis
    _session_store.pop(key, None)

    logger.info(
        "session_end: Chain 2 completed successfully for session=%r user=%r",
        req.session_id,
        req.user_id,
    )

    return SessionEndResponse(
        status="ok",
        message=(
            f"Session ended. Trait synthesis complete "
            f"(entry_point='{entry_point}')."
        ),
        user_id=req.user_id,
        session_id=req.session_id,
    )
