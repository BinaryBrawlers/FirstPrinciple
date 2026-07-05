"""
Session router — mode_switch endpoint.

POST /session/mode_switch

Handles Trigger 3 for the Trait Synthesis Agent:
  "On mode switch, the application layer emits a mode_switch event that
   invokes Chain 2 (trait synthesis)."

When the frontend switches the learner from "teacher" → "interviewer" (or
vice versa), the current session's agent-memory traces should be synthesised
into Track B before the new mode begins.

Chain 2 entry point is chosen based on the *current* mode (the mode being
switched away FROM):
  teacher    → entry_point="teacher_node"
  interviewer → entry_point="interviewer_node"

Requirements: 8.2
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chains.langgraph_chains import invoke_chain2
from models.schemas import TutorState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/session", tags=["session"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ModeSwitchRequest(BaseModel):
    user_id: str
    session_id: str
    current_mode: Literal["teacher", "interviewer", "digest"]
    new_mode: Literal["teacher", "interviewer", "digest"]
    # Optional TutorState fields to carry forward; defaults are safe
    topic: str = ""
    current_episode: str = ""
    nudge_count: int = 0
    answer_history: list[dict] = []
    trait_snapshot: list[str] = []
    ingest_needed: bool = False


class ModeSwitchResponse(BaseModel):
    status: str
    message: str
    user_id: str
    session_id: str
    new_mode: str


# ---------------------------------------------------------------------------
# POST /session/mode_switch
# ---------------------------------------------------------------------------


@router.post("/mode_switch", response_model=ModeSwitchResponse)
async def mode_switch(req: ModeSwitchRequest) -> ModeSwitchResponse:
    """
    Emit a mode-switch event that triggers Chain 2 (Trait Synthesis Agent).

    Steps:
    1. Build a TutorState from the request body.
    2. Determine the Chain 2 entry point based on *current_mode*.
       - "teacher"    → "teacher_node"
       - "interviewer" → "interviewer_node"
       - "digest"     → no trait synthesis (digest mode doesn't accumulate traces)
    3. Invoke Chain 2 via invoke_chain2().
    4. Return a confirmation response.

    Requirements: 8.2
    """
    logger.info(
        "mode_switch: user=%r session=%r %r → %r",
        req.user_id,
        req.session_id,
        req.current_mode,
        req.new_mode,
    )

    # Digest mode switching away doesn't accumulate agent-memory traces;
    # skip trait synthesis in that case.
    if req.current_mode == "digest":
        logger.info(
            "mode_switch: switching away from digest mode — skipping trait synthesis"
        )
        return ModeSwitchResponse(
            status="ok",
            message="Mode switched from digest — no trait synthesis triggered.",
            user_id=req.user_id,
            session_id=req.session_id,
            new_mode=req.new_mode,
        )

    # Map current mode → Chain 2 entry point
    entry_point_map: dict[str, str] = {
        "teacher": "teacher_node",
        "interviewer": "interviewer_node",
    }
    entry_point = entry_point_map[req.current_mode]

    # Build TutorState for Chain 2
    state: TutorState = {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "mode": req.current_mode,
        "topic": req.topic,
        "current_episode": req.current_episode,
        "nudge_count": req.nudge_count,
        "answer_history": req.answer_history,
        "trait_snapshot": req.trait_snapshot,
        "ingest_needed": req.ingest_needed,
    }

    try:
        await invoke_chain2(state, entry_point=entry_point)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "mode_switch: Chain 2 invocation failed for user=%r session=%r (%s)",
            req.user_id,
            req.session_id,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trait synthesis chain failed during mode switch: {exc}. "
                "The mode switch itself is not blocked — retry if synthesis is required."
            ),
        ) from exc

    logger.info(
        "mode_switch: Chain 2 completed successfully for user=%r session=%r",
        req.user_id,
        req.session_id,
    )

    return ModeSwitchResponse(
        status="ok",
        message=(
            f"Mode switched from '{req.current_mode}' to '{req.new_mode}'. "
            "Trait synthesis complete."
        ),
        user_id=req.user_id,
        session_id=req.session_id,
        new_mode=req.new_mode,
    )
