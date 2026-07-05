"""
LangGraph Chains — Chain 2 (session-end write-back / trait synthesis).

Chain 1 (ingestion_node → teacher_node) is a future task (Task 11) and is
stubbed here for completeness but NOT compiled or invoked.

Chain 2 wires:
  teacher_node   → trait_synthesis_node
  interviewer_node → trait_synthesis_node

Entry point is set dynamically at invocation time (see invoke_chain2()).

LangGraph RetryPolicy: max_attempts=3, backoff_factor=2.0

Requirements: 8.2
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy

# agents — trait_synthesis must exist; teacher/interviewer are already present
from agents.trait_synthesis import trait_synthesis_agent
from models.schemas import TutorState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RetryPolicy shared across all nodes
# ---------------------------------------------------------------------------

retry = RetryPolicy(max_attempts=3, backoff_factor=2.0)

# ---------------------------------------------------------------------------
# Node wrappers  (LangGraph nodes must accept state and return state)
# ---------------------------------------------------------------------------


async def teacher_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping the Teacher Agent.

    In Chain 2 this node represents the completion of a teacher session.
    The actual streaming has already happened; here we just pass state through
    so LangGraph can route to trait_synthesis_node.

    Requirements: 8.2
    """
    # Teacher session work already done by the time Chain 2 fires.
    # This node is the "after teacher session end" trigger point.
    return state


async def interviewer_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping the Interviewer Agent.

    In Chain 2 this node represents the completion of an interviewer session.
    The actual streaming has already happened; here we just pass state through
    so LangGraph can route to trait_synthesis_node.

    Requirements: 8.2
    """
    # Interviewer session work already done by the time Chain 2 fires.
    # This node is the "after interviewer session end" trigger point.
    return state


async def trait_synthesis_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping trait_synthesis_agent.

    Calls the agent (side-effects only — Track B writes) and returns state
    unchanged.

    Requirements: 8.2, 8.3, 8.4, 8.5
    """
    await trait_synthesis_agent(state)
    return state


# ---------------------------------------------------------------------------
# Chain 2 — session-end write-back
# Trigger points:
#   1. After Teacher session end:    entry_point="teacher_node"
#   2. After Interviewer session end: entry_point="interviewer_node"
#   3. On mode switch:               entry_point varies (see invoke_chain2)
# ---------------------------------------------------------------------------

def _build_chain2() -> StateGraph:
    """
    Build (but do not compile) the Chain 2 StateGraph.

    Edges:
      teacher_node    → trait_synthesis_node
      interviewer_node → trait_synthesis_node

    Returns an uncompiled graph; callers must call .compile() before use.
    """
    graph = StateGraph(TutorState)

    graph.add_node("teacher_node", teacher_node, retry=retry)
    graph.add_node("interviewer_node", interviewer_node, retry=retry)
    graph.add_node("trait_synthesis_node", trait_synthesis_node, retry=retry)

    # Trigger 1: Teacher session end → trait synthesis
    graph.add_edge("teacher_node", "trait_synthesis_node")

    # Trigger 2: Interviewer session end → trait synthesis
    graph.add_edge("interviewer_node", "trait_synthesis_node")

    return graph


async def invoke_chain2(
    state: TutorState,
    entry_point: str = "teacher_node",
) -> TutorState:
    """
    Compile and invoke Chain 2 with the given *entry_point*.

    Entry point choices:
      "teacher_node"    — after Teacher session ends  (Trigger 1)
      "interviewer_node" — after Interviewer session ends  (Trigger 2)

    For Trigger 3 (mode switch), the application layer calls this function
    with the appropriate entry point based on the mode being switched away from.

    Args:
        state:        The current TutorState.
        entry_point:  Which node begins execution; must be one of
                      "teacher_node" or "interviewer_node".

    Returns:
        Updated TutorState (trait synthesis modifies Track B only; state is
        returned unchanged from trait_synthesis_node).

    Requirements: 8.2
    """
    valid_entry_points = {"teacher_node", "interviewer_node"}
    if entry_point not in valid_entry_points:
        raise ValueError(
            f"invoke_chain2: invalid entry_point {entry_point!r}. "
            f"Must be one of {valid_entry_points}."
        )

    graph = _build_chain2()
    graph.set_entry_point(entry_point)
    compiled = graph.compile()

    logger.info(
        "invoke_chain2: invoking Chain 2 with entry_point=%r for user=%r session=%r",
        entry_point,
        state.get("user_id"),
        state.get("session_id"),
    )

    result: TutorState = await compiled.ainvoke(state)
    return result


# ---------------------------------------------------------------------------
# Chain 1 stub — topic ingestion (Task 11, not yet implemented)
# ---------------------------------------------------------------------------

def _build_chain1_stub() -> None:
    """
    Placeholder for Chain 1 (ingestion_node → teacher_node).

    Implemented in Task 11. Listed here for architectural completeness.
    """
    # chain1 = StateGraph(TutorState)
    # chain1.add_node("ingestion_node", ingestion_agent, retry=retry)
    # chain1.add_node("teacher_node",   teacher_agent,   retry=retry)
    # chain1.add_edge("ingestion_node", "teacher_node")
    # chain1.set_entry_point("ingestion_node")
    # return chain1.compile()
    pass
