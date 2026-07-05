"""
LangGraph Chains — Chain 1 (ingestion) and Chain 2 (session-end write-back / trait synthesis).

Chain 1 wires:
  ingestion_node → teacher_node

Chain 2 wires:
  teacher_node     → trait_synthesis_node
  interviewer_node → trait_synthesis_node

Entry point for Chain 2 is set dynamically at invocation time (see invoke_chain2()).

LangGraph RetryPolicy: max_attempts=3, backoff_factor=2.0

Requirements: 8.2, 9.2
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy

# agents
from agents.ingestion import IngestionAgent
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


async def ingestion_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping the Ingestion Agent (Chain 1 entry point).

    Checks ``state["ingest_needed"]``. If True, instantiates IngestionAgent
    and delegates to ``IngestionAgent.run(topic)`` to decompose the topic,
    fetch content, and write HistoricalEpisodes to Track A memory.

    After a successful run ``ingest_needed`` is set to False so that
    downstream nodes and subsequent chain invocations can skip re-ingestion.

    Requirements: 9.2
    """
    if state.get("ingest_needed"):
        agent = IngestionAgent()
        await agent.run(state["topic"])
        return {**state, "ingest_needed": False}
    return state


async def teacher_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping the Teacher Agent.

    Chain 1 role: follows ingestion_node — by the time this node executes the
    topic has been ingested and the session can begin. No work is done here;
    the actual streaming is handled by the FastAPI SSE layer.

    Chain 2 role: represents the completion of a teacher session. The actual
    streaming has already happened; here we just pass state through so
    LangGraph can route to trait_synthesis_node.

    Requirements: 8.2, 9.2
    """
    # Teacher session streaming is handled by the FastAPI SSE layer.
    # This node satisfies LangGraph chain topology and carries the RetryPolicy.
    return state


async def interviewer_node(state: TutorState) -> TutorState:
    """
    LangGraph node wrapping the Interviewer Agent.

    Represents the completion of an interviewer session. The actual streaming
    has already happened; here we just pass state through so LangGraph can
    route to trait_synthesis_node.

    Requirements: 8.2
    """
    # Interviewer session streaming is handled by the FastAPI SSE layer.
    # This node satisfies LangGraph chain topology and carries the RetryPolicy.
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
# Chain 1 — topic ingestion
# Triggered when state["ingest_needed"] is True (new topic not yet in Track A)
# ---------------------------------------------------------------------------

def _build_chain1() -> StateGraph:
    """
    Build (but do not compile) the Chain 1 StateGraph.

    Edges:
      ingestion_node → teacher_node

    Returns an uncompiled graph; callers must call .compile() before use.

    Requirements: 9.2
    """
    graph = StateGraph(TutorState)

    graph.add_node("ingestion_node", ingestion_node, retry=retry)
    graph.add_node("teacher_node", teacher_node, retry=retry)

    graph.add_edge("ingestion_node", "teacher_node")
    graph.set_entry_point("ingestion_node")

    return graph


async def invoke_chain1(state: TutorState) -> TutorState:
    """
    Compile and invoke Chain 1 (ingestion_node → teacher_node).

    Should be called when ``state["ingest_needed"]`` is True and the topic
    has not yet been decomposed into HistoricalEpisodes.

    Args:
        state: The current TutorState.

    Returns:
        Updated TutorState with ``ingest_needed`` set to False after
        successful ingestion.

    Requirements: 9.2
    """
    graph = _build_chain1()
    compiled = graph.compile()

    logger.info(
        "invoke_chain1: invoking Chain 1 for topic=%r user=%r session=%r",
        state.get("topic"),
        state.get("user_id"),
        state.get("session_id"),
    )

    result: TutorState = await compiled.ainvoke(state)
    return result
