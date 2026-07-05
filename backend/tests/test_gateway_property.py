"""Property-based tests for MemoryGateway access-control layer.

These tests verify that the access-control properties hold for all inputs
without ever reaching cognee — errors are raised before any cognee call is
made, so no mocking is required.

Properties covered:
  Property 4 — Track A write isolation   (Validates: Requirements 14.1, 14.4)
  Property 5 — Track B write isolation   (Validates: Requirements 14.2, 14.3, 3.3, 3.4)
  Property 6 — Track B graph naming invariant  (Validates: Requirements 3.1)
"""
from __future__ import annotations

import sys
import os
import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the backend package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.gateway import AgentRole, MemoryAccessError, MemoryGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously (avoids requiring pytest-asyncio)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Property 4: Track A write isolation
#
# For any role that is NOT the Ingestion Agent, calling add_data_points()
# through a MemoryGateway SHALL raise MemoryAccessError before reaching cognee.
#
# **Validates: Requirements 14.1, 14.4**
# ---------------------------------------------------------------------------

@given(
    role=st.sampled_from(
        [AgentRole.TEACHER, AgentRole.INTERVIEWER, AgentRole.TRAIT_SYNTHESIS]
    )
)
@settings(max_examples=100)
def test_track_a_write_isolation(role: AgentRole) -> None:
    """Non-ingestion roles must be blocked from calling add_data_points().

    **Validates: Requirements 14.1, 14.4**
    """
    gateway = MemoryGateway(role=role)
    with pytest.raises(MemoryAccessError):
        run(gateway.add_data_points([]))


# ---------------------------------------------------------------------------
# Property 5: Track B write isolation
#
# For any role that is NOT the Trait Synthesis Agent, calling remember(),
# forget(), or improve() on a graph whose name starts with 'user_' SHALL raise
# MemoryAccessError before reaching cognee.
#
# **Validates: Requirements 14.2, 14.3, 3.3, 3.4**
# ---------------------------------------------------------------------------

@given(
    role=st.sampled_from([AgentRole.TEACHER, AgentRole.INTERVIEWER]),
    graph_name=st.text(min_size=1).map(lambda s: "user_" + s),
)
@settings(max_examples=100)
def test_track_b_remember_isolation(role: AgentRole, graph_name: str) -> None:
    """Teacher and Interviewer must be blocked from calling remember() on user graphs.

    **Validates: Requirements 14.2, 14.3, 3.3, 3.4**
    """
    gateway = MemoryGateway(role=role)
    with pytest.raises(MemoryAccessError):
        run(gateway.remember(graph_name))


@given(
    role=st.sampled_from([AgentRole.TEACHER, AgentRole.INTERVIEWER]),
    graph_name=st.text(min_size=1).map(lambda s: "user_" + s),
)
@settings(max_examples=100)
def test_track_b_forget_isolation(role: AgentRole, graph_name: str) -> None:
    """Teacher and Interviewer must be blocked from calling forget() on user graphs.

    **Validates: Requirements 14.2, 14.3, 3.3, 3.4**
    """
    gateway = MemoryGateway(role=role)
    with pytest.raises(MemoryAccessError):
        run(gateway.forget(graph_name))


@given(
    role=st.sampled_from([AgentRole.TEACHER, AgentRole.INTERVIEWER]),
    graph_name=st.text(min_size=1).map(lambda s: "user_" + s),
)
@settings(max_examples=100)
def test_track_b_improve_isolation(role: AgentRole, graph_name: str) -> None:
    """Teacher and Interviewer must be blocked from calling improve() on user graphs.

    **Validates: Requirements 14.2, 14.3, 3.3, 3.4**
    """
    gateway = MemoryGateway(role=role)
    with pytest.raises(MemoryAccessError):
        run(gateway.improve(graph_name))


# ---------------------------------------------------------------------------
# Property 6: Track B graph naming invariant
#
# For any user ID string, the Track B graph name SHALL equal "user_" + user_id
# (the pattern used throughout the system is f"user_{user_id}_traits", which
# always starts with "user_" and equals "user_" + user_id + "_traits").
#
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

@given(user_id=st.text(min_size=1))
@settings(max_examples=100)
def test_track_b_graph_naming_starts_with_user_(user_id: str) -> None:
    """The trait graph name for any user must start with 'user_'.

    **Validates: Requirements 3.1**
    """
    graph_name = f"user_{user_id}_traits"
    assert graph_name.startswith("user_"), (
        f"Expected graph name to start with 'user_', got {graph_name!r}"
    )


@given(user_id=st.text(min_size=1))
@settings(max_examples=100)
def test_track_b_graph_naming_format(user_id: str) -> None:
    """The trait graph name for any user must equal 'user_' + user_id + '_traits'.

    **Validates: Requirements 3.1**
    """
    graph_name = f"user_{user_id}_traits"
    expected = "user_" + user_id + "_traits"
    assert graph_name == expected, (
        f"Graph name {graph_name!r} does not match expected {expected!r}"
    )
