"""Memory isolation layer for FirstPrinciple.

All cognee write operations are gated through MemoryGateway, which enforces
role-based access control before dispatching to the underlying cognee API.

Requirements: 14.1, 14.2, 14.3, 14.4
"""
from __future__ import annotations

from enum import Enum

import cognee  # type: ignore[import-untyped]


class AgentRole(str, Enum):
    INGESTION = "ingestion"
    TEACHER = "teacher"
    INTERVIEWER = "interviewer"
    TRAIT_SYNTHESIS = "trait_synthesis"


class MemoryAccessError(RuntimeError):
    """Raised when an agent attempts a write operation it is not permitted to perform."""


# Only the Ingestion Agent may write to Track A via add_data_points().
_TRACK_A_WRITERS = {AgentRole.INGESTION}

# Only the Trait Synthesis Agent may write to Track B graphs (user_* graph names).
_TRACK_B_WRITERS = {AgentRole.TRAIT_SYNTHESIS}


class MemoryGateway:
    """Access-controlled proxy for cognee write operations.

    Every agent instantiates a MemoryGateway with its own AgentRole.  All
    cognee write calls (add_data_points, remember, forget, improve) must go
    through this gateway; the gateway checks the caller's role before
    dispatching to cognee.

    Args:
        role: The AgentRole of the agent that owns this gateway instance.
    """

    def __init__(self, role: AgentRole) -> None:
        self._role = role

    # ------------------------------------------------------------------
    # Track A writes
    # ------------------------------------------------------------------

    async def add_data_points(self, data_points, *, temporal_cognify: bool = True):
        """Write data points to Track A (content_track).

        Raises:
            MemoryAccessError: If the gateway's role is not in _TRACK_A_WRITERS.
        """
        if self._role not in _TRACK_A_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role!r} is not permitted to call add_data_points() "
                f"(Track A write). Only {_TRACK_A_WRITERS} may write to Track A."
            )
        await cognee.add_data_points(data_points, temporal_cognify=temporal_cognify)

    # ------------------------------------------------------------------
    # Track B writes
    # ------------------------------------------------------------------

    async def remember(self, graph_name: str, *args, **kwargs):
        """Write a memory entry to the specified graph.

        If graph_name starts with 'user_', enforces that the role is a
        Track B writer.
        """
        self._assert_track_b_writer(graph_name)
        await cognee.remember(*args, graph_name=graph_name, **kwargs)

    async def forget(self, graph_name: str, *args, **kwargs):
        """Remove a memory entry from the specified graph.

        If graph_name starts with 'user_', enforces that the role is a
        Track B writer.
        """
        self._assert_track_b_writer(graph_name)
        await cognee.forget(*args, graph_name=graph_name, **kwargs)

    async def improve(self, graph_name: str, *args, **kwargs):
        """Update/improve an existing memory entry in the specified graph.

        If graph_name starts with 'user_', enforces that the role is a
        Track B writer.
        """
        self._assert_track_b_writer(graph_name)
        await cognee.improve(*args, graph_name=graph_name, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_track_b_writer(self, graph_name: str) -> None:
        """Raise MemoryAccessError if this role cannot write to the given Track B graph.

        Only graphs whose names start with 'user_' are considered Track B.
        Non-user graphs are unrestricted through this check.

        Args:
            graph_name: The cognee graph being targeted by the write operation.

        Raises:
            MemoryAccessError: If graph_name starts with 'user_' and the
                gateway's role is not in _TRACK_B_WRITERS.
        """
        if not graph_name.startswith("user_"):
            # Not a Track B graph; no restriction applied here.
            return
        if self._role not in _TRACK_B_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role!r} is not permitted to write to Track B graph "
                f"{graph_name!r}. Only {_TRACK_B_WRITERS} may write to Track B."
            )
