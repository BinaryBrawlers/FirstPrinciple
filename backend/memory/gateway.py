# memory/gateway.py

from enum import Enum
from typing import Callable
import cognee


class AgentRole(str, Enum):
    INGESTION       = "ingestion"
    TEACHER         = "teacher"
    INTERVIEWER     = "interviewer"
    TRAIT_SYNTHESIS = "trait_synthesis"


_TRACK_A_WRITERS = {AgentRole.INGESTION}
_TRACK_B_WRITERS = {AgentRole.TRAIT_SYNTHESIS}


class MemoryAccessError(RuntimeError):
    pass


class MemoryGateway:
    def __init__(self, role: AgentRole):
        self._role = role

    # --- Track A writes ---
    async def add_data_points(self, data_points, *, temporal_cognify: bool = True):
        if self._role not in _TRACK_A_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track A"
            )
        await cognee.add_data_points(data_points, temporal_cognify=temporal_cognify)

    # --- Track B writes ---
    async def remember(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.remember(*args, graph_name=graph_name, **kwargs)

    async def forget(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.forget(*args, graph_name=graph_name, **kwargs)

    async def improve(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.improve(*args, graph_name=graph_name, **kwargs)

    def _assert_track_b_writer(self, graph_name: str):
        if not graph_name.startswith("user_"):
            return  # not a Track B graph; no restriction
        if self._role not in _TRACK_B_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track B ({graph_name})"
            )
