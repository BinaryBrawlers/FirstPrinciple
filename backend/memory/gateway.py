# memory/gateway.py
#
# MemoryGateway — thin role-checking layer over the cognee API.
#
# The installed cognee version exposes this write API:
#   cognee.remember(data, dataset_name)  → add + cognify (permanent memory)
#   cognee.recall(query_text, datasets=) → search the knowledge graph
#   cognee.improve(dataset)              → enrich existing graph
#   cognee.forget(dataset=, ...)         → remove data
#
# "add_data_points" does not exist in this version; Track A ingestion goes
# through cognee.remember() with dataset_name="content_track".
# Track B writes use dataset_name=f"user_{user_id}_traits".
#
# Role-enforcement rules (unchanged from spec):
#   Track A (content_track)          → only INGESTION may write
#   Track B (user_*_traits)          → only TRAIT_SYNTHESIS may write
#
# The DATASET_NAME passed to each method determines which track is being
# targeted; the gateway checks it against the caller's role.

from enum import Enum
import cognee


class AgentRole(str, Enum):
    INGESTION       = "ingestion"
    TEACHER         = "teacher"
    INTERVIEWER     = "interviewer"
    TRAIT_SYNTHESIS = "trait_synthesis"


_TRACK_A_WRITERS = {AgentRole.INGESTION}
_TRACK_B_WRITERS = {AgentRole.TRAIT_SYNTHESIS}

CONTENT_TRACK = "content_track"


class MemoryAccessError(RuntimeError):
    pass


class MemoryGateway:
    def __init__(self, role: AgentRole):
        self._role = role

    # ------------------------------------------------------------------
    # Track A writes — dataset_name="content_track"
    # Replaces the old add_data_points(temporal_cognify=True) call.
    # ------------------------------------------------------------------
    async def add_data_points(self, data, *, temporal_cognify: bool = True):
        """Write HistoricalEpisode data to Track A (content_track).

        Uses cognee.remember() which runs add() + cognify() internally,
        equivalent to the old add_data_points(temporal_cognify=True).
        The `temporal_cognify` flag is accepted for API compatibility but
        cognee.remember() always cognifies; there is no separate flag.
        """
        if self._role not in _TRACK_A_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track A"
            )
        await cognee.remember(data, dataset_name=CONTENT_TRACK)

    # ------------------------------------------------------------------
    # Track B writes — dataset_name must start with "user_"
    # ------------------------------------------------------------------
    async def remember(self, graph_name: str, data, **kwargs):
        """Write a learner trait to Track B."""
        self._assert_track_b_writer(graph_name)
        await cognee.remember(data, dataset_name=graph_name)

    async def forget(self, graph_name: str, *args, **kwargs):
        """Remove a learner trait from Track B."""
        self._assert_track_b_writer(graph_name)
        await cognee.forget(dataset=graph_name, **kwargs)

    async def improve(self, graph_name: str, *args, **kwargs):
        """Update/enrich an existing learner trait in Track B."""
        self._assert_track_b_writer(graph_name)
        await cognee.improve(dataset=graph_name, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assert_track_b_writer(self, graph_name: str):
        if not graph_name.startswith("user_"):
            return  # not a Track B graph; no restriction
        if self._role not in _TRACK_B_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track B ({graph_name})"
            )
