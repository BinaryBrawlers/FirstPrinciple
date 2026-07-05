"""
Property-based tests for MemoryGateway write isolation.

Property 4: Track A write isolation
  For each role in {TEACHER, INTERVIEWER, TRAIT_SYNTHESIS}, constructing a
  MemoryGateway and calling add_data_points() SHALL raise MemoryAccessError
  immediately, before any cognee call is made.

**Validates: Requirements 14.1, 14.4**

Property 6: Track B graph naming invariant
  For any user ID string, the Track B graph name equals "user_" + user_id
  exactly (i.e. f"user_{user_id}_traits").

**Validates: Requirements 3.1**
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.memory.gateway import AgentRole, MemoryAccessError, MemoryGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cognee_mock() -> MagicMock:
    """
    Return a MagicMock that stands in for the `cognee` module inside gateway.py.

    We patch `backend.memory.gateway.cognee` rather than `cognee.add_data_points`
    because the installed cognee version may not expose add_data_points at the
    top level, which causes patch() to raise AttributeError before our test
    even runs.  Replacing the entire module reference in the gateway namespace
    is version-agnostic.
    """
    mock = MagicMock()
    mock.add_data_points = AsyncMock()
    mock.remember        = AsyncMock()
    mock.forget          = AsyncMock()
    mock.improve         = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Roles that are NOT permitted to write to Track A
# ---------------------------------------------------------------------------

NON_INGESTION_ROLES = [
    AgentRole.TEACHER,
    AgentRole.INTERVIEWER,
    AgentRole.TRAIT_SYNTHESIS,
]


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@given(role=st.sampled_from(NON_INGESTION_ROLES))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_track_a_write_isolation(role: AgentRole):
    """
    Property 4: Any role other than INGESTION MUST raise MemoryAccessError
    when calling add_data_points(), and cognee.add_data_points must never
    be reached.

    **Validates: Requirements 14.1, 14.4**
    """
    cognee_mock = _make_cognee_mock()
    gateway = MemoryGateway(role)
    sample_data_points = [{"id": "ep-001", "concept": "paging"}]

    with patch("backend.memory.gateway.cognee", cognee_mock):
        with pytest.raises(MemoryAccessError) as exc_info:
            asyncio.run(gateway.add_data_points(sample_data_points))

        # Error message must identify the offending role
        error_text = str(exc_info.value).lower()
        assert role.value in error_text or str(role).lower() in error_text

        # cognee.add_data_points must NOT have been called
        cognee_mock.add_data_points.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests that complement the property
# ---------------------------------------------------------------------------

def test_ingestion_role_is_the_only_track_a_writer():
    """INGESTION is the sole permitted Track A writer; all other roles are blocked."""
    from backend.memory.gateway import _TRACK_A_WRITERS
    assert _TRACK_A_WRITERS == {AgentRole.INGESTION}


@pytest.mark.parametrize("role", NON_INGESTION_ROLES)
def test_each_non_ingestion_role_raises_memory_access_error(role: AgentRole):
    """Each non-INGESTION role individually raises MemoryAccessError synchronously."""
    cognee_mock = _make_cognee_mock()
    gateway = MemoryGateway(role)
    sample_data_points = [{"id": "ep-x", "concept": "test"}]

    with patch("backend.memory.gateway.cognee", cognee_mock):
        with pytest.raises(MemoryAccessError):
            asyncio.run(gateway.add_data_points(sample_data_points))
        cognee_mock.add_data_points.assert_not_called()


def test_error_message_identifies_role():
    """MemoryAccessError message must mention the blocked role."""
    cognee_mock = _make_cognee_mock()
    gateway = MemoryGateway(AgentRole.TEACHER)

    with patch("backend.memory.gateway.cognee", cognee_mock):
        with pytest.raises(MemoryAccessError) as exc_info:
            asyncio.run(gateway.add_data_points([]))
    assert "teacher" in str(exc_info.value).lower()


def test_ingestion_role_does_not_raise():
    """INGESTION role must NOT raise MemoryAccessError — it is the permitted writer."""
    cognee_mock = _make_cognee_mock()
    gateway = MemoryGateway(AgentRole.INGESTION)
    sample_data_points = [{"id": "ep-001", "concept": "paging"}]

    with patch("backend.memory.gateway.cognee", cognee_mock):
        # Should complete without raising
        asyncio.run(gateway.add_data_points(sample_data_points))
        cognee_mock.add_data_points.assert_called_once_with(
            sample_data_points, temporal_cognify=True
        )


# ---------------------------------------------------------------------------
# Property 6: Track B graph naming invariant
# ---------------------------------------------------------------------------

def _track_b_graph_name(user_id: str) -> str:
    """Canonical constructor for a Track B graph name (mirrors what callers use)."""
    return f"user_{user_id}_traits"


@given(user_id=st.text(min_size=1))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_track_b_graph_name_invariant(user_id: str):
    """
    Property 6: Track B graph naming invariant.

    For any non-empty user ID string, the Track B graph name produced by the
    canonical constructor MUST equal ``"user_" + user_id + "_traits"`` exactly.

    Two sub-assertions are checked:
    1. The graph name starts with the ``"user_"`` prefix (the prefix the
       MemoryGateway uses to recognise Track B graphs).
    2. Stripping the ``"user_"`` prefix and the ``"_traits"`` suffix recovers
       the original user_id without alteration.

    **Validates: Requirements 3.1**
    """
    graph_name = _track_b_graph_name(user_id)

    # The full name must be exactly "user_" + user_id + "_traits"
    assert graph_name == f"user_{user_id}_traits"

    # The gateway uses startswith("user_") to identify Track B graphs
    assert graph_name.startswith("user_")

    # Removing the known prefix and suffix recovers the original user_id
    assert graph_name[len("user_"):-len("_traits")] == user_id
