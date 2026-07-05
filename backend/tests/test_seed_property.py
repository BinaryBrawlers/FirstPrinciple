"""Property-based test for seed idempotency.

**Property 16: Seed topics are complete and non-duplicated**

For any number of times ``seed_tracks_if_absent()`` is called, Track A SHALL
contain exactly the hand-authored episodes for OS memory management and deep
learning, with no duplicate episode IDs.

**Validates: Requirements 10.1, 10.2, 10.3**
"""
from __future__ import annotations

import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure the backend package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.seed import (
    seed_tracks_if_absent,
    OS_MEMORY_EPISODES,
    DEEP_LEARNING_EPISODES,
)
from models.schemas import HistoricalEpisode


# ---------------------------------------------------------------------------
# Expected episode IDs (derived directly from the seed data constants)
# ---------------------------------------------------------------------------

EXPECTED_OS_IDS: list[str] = [ep.id for ep in OS_MEMORY_EPISODES]
EXPECTED_DL_IDS: list[str] = [ep.id for ep in DEEP_LEARNING_EPISODES]
ALL_EXPECTED_IDS: set[str] = set(EXPECTED_OS_IDS) | set(EXPECTED_DL_IDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously (avoids requiring pytest-asyncio)."""
    return asyncio.run(coro)


def _make_cognee_mock(recall_empty: bool):
    """Return a mock cognee module.

    Args:
        recall_empty: When True, ``cognee.recall()`` returns an empty list
            (topic absent). When False, it returns a non-empty sentinel list
            (topic already present — seed should be skipped).
    """
    mock = MagicMock()

    recall_return = [] if recall_empty else [{"id": "sentinel"}]
    mock.recall = AsyncMock(return_value=recall_return)
    mock.add_data_points = AsyncMock(return_value=None)
    mock.consolidate_entity_descriptions_pipeline = AsyncMock(return_value=None)

    return mock


# ---------------------------------------------------------------------------
# Utility: run seed_tracks_if_absent() with a controlled cognee mock and
# collect all data-point lists that were written.
# ---------------------------------------------------------------------------

def _run_seed_with_mock(cognee_mock) -> list[HistoricalEpisode]:
    """Invoke ``seed_tracks_if_absent()`` with the given cognee mock and return
    the flat list of all HistoricalEpisode objects that were passed to
    ``add_data_points``."""
    # We need to patch cognee inside the seed module so that both the
    # ``MemoryGateway.add_data_points()`` call and the direct cognee.recall /
    # cognee.consolidate_entity_descriptions_pipeline calls all use our mock.
    with patch("memory.seed.cognee", cognee_mock), \
         patch("memory.gateway.cognee", cognee_mock):
        run(seed_tracks_if_absent())

    # Collect every batch of episodes passed to add_data_points
    written: list[HistoricalEpisode] = []
    for call_args in cognee_mock.add_data_points.call_args_list:
        batch = call_args.args[0]  # first positional arg is the episodes list
        written.extend(batch)
    return written


# ---------------------------------------------------------------------------
# Property 16: Seed topics are complete and non-duplicated
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(num_additional_calls=st.integers(min_value=0, max_value=3))
def test_seed_idempotency_no_duplicates(num_additional_calls: int) -> None:
    """Calling seed_tracks_if_absent() repeatedly must not produce duplicate IDs.

    Strategy:
    - First call: cognee.recall() returns [] (absent) → episodes are written.
    - Subsequent calls: cognee.recall() returns a non-empty list (present) →
      episodes are NOT written again.

    We verify that the union of all written episodes contains no duplicate IDs.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    all_written: list[HistoricalEpisode] = []

    # First call: topics absent — episodes should be written
    first_mock = _make_cognee_mock(recall_empty=True)
    all_written.extend(_run_seed_with_mock(first_mock))

    # Additional calls: topics present — no new episodes should be written
    for _ in range(num_additional_calls):
        subsequent_mock = _make_cognee_mock(recall_empty=False)
        all_written.extend(_run_seed_with_mock(subsequent_mock))

    written_ids = [ep.id for ep in all_written]

    # No duplicates
    assert len(written_ids) == len(set(written_ids)), (
        f"Duplicate episode IDs found after {1 + num_additional_calls} call(s): "
        f"{[eid for eid in written_ids if written_ids.count(eid) > 1]}"
    )


@settings(max_examples=10)
@given(st.just(None))  # single-case parametrised; Hypothesis drives reruns
def test_seed_first_call_writes_all_expected_episodes(_: None) -> None:
    """On first call (Track A empty) all expected episode IDs must be written.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    mock = _make_cognee_mock(recall_empty=True)
    written = _run_seed_with_mock(mock)

    written_ids = {ep.id for ep in written}

    missing = ALL_EXPECTED_IDS - written_ids
    assert not missing, (
        f"Missing episode IDs after first seed call: {missing}"
    )


@settings(max_examples=10)
@given(st.just(None))
def test_seed_subsequent_call_writes_nothing(_: None) -> None:
    """On subsequent calls (Track A already populated) nothing should be written.

    **Validates: Requirements 10.3**
    """
    mock = _make_cognee_mock(recall_empty=False)
    written = _run_seed_with_mock(mock)

    assert written == [], (
        f"Expected no writes on a subsequent seed call, but got: "
        f"{[ep.id for ep in written]}"
    )


@settings(max_examples=10)
@given(st.just(None))
def test_seed_first_call_contains_os_episodes_in_order(_: None) -> None:
    """The OS Memory Management episodes must be written in the declared order.

    **Validates: Requirements 10.1**
    """
    mock = _make_cognee_mock(recall_empty=True)
    written = _run_seed_with_mock(mock)

    # Filter to OS episodes only (preserving insertion order)
    os_written = [ep for ep in written if ep.id in set(EXPECTED_OS_IDS)]
    os_written_ids = [ep.id for ep in os_written]

    assert os_written_ids == EXPECTED_OS_IDS, (
        f"OS episodes written in unexpected order.\n"
        f"Expected: {EXPECTED_OS_IDS}\n"
        f"Got:      {os_written_ids}"
    )


@settings(max_examples=10)
@given(st.just(None))
def test_seed_first_call_contains_dl_episodes_in_order(_: None) -> None:
    """The Deep Learning episodes must be written in the declared order.

    **Validates: Requirements 10.2**
    """
    mock = _make_cognee_mock(recall_empty=True)
    written = _run_seed_with_mock(mock)

    # Filter to DL episodes only (preserving insertion order)
    dl_written = [ep for ep in written if ep.id in set(EXPECTED_DL_IDS)]
    dl_written_ids = [ep.id for ep in dl_written]

    assert dl_written_ids == EXPECTED_DL_IDS, (
        f"Deep Learning episodes written in unexpected order.\n"
        f"Expected: {EXPECTED_DL_IDS}\n"
        f"Got:      {dl_written_ids}"
    )


@settings(max_examples=50)
@given(num_calls=st.integers(min_value=1, max_value=5))
def test_seed_total_unique_ids_always_equals_expected(num_calls: int) -> None:
    """After N calls the set of unique written IDs equals ALL_EXPECTED_IDS exactly.

    - Call 1 is done with an empty Track A (recall returns []).
    - Calls 2..N are done with a populated Track A (recall returns non-empty).

    The union of all written IDs across all calls must equal ALL_EXPECTED_IDS
    with no extras and no missing entries.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    all_written: list[HistoricalEpisode] = []

    # First call always writes
    first_mock = _make_cognee_mock(recall_empty=True)
    all_written.extend(_run_seed_with_mock(first_mock))

    # Remaining calls are no-ops
    for _ in range(num_calls - 1):
        subsequent_mock = _make_cognee_mock(recall_empty=False)
        all_written.extend(_run_seed_with_mock(subsequent_mock))

    unique_written_ids = {ep.id for ep in all_written}

    extra = unique_written_ids - ALL_EXPECTED_IDS
    missing = ALL_EXPECTED_IDS - unique_written_ids

    assert not extra, f"Unexpected episode IDs written: {extra}"
    assert not missing, f"Expected episode IDs not written: {missing}"
