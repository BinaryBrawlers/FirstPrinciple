"""Unit tests for narrative_sort() — topological ordering of HistoricalEpisode lists.

**Validates: Requirements 4.5**

The ``narrative_sort`` function must order episodes by their ``requires`` edges
(a topological sort over the dependency DAG) and use ``published_date`` only as a
tiebreaker when multiple episodes are simultaneously eligible for scheduling.

Key property tested: dates alone are intentionally in *wrong* order in the test
fixtures, which proves that the function does NOT use date as the primary ordering
criterion.
"""
from __future__ import annotations

import sys
import os
from collections import defaultdict, deque
from datetime import date
from typing import Optional

import pytest

# Ensure the backend package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence


# ---------------------------------------------------------------------------
# Standalone narrative_sort implementation
# (will be superseded by the real one from backend/agents/ingestion.py in
# task 5.3; kept here so tests can run independently of that module)
# ---------------------------------------------------------------------------

def narrative_sort(episodes: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
    """Return *episodes* sorted into narrative (topological) order.

    Algorithm: Kahn's BFS-based topological sort.

    - Primary criterion: ``requires`` edges — if episode B requires episode A,
      A must appear before B in the output.
    - Tiebreaker: when multiple episodes are simultaneously eligible (in-degree
      zero at the same step), order them by ``published_date`` ascending; episodes
      with no date sort last among ties (treated as ``date.max``).
    - Raises ``ValueError`` if the ``requires`` graph contains a cycle.

    Requirements: 4.5
    """
    if not episodes:
        return []

    id_to_episode: dict[str, HistoricalEpisode] = {ep.id: ep for ep in episodes}

    # Build the in-degree map and adjacency list for known episode IDs only.
    in_degree: dict[str, int] = {ep.id: 0 for ep in episodes}
    successors: dict[str, list[str]] = defaultdict(list)

    for ep in episodes:
        for req_id in ep.requires:
            if req_id in id_to_episode:
                # ep depends on req_id → req_id must come first
                in_degree[ep.id] += 1
                successors[req_id].append(ep.id)

    def _sort_key(ep_id: str) -> tuple:
        ep = id_to_episode[ep_id]
        d = ep.published_date if ep.published_date is not None else date.max
        return (d, ep_id)  # ep_id as secondary tiebreaker for stability

    # Start with all episodes that have no unsatisfied dependencies.
    ready: list[str] = sorted(
        [ep_id for ep_id, deg in in_degree.items() if deg == 0],
        key=_sort_key,
    )
    queue: deque[str] = deque(ready)

    result: list[HistoricalEpisode] = []

    while queue:
        current_id = queue.popleft()
        result.append(id_to_episode[current_id])

        # Collect newly eligible successors, sort them, then enqueue.
        newly_eligible: list[str] = []
        for successor_id in successors[current_id]:
            in_degree[successor_id] -= 1
            if in_degree[successor_id] == 0:
                newly_eligible.append(successor_id)

        for ep_id in sorted(newly_eligible, key=_sort_key):
            queue.append(ep_id)

    if len(result) != len(episodes):
        visited = {ep.id for ep in result}
        cycle_members = [ep.id for ep in episodes if ep.id not in visited]
        raise ValueError(
            f"narrative_sort: cycle detected among episode IDs: {cycle_members}"
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_topologically_valid(
    ordered: list[HistoricalEpisode],
    all_ids: set[str],
) -> bool:
    """Return True iff for every episode B in *ordered*, all episodes that B
    requires (and that exist in *all_ids*) appear before B.
    """
    seen: set[str] = set()
    for ep in ordered:
        for req_id in ep.requires:
            if req_id in all_ids and req_id not in seen:
                return False
        seen.add(ep.id)
    return True


def make_episode(
    ep_id: str,
    requires: list[str] | None = None,
    published_date: date | None = None,
    outcome: Outcome = Outcome.SUCCESS,
) -> HistoricalEpisode:
    """Convenience factory for concise test episode construction."""
    return HistoricalEpisode(
        id=ep_id,
        concept=ep_id.replace("_", " ").title(),
        problem_posed=f"Problem posed for {ep_id}",
        attempted_solution=f"Solution attempted for {ep_id}",
        outcome=outcome,
        why=f"Explanation for {ep_id}",
        requires=requires or [],
        source_confidence=SourceConfidence.REASONED,
        published_date=published_date,
    )


# ---------------------------------------------------------------------------
# Fixture: a small DAG where dates are intentionally in the WRONG topological
# order.  Date order alone would give: D → C → B → A (newest first would be
# wrong; oldest first = A→B→C→D but that also happens to be correct here).
#
# We build a more interesting fixture where date order conflicts with requires:
#
#   Episode IDs and their dates (most recent first means wrong dependency order):
#     ep_a  requires=[]        date=2020  (root — should come first)
#     ep_b  requires=[ep_a]    date=2015  (OLDER than ep_a — date suggests b before a)
#     ep_c  requires=[ep_a]    date=2010  (even older — date suggests c before a)
#     ep_d  requires=[ep_b, ep_c]  date=2025 (newest — must come last)
#
#   Chronological (date ASC) order: ep_c(2010) → ep_b(2015) → ep_a(2020) → ep_d(2025)
#   That order is INVALID: ep_c and ep_b appear before ep_a, which they require.
#
#   Correct topological order: ep_a first, then ep_b and ep_c (either order),
#   then ep_d.
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_date_episodes() -> list[HistoricalEpisode]:
    """Four episodes with dates intentionally violating topological order."""
    return [
        make_episode("ep_a", requires=[],             published_date=date(2020, 1, 1)),
        make_episode("ep_b", requires=["ep_a"],       published_date=date(2015, 6, 1)),
        make_episode("ep_c", requires=["ep_a"],       published_date=date(2010, 3, 1)),
        make_episode("ep_d", requires=["ep_b", "ep_c"], published_date=date(2025, 9, 1)),
    ]


# ---------------------------------------------------------------------------
# Test: empty list
# ---------------------------------------------------------------------------

def test_narrative_sort_empty_list() -> None:
    """narrative_sort on an empty list returns an empty list."""
    assert narrative_sort([]) == []


# ---------------------------------------------------------------------------
# Test: single episode
# ---------------------------------------------------------------------------

def test_narrative_sort_single_episode() -> None:
    """narrative_sort on a single episode returns that episode unchanged."""
    ep = make_episode("solo", published_date=date(2000, 1, 1))
    result = narrative_sort([ep])
    assert len(result) == 1
    assert result[0].id == "solo"


# ---------------------------------------------------------------------------
# Test: topological validity on mixed-date fixture
# ---------------------------------------------------------------------------

def test_narrative_sort_produces_topologically_valid_order(
    mixed_date_episodes: list[HistoricalEpisode],
) -> None:
    """narrative_sort must produce a topologically valid order.

    For every episode in the result, every episode it requires must appear
    before it.

    **Validates: Requirements 4.5**
    """
    sorted_episodes = narrative_sort(mixed_date_episodes)
    all_ids = {ep.id for ep in mixed_date_episodes}

    assert is_topologically_valid(sorted_episodes, all_ids), (
        f"Topological invariant violated. Order: {[ep.id for ep in sorted_episodes]}"
    )


# ---------------------------------------------------------------------------
# Test: dates are NOT the primary sort key (core correctness check)
# ---------------------------------------------------------------------------

def test_narrative_sort_dates_not_primary_criterion(
    mixed_date_episodes: list[HistoricalEpisode],
) -> None:
    """Date order alone would place ep_c(2010) before ep_a(2020), violating
    the requires edge ep_c → ep_a.  narrative_sort must put ep_a first.

    This test directly proves that dates are not the primary ordering criterion.

    **Validates: Requirements 4.5**
    """
    sorted_episodes = narrative_sort(mixed_date_episodes)
    sorted_ids = [ep.id for ep in sorted_episodes]

    # ep_a must come before ep_b (ep_b requires ep_a)
    assert sorted_ids.index("ep_a") < sorted_ids.index("ep_b"), (
        f"ep_a must precede ep_b (requires edge). Got order: {sorted_ids}"
    )

    # ep_a must come before ep_c (ep_c requires ep_a), even though ep_c is older
    assert sorted_ids.index("ep_a") < sorted_ids.index("ep_c"), (
        f"ep_a must precede ep_c (requires edge) even though ep_c has an earlier "
        f"published_date. Got order: {sorted_ids}"
    )

    # ep_d must come last (requires both ep_b and ep_c)
    assert sorted_ids.index("ep_b") < sorted_ids.index("ep_d"), (
        f"ep_b must precede ep_d. Got order: {sorted_ids}"
    )
    assert sorted_ids.index("ep_c") < sorted_ids.index("ep_d"), (
        f"ep_c must precede ep_d. Got order: {sorted_ids}"
    )


# ---------------------------------------------------------------------------
# Test: date is used as tiebreaker among simultaneously eligible episodes
# ---------------------------------------------------------------------------

def test_narrative_sort_date_used_as_tiebreaker() -> None:
    """When multiple episodes are simultaneously eligible (no unsatisfied
    dependencies), the one with the earlier published_date should come first.

    **Validates: Requirements 4.5**
    """
    # ep_root has no requirements; ep_early and ep_late both require ep_root.
    # ep_early has an earlier date so it should appear before ep_late.
    ep_root  = make_episode("ep_root",  requires=[],           published_date=date(2000, 1, 1))
    ep_early = make_episode("ep_early", requires=["ep_root"],  published_date=date(2005, 1, 1))
    ep_late  = make_episode("ep_late",  requires=["ep_root"],  published_date=date(2010, 1, 1))

    # Pass them in *reversed* date order to make sure it's not just input-order
    result = narrative_sort([ep_root, ep_late, ep_early])
    result_ids = [ep.id for ep in result]

    assert result_ids[0] == "ep_root", f"ep_root must be first. Got: {result_ids}"
    assert result_ids.index("ep_early") < result_ids.index("ep_late"), (
        f"ep_early (earlier date) must precede ep_late when both are eligible. "
        f"Got: {result_ids}"
    )


# ---------------------------------------------------------------------------
# Test: episodes without published_date sort after dated episodes in ties
# ---------------------------------------------------------------------------

def test_narrative_sort_no_date_sorts_after_dated_episodes() -> None:
    """An episode with no published_date should sort after dated episodes when
    both are simultaneously eligible (no dependencies between them).

    **Validates: Requirements 4.5**
    """
    ep_root  = make_episode("ep_root",  requires=[],          published_date=date(2000, 1, 1))
    ep_dated = make_episode("ep_dated", requires=["ep_root"],  published_date=date(2010, 1, 1))
    ep_none  = make_episode("ep_none",  requires=["ep_root"],  published_date=None)

    result = narrative_sort([ep_root, ep_none, ep_dated])
    result_ids = [ep.id for ep in result]

    assert result_ids[0] == "ep_root", f"ep_root must be first. Got: {result_ids}"
    assert result_ids.index("ep_dated") < result_ids.index("ep_none"), (
        f"ep_dated must precede ep_none (no date) among simultaneous candidates. "
        f"Got: {result_ids}"
    )


# ---------------------------------------------------------------------------
# Test: linear chain
# ---------------------------------------------------------------------------

def test_narrative_sort_linear_chain() -> None:
    """A strictly linear requires chain A→B→C→D must be returned in that exact
    order, regardless of the episode input order.

    **Validates: Requirements 4.5**
    """
    ep_a = make_episode("ep_a", requires=[],       published_date=date(2000, 1, 1))
    ep_b = make_episode("ep_b", requires=["ep_a"], published_date=date(2001, 1, 1))
    ep_c = make_episode("ep_c", requires=["ep_b"], published_date=date(2002, 1, 1))
    ep_d = make_episode("ep_d", requires=["ep_c"], published_date=date(2003, 1, 1))

    # Shuffle input order to confirm it's not just identity
    shuffled = [ep_d, ep_b, ep_a, ep_c]
    result = narrative_sort(shuffled)
    result_ids = [ep.id for ep in result]

    assert result_ids == ["ep_a", "ep_b", "ep_c", "ep_d"], (
        f"Linear chain must be returned in dependency order. Got: {result_ids}"
    )


# ---------------------------------------------------------------------------
# Test: cycle detection raises ValueError
# ---------------------------------------------------------------------------

def test_narrative_sort_raises_on_cycle() -> None:
    """narrative_sort must raise ValueError when the requires graph contains a
    cycle, rather than silently returning a partial result.

    **Validates: Requirements 4.5**
    """
    ep_x = make_episode("ep_x", requires=["ep_z"])
    ep_y = make_episode("ep_y", requires=["ep_x"])
    ep_z = make_episode("ep_z", requires=["ep_y"])

    with pytest.raises(ValueError, match="cycle"):
        narrative_sort([ep_x, ep_y, ep_z])


# ---------------------------------------------------------------------------
# Test: external requires IDs (not in the episode list) are ignored gracefully
# ---------------------------------------------------------------------------

def test_narrative_sort_ignores_external_requires() -> None:
    """If an episode requires an ID that is not in the input list (e.g. it was
    already ingested earlier), that dependency should be ignored — the episode
    is treated as having no unsatisfied dependencies with respect to the
    current batch.

    **Validates: Requirements 4.5**
    """
    # ep_b requires ep_ext which is NOT in the list → treated as satisfied
    ep_a = make_episode("ep_a", requires=[],                    published_date=date(2000, 1, 1))
    ep_b = make_episode("ep_b", requires=["ep_ext", "ep_a"],   published_date=date(2001, 1, 1))

    result = narrative_sort([ep_b, ep_a])
    result_ids = [ep.id for ep in result]

    # ep_a must precede ep_b (ep_b requires ep_a which IS in the list)
    assert result_ids.index("ep_a") < result_ids.index("ep_b"), (
        f"ep_a must precede ep_b. Got: {result_ids}"
    )
    # Both episodes must be returned
    assert set(result_ids) == {"ep_a", "ep_b"}
