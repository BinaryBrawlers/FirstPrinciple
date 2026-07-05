"""
Property-based tests for HistoricalEpisode schema completeness.

Property 1: HistoricalEpisode schema completeness
  For any object that the Ingestion Agent writes to Track A, it SHALL contain
  all required fields (id, concept, problem_posed, attempted_solution, outcome,
  why, requires, concurrent_with, source_confidence) with values conforming to
  their declared types and enumerations.

Validates: Requirements 1.1
"""

import dataclasses
from datetime import date
from typing import Optional

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.models.schemas import HistoricalEpisode, Outcome, SourceConfidence


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty strings for required string fields
nonempty_text = st.text(min_size=1, max_size=200)

# Optional strings and dates
optional_text = st.one_of(st.none(), nonempty_text)
optional_date = st.one_of(st.none(), st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31)))

# A list-of-episode-IDs strategy (may be empty)
id_list = st.lists(nonempty_text, max_size=5)

# Strategy that builds a valid HistoricalEpisode from arbitrary data
valid_episode = st.builds(
    HistoricalEpisode,
    id=nonempty_text,
    concept=nonempty_text,
    problem_posed=nonempty_text,
    attempted_solution=nonempty_text,
    outcome=st.sampled_from(Outcome),
    why=nonempty_text,
    requires=id_list,
    concurrent_with=id_list,
    source_confidence=st.sampled_from(SourceConfidence),
    source=optional_text,
    published_date=optional_date,
)


# ---------------------------------------------------------------------------
# Required fields and their expected types (Requirement 1.1)
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id":                 str,
    "concept":            str,
    "problem_posed":      str,
    "attempted_solution": str,
    "outcome":            Outcome,
    "why":                str,
    "requires":           list,
    "concurrent_with":    list,
    "source_confidence":  SourceConfidence,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_episode_schema(episode: HistoricalEpisode) -> None:
    """Assert all required fields are present with correct types/enum values."""
    field_names = {f.name for f in dataclasses.fields(episode)}

    for field_name, expected_type in REQUIRED_FIELDS.items():
        # Field must exist on the dataclass
        assert field_name in field_names, (
            f"Required field '{field_name}' is missing from HistoricalEpisode"
        )
        value = getattr(episode, field_name)

        # Value must not be None for required fields
        assert value is not None, (
            f"Required field '{field_name}' must not be None"
        )

        # Value must match its declared type
        assert isinstance(value, expected_type), (
            f"Field '{field_name}' has value {value!r} of type "
            f"{type(value).__name__}, expected {expected_type}"
        )

    # requires and concurrent_with must contain only strings
    for list_field in ("requires", "concurrent_with"):
        lst = getattr(episode, list_field)
        assert all(isinstance(item, str) for item in lst), (
            f"All items in '{list_field}' must be strings"
        )

    # outcome must be a valid Outcome enum member
    assert episode.outcome in Outcome, (
        f"outcome {episode.outcome!r} is not a valid Outcome member"
    )

    # source_confidence must be a valid SourceConfidence enum member
    assert episode.source_confidence in SourceConfidence, (
        f"source_confidence {episode.source_confidence!r} is not a valid "
        "SourceConfidence member"
    )

    # Optional fields: if present, must match expected types
    if episode.source is not None:
        assert isinstance(episode.source, str), (
            f"'source' must be str when set, got {type(episode.source).__name__}"
        )
    if episode.published_date is not None:
        assert isinstance(episode.published_date, date), (
            f"'published_date' must be date when set, "
            f"got {type(episode.published_date).__name__}"
        )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@given(episode=valid_episode)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_historical_episode_schema_completeness(episode: HistoricalEpisode):
    """
    Property 1: Any HistoricalEpisode instance must have all required fields
    present and each value must match its declared type or enum.
    """
    _check_episode_schema(episode)


# ---------------------------------------------------------------------------
# Edge-case unit tests that complement the property
# ---------------------------------------------------------------------------

def test_outcome_enum_values():
    """Outcome enum must define exactly the three canonical values."""
    assert set(Outcome) == {Outcome.SUCCESS, Outcome.FAILURE, Outcome.PARTIAL}
    assert Outcome.SUCCESS.value == "success"
    assert Outcome.FAILURE.value == "failure"
    assert Outcome.PARTIAL.value == "partial"


def test_source_confidence_enum_values():
    """SourceConfidence enum must define exactly the three confidence tiers."""
    assert set(SourceConfidence) == {
        SourceConfidence.CITED_SOURCE,
        SourceConfidence.NAMED_REFERENCE,
        SourceConfidence.REASONED,
    }
    assert SourceConfidence.CITED_SOURCE.value    == "cited_source"
    assert SourceConfidence.NAMED_REFERENCE.value == "named_reference"
    assert SourceConfidence.REASONED.value        == "reasoned"


def test_default_source_confidence_is_reasoned():
    """A HistoricalEpisode constructed without an explicit source_confidence
    must default to SourceConfidence.REASONED (Requirement 1.4 guard)."""
    ep = HistoricalEpisode(
        id="ep-001",
        concept="test concept",
        problem_posed="what?",
        attempted_solution="try this",
        outcome=Outcome.SUCCESS,
        why="because",
    )
    assert ep.source_confidence is SourceConfidence.REASONED


def test_requires_and_concurrent_with_default_to_empty_list():
    """Mutable defaults must be independent lists (not shared)."""
    ep1 = HistoricalEpisode(
        id="ep-a", concept="c", problem_posed="p",
        attempted_solution="s", outcome=Outcome.SUCCESS, why="w",
    )
    ep2 = HistoricalEpisode(
        id="ep-b", concept="c", problem_posed="p",
        attempted_solution="s", outcome=Outcome.SUCCESS, why="w",
    )
    ep1.requires.append("some-id")
    assert ep2.requires == [], "requires lists must not be shared between instances"

    ep1.concurrent_with.append("other-id")
    assert ep2.concurrent_with == [], "concurrent_with lists must not be shared"


def test_episode_with_all_optional_fields():
    """An episode with all optional fields set must still pass schema checks."""
    ep = HistoricalEpisode(
        id="ep-full",
        concept="paging",
        problem_posed="How to address more memory than base+limit allows?",
        attempted_solution="Divide memory into fixed-size pages",
        outcome=Outcome.SUCCESS,
        why="Pages eliminate external fragmentation",
        requires=["ep-segmentation"],
        concurrent_with=["ep-swapping"],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source="Silberschatz OS Concepts",
        published_date=date(1969, 1, 1),
    )
    _check_episode_schema(ep)
