"""Property-based tests for HistoricalEpisode schema completeness.

**Validates: Requirements 1.1**

Property 1: HistoricalEpisode schema completeness
  For any object that the Ingestion Agent writes to Track A, it SHALL contain
  all required fields with values conforming to their declared types and
  enumerations.
"""
from __future__ import annotations

import sys
import os
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure the backend package root is on sys.path so we can import from models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for non-empty strings (ids, concepts, etc.)
nonempty_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters="-_.,!?'",
    ),
    min_size=1,
    max_size=200,
)

# Strategy for lists of episode ID strings (may be empty)
episode_id_list = st.lists(nonempty_text, min_size=0, max_size=10)

# Strategy for the optional source field
optional_source = st.one_of(st.none(), nonempty_text)

# Strategy for the optional published_date field
optional_date = st.one_of(
    st.none(),
    st.dates(min_value=date(1800, 1, 1), max_value=date(2100, 12, 31)),
)

# Strategy for Outcome enum values
outcome_strategy = st.sampled_from(Outcome)

# Strategy for SourceConfidence enum values
source_confidence_strategy = st.sampled_from(SourceConfidence)

# Composite strategy that builds a complete HistoricalEpisode
historical_episode_strategy = st.builds(
    HistoricalEpisode,
    id=nonempty_text,
    concept=nonempty_text,
    problem_posed=nonempty_text,
    attempted_solution=nonempty_text,
    outcome=outcome_strategy,
    why=nonempty_text,
    requires=episode_id_list,
    concurrent_with=episode_id_list,
    source_confidence=source_confidence_strategy,
    source=optional_source,
    published_date=optional_date,
)


# ---------------------------------------------------------------------------
# Property 1: HistoricalEpisode schema completeness
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "id",
    "concept",
    "problem_posed",
    "attempted_solution",
    "outcome",
    "why",
    "requires",
    "concurrent_with",
    "source_confidence",
)

OPTIONAL_FIELDS = ("source", "published_date")


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_all_required_fields_present(episode: HistoricalEpisode) -> None:
    """All required fields must be present on every generated HistoricalEpisode.

    **Validates: Requirements 1.1**
    """
    for field_name in REQUIRED_FIELDS:
        assert hasattr(episode, field_name), (
            f"Required field '{field_name}' is missing from HistoricalEpisode"
        )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_string_fields_are_str(episode: HistoricalEpisode) -> None:
    """String fields must be instances of str.

    **Validates: Requirements 1.1**
    """
    str_fields = ("id", "concept", "problem_posed", "attempted_solution", "why")
    for field_name in str_fields:
        value = getattr(episode, field_name)
        assert isinstance(value, str), (
            f"Field '{field_name}' should be str, got {type(value).__name__!r}"
        )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_outcome_is_outcome_enum(episode: HistoricalEpisode) -> None:
    """The outcome field must be a member of the Outcome enum.

    **Validates: Requirements 1.1**
    """
    assert isinstance(episode.outcome, Outcome), (
        f"Field 'outcome' should be Outcome, got {type(episode.outcome).__name__!r}"
    )
    assert episode.outcome in (Outcome.SUCCESS, Outcome.FAILURE, Outcome.PARTIAL), (
        f"Unexpected outcome value: {episode.outcome!r}"
    )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_source_confidence_is_enum(episode: HistoricalEpisode) -> None:
    """The source_confidence field must be a member of the SourceConfidence enum.

    **Validates: Requirements 1.1**
    """
    assert isinstance(episode.source_confidence, SourceConfidence), (
        f"Field 'source_confidence' should be SourceConfidence, "
        f"got {type(episode.source_confidence).__name__!r}"
    )
    valid_values = (
        SourceConfidence.CITED_SOURCE,
        SourceConfidence.NAMED_REFERENCE,
        SourceConfidence.REASONED,
    )
    assert episode.source_confidence in valid_values, (
        f"Unexpected source_confidence value: {episode.source_confidence!r}"
    )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_requires_and_concurrent_with_are_lists(episode: HistoricalEpisode) -> None:
    """The requires and concurrent_with fields must be lists.

    **Validates: Requirements 1.1**
    """
    assert isinstance(episode.requires, list), (
        f"Field 'requires' should be list, got {type(episode.requires).__name__!r}"
    )
    assert isinstance(episode.concurrent_with, list), (
        f"Field 'concurrent_with' should be list, "
        f"got {type(episode.concurrent_with).__name__!r}"
    )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_list_fields_contain_strings(episode: HistoricalEpisode) -> None:
    """All entries in requires and concurrent_with must be strings.

    **Validates: Requirements 1.1**
    """
    for field_name in ("requires", "concurrent_with"):
        for item in getattr(episode, field_name):
            assert isinstance(item, str), (
                f"Items in '{field_name}' should be str, "
                f"got {type(item).__name__!r}: {item!r}"
            )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_optional_source_type_when_present(episode: HistoricalEpisode) -> None:
    """When source is not None, it must be a str.

    **Validates: Requirements 1.1**
    """
    if episode.source is not None:
        assert isinstance(episode.source, str), (
            f"Optional field 'source' should be str when present, "
            f"got {type(episode.source).__name__!r}"
        )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_optional_published_date_type_when_present(episode: HistoricalEpisode) -> None:
    """When published_date is not None, it must be a datetime.date instance.

    **Validates: Requirements 1.1**
    """
    if episode.published_date is not None:
        assert isinstance(episode.published_date, date), (
            f"Optional field 'published_date' should be date when present, "
            f"got {type(episode.published_date).__name__!r}"
        )


@given(episode=historical_episode_strategy)
@settings(max_examples=100)
def test_all_required_fields_non_none(episode: HistoricalEpisode) -> None:
    """All required fields must be non-None (optional fields may be None).

    **Validates: Requirements 1.1**
    """
    for field_name in REQUIRED_FIELDS:
        value = getattr(episode, field_name)
        assert value is not None, (
            f"Required field '{field_name}' must not be None"
        )
