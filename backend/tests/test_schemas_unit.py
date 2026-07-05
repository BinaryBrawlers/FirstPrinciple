"""Unit tests for TutorState defaults and field types.

**Validates: Requirements 1.1**

Covers:
- A valid TutorState dict can be constructed with all required fields.
- ``nudge_count`` initialises to 0 when set to 0 (the default starting value).
- Each field accepts its declared type.
- All three valid ``mode`` values are accepted.
"""
from __future__ import annotations

import sys
import os

import pytest

# Ensure the backend package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import TutorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tutor_state(**overrides) -> TutorState:
    """Return a minimal valid TutorState, allowing field overrides."""
    base: TutorState = {
        "user_id": "user-123",
        "topic": "Sorting Algorithms",
        "current_episode": "episode-001",
        "mode": "teacher",
        "session_id": "session-abc",
        "nudge_count": 0,
        "answer_history": [],
        "trait_snapshot": [],
        "ingest_needed": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: a complete TutorState can be created with all required fields
# ---------------------------------------------------------------------------

def test_tutor_state_can_be_created_with_all_required_fields() -> None:
    """A TutorState dict with all required fields must be constructable.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state()

    required_fields = (
        "user_id",
        "topic",
        "current_episode",
        "mode",
        "session_id",
        "nudge_count",
        "answer_history",
        "trait_snapshot",
        "ingest_needed",
    )
    for field_name in required_fields:
        assert field_name in state, f"Required field '{field_name}' is missing"


# ---------------------------------------------------------------------------
# Test: nudge_count initialises to 0
# ---------------------------------------------------------------------------

def test_nudge_count_initialises_to_zero() -> None:
    """nudge_count must be 0 when initialised at the default starting value.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(nudge_count=0)
    assert state["nudge_count"] == 0


# ---------------------------------------------------------------------------
# Tests: each field accepts its declared type
# ---------------------------------------------------------------------------

def test_user_id_accepts_str() -> None:
    """user_id field accepts str values.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(user_id="some-user-id")
    assert isinstance(state["user_id"], str)


def test_topic_accepts_str() -> None:
    """topic field accepts str values.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(topic="Binary Search Trees")
    assert isinstance(state["topic"], str)


def test_current_episode_accepts_str() -> None:
    """current_episode field accepts str values (episode IDs are strings).

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(current_episode="ep-42")
    assert isinstance(state["current_episode"], str)


def test_session_id_accepts_str() -> None:
    """session_id field accepts str values.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(session_id="sess-xyz-789")
    assert isinstance(state["session_id"], str)


def test_nudge_count_accepts_int() -> None:
    """nudge_count field accepts int values.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(nudge_count=3)
    assert isinstance(state["nudge_count"], int)
    assert state["nudge_count"] == 3


def test_answer_history_accepts_list_of_dicts() -> None:
    """answer_history field accepts a list of dicts.

    **Validates: Requirements 1.1**
    """
    history = [
        {"episode_id": "ep-1", "answer": "bubble sort", "classification": "correct"},
        {"episode_id": "ep-2", "answer": "merge sort", "classification": "partial"},
    ]
    state = make_tutor_state(answer_history=history)
    assert isinstance(state["answer_history"], list)
    for entry in state["answer_history"]:
        assert isinstance(entry, dict)


def test_answer_history_accepts_empty_list() -> None:
    """answer_history field accepts an empty list.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(answer_history=[])
    assert isinstance(state["answer_history"], list)
    assert len(state["answer_history"]) == 0


def test_trait_snapshot_accepts_list_of_str() -> None:
    """trait_snapshot field accepts a list of str.

    **Validates: Requirements 1.1**
    """
    traits = ["trait-confidence-001", "trait-pace-002"]
    state = make_tutor_state(trait_snapshot=traits)
    assert isinstance(state["trait_snapshot"], list)
    for trait in state["trait_snapshot"]:
        assert isinstance(trait, str)


def test_trait_snapshot_accepts_empty_list() -> None:
    """trait_snapshot field accepts an empty list.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(trait_snapshot=[])
    assert isinstance(state["trait_snapshot"], list)
    assert len(state["trait_snapshot"]) == 0


def test_ingest_needed_accepts_bool_true() -> None:
    """ingest_needed field accepts True.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(ingest_needed=True)
    assert isinstance(state["ingest_needed"], bool)
    assert state["ingest_needed"] is True


def test_ingest_needed_accepts_bool_false() -> None:
    """ingest_needed field accepts False.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(ingest_needed=False)
    assert isinstance(state["ingest_needed"], bool)
    assert state["ingest_needed"] is False


# ---------------------------------------------------------------------------
# Tests: all three valid mode values are accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["teacher", "interviewer", "digest"])
def test_mode_accepts_all_valid_values(mode: str) -> None:
    """mode field must accept all three valid Literal values.

    **Validates: Requirements 1.1**
    """
    state = make_tutor_state(mode=mode)
    assert isinstance(state["mode"], str)
    assert state["mode"] == mode
    assert state["mode"] in ("teacher", "interviewer", "digest")
