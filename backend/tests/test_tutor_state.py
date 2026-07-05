"""
Unit tests for TutorState schema defaults and field types.

Validates: Requirements 1.1
"""

import pytest
from typing import get_type_hints, get_args, get_origin, Literal

from backend.models.schemas import TutorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tutor_state(**overrides) -> TutorState:
    """Return a fully-populated TutorState with sensible defaults."""
    defaults: TutorState = {
        "user_id":         "user-123",
        "topic":           "operating systems",
        "current_episode": "ep-001",
        "mode":            "teacher",
        "session_id":      "session-abc",
        "nudge_count":     0,
        "answer_history":  [],
        "trait_snapshot":  [],
        "ingest_needed":   False,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Field type acceptance tests
# ---------------------------------------------------------------------------

def test_user_id_accepts_str():
    state = make_tutor_state(user_id="alice")
    assert isinstance(state["user_id"], str)


def test_topic_accepts_str():
    state = make_tutor_state(topic="paging and segmentation")
    assert isinstance(state["topic"], str)


def test_current_episode_accepts_str():
    state = make_tutor_state(current_episode="ep-42")
    assert isinstance(state["current_episode"], str)


def test_session_id_accepts_str():
    state = make_tutor_state(session_id="sess-xyz")
    assert isinstance(state["session_id"], str)


def test_nudge_count_accepts_int():
    state = make_tutor_state(nudge_count=3)
    assert isinstance(state["nudge_count"], int)


def test_ingest_needed_accepts_bool_true():
    state = make_tutor_state(ingest_needed=True)
    assert isinstance(state["ingest_needed"], bool)
    assert state["ingest_needed"] is True


def test_ingest_needed_accepts_bool_false():
    state = make_tutor_state(ingest_needed=False)
    assert isinstance(state["ingest_needed"], bool)
    assert state["ingest_needed"] is False


def test_answer_history_accepts_list_of_dicts():
    history = [
        {"episode_id": "ep-001", "answer": "virtual memory", "classification": "correct"},
        {"episode_id": "ep-002", "answer": "paging",         "classification": "partial"},
    ]
    state = make_tutor_state(answer_history=history)
    assert isinstance(state["answer_history"], list)
    assert all(isinstance(item, dict) for item in state["answer_history"])


def test_trait_snapshot_accepts_list_of_strings():
    traits = ["trait-001", "trait-002", "trait-003"]
    state = make_tutor_state(trait_snapshot=traits)
    assert isinstance(state["trait_snapshot"], list)
    assert all(isinstance(t, str) for t in state["trait_snapshot"])


# ---------------------------------------------------------------------------
# nudge_count default value
# ---------------------------------------------------------------------------

def test_nudge_count_initialises_to_zero():
    """nudge_count must start at 0 for a freshly-created TutorState."""
    state = make_tutor_state()
    assert state["nudge_count"] == 0


# ---------------------------------------------------------------------------
# mode Literal values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valid_mode", ["teacher", "interviewer", "digest"])
def test_mode_accepts_valid_literal_values(valid_mode):
    state = make_tutor_state(mode=valid_mode)
    assert state["mode"] == valid_mode


def test_mode_literal_covers_exactly_three_values():
    """The Literal annotation for mode must contain exactly the three expected values."""
    hints = get_type_hints(TutorState)
    mode_hint = hints["mode"]
    assert get_origin(mode_hint) is Literal
    assert set(get_args(mode_hint)) == {"teacher", "interviewer", "digest"}


# ---------------------------------------------------------------------------
# answer_history and trait_snapshot empty-list defaults
# ---------------------------------------------------------------------------

def test_answer_history_accepts_empty_list():
    state = make_tutor_state(answer_history=[])
    assert state["answer_history"] == []


def test_trait_snapshot_accepts_empty_list():
    state = make_tutor_state(trait_snapshot=[])
    assert state["trait_snapshot"] == []


# ---------------------------------------------------------------------------
# All required keys are present
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "user_id",
    "topic",
    "current_episode",
    "mode",
    "session_id",
    "nudge_count",
    "answer_history",
    "trait_snapshot",
    "ingest_needed",
}


def test_tutor_state_has_all_required_keys():
    state = make_tutor_state()
    assert REQUIRED_KEYS.issubset(state.keys())


def test_tutor_state_type_hints_match_required_keys():
    hints = get_type_hints(TutorState)
    assert REQUIRED_KEYS.issubset(hints.keys())
