"""
Tests for Task 8.2 — Inline grading, confidence prompt, and confidently-wrong penalty.

Tests:
  1. test_compute_penalty_correct_is_zero
  2. test_compute_penalty_wrong_low_confidence
  3. test_compute_penalty_wrong_high_confidence_is_harsher
  4. test_compute_penalty_wrong_confidence_5_harsher_than_2
  5. test_compute_penalty_partial_high_confidence_is_harsher
  6. test_grade_answer_returns_valid_label
  7. test_grade_answer_llm_down_returns_valid_label
  8. test_on_answer_streams_grade_and_confidence_prompt
  9. test_on_confidence_received_wrong_high_confidence_mentions_overconfidence

Requirements: 7.4, 7.5, 7.6
"""
from __future__ import annotations

import sys
import os

# Ensure backend/ is on the path
_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.interviewer import (
    HARSH_MULTIPLIER,
    compute_penalty,
    grade_answer,
    on_answer,
    on_confidence_received,
    request_confidence_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_question(concept: str = "Paging") -> dict:
    return {
        "episode_id": "ep_paging",
        "concept": concept,
        "question": (
            "The historical approach to 'Paging' ultimately failed. "
            "Why do you think it did not work?"
        ),
        "from_failure": True,
    }


def _make_non_streaming_response(content: str):
    """Build a mock litellm non-streaming response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


async def _collect(gen) -> str:
    """Drain an async generator and join all tokens into a single string."""
    tokens = []
    async for token in gen:
        tokens.append(token)
    return "".join(tokens)


# ---------------------------------------------------------------------------
# compute_penalty tests
# ---------------------------------------------------------------------------

def test_compute_penalty_correct_is_zero():
    """Any confidence score with a correct grade yields 0.0 penalty."""
    for confidence in range(1, 6):
        assert compute_penalty("correct", confidence) == 0.0, (
            f"Expected 0.0 for correct with confidence={confidence}"
        )


def test_compute_penalty_wrong_low_confidence():
    """Grade 'wrong' + confidence 1 → base penalty (1.0), no multiplier."""
    penalty = compute_penalty("wrong", 1)
    assert penalty == 1.0


def test_compute_penalty_wrong_high_confidence_is_harsher():
    """Grade 'wrong' + confidence 4 → penalty > penalty(wrong, confidence=1).

    Validates: Requirements 7.6 (Property 14 — confidently-wrong is penalised harder)
    """
    low = compute_penalty("wrong", 1)
    high = compute_penalty("wrong", 4)
    assert high > low, (
        f"Expected penalty(wrong, 4)={high} > penalty(wrong, 1)={low}"
    )


def test_compute_penalty_wrong_confidence_5_harsher_than_2():
    """Grade 'wrong' + confidence 5 → penalty > penalty(wrong, confidence=2).

    Validates: Requirements 7.6 (Property 14)
    """
    low = compute_penalty("wrong", 2)
    high = compute_penalty("wrong", 5)
    assert high > low, (
        f"Expected penalty(wrong, 5)={high} > penalty(wrong, 2)={low}"
    )


def test_compute_penalty_partial_high_confidence_is_harsher():
    """Grade 'partial' + confidence 5 → penalty > penalty(partial, confidence=1).

    Validates: Requirements 7.6
    """
    low = compute_penalty("partial", 1)
    high = compute_penalty("partial", 5)
    assert high > low, (
        f"Expected penalty(partial, 5)={high} > penalty(partial, 1)={low}"
    )


def test_harsh_multiplier_value():
    """HARSH_MULTIPLIER is applied exactly — wrong+4 == 1.0 * HARSH_MULTIPLIER."""
    assert compute_penalty("wrong", 4) == 1.0 * HARSH_MULTIPLIER
    assert compute_penalty("wrong", 5) == 1.0 * HARSH_MULTIPLIER
    assert compute_penalty("partial", 4) == 0.5 * HARSH_MULTIPLIER
    assert compute_penalty("partial", 5) == 0.5 * HARSH_MULTIPLIER


# ---------------------------------------------------------------------------
# grade_answer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grade_answer_returns_valid_label():
    """Mock LLM returning 'correct'; assert grade_answer returns 'correct'.

    Validates: Requirements 7.5
    """
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("correct")),
    ):
        result = await grade_answer(question, "Fixed-size pages eliminate external fragmentation.")

    assert result == "correct"


@pytest.mark.asyncio
async def test_grade_answer_llm_down_returns_valid_label():
    """Mock LLM raising RuntimeError; assert grade_answer still returns a valid label.

    Validates: Requirements 7.5 (fallback path)
    """
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(side_effect=RuntimeError("LLM unavailable")),
    ):
        result = await grade_answer(question, "Some answer.")

    assert result in {"correct", "partial", "wrong"}, (
        f"grade_answer fallback returned unexpected value: {result!r}"
    )


@pytest.mark.asyncio
async def test_grade_answer_partial_label():
    """Mock LLM returning 'partial'; verify grade_answer returns 'partial'."""
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("partial")),
    ):
        result = await grade_answer(question, "Kind of related answer.")

    assert result == "partial"


@pytest.mark.asyncio
async def test_grade_answer_wrong_label():
    """Mock LLM returning 'wrong'; verify grade_answer returns 'wrong'."""
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("wrong")),
    ):
        result = await grade_answer(question, "I have no idea.")

    assert result == "wrong"


# ---------------------------------------------------------------------------
# request_confidence_score tests
# ---------------------------------------------------------------------------

def test_request_confidence_score_is_string():
    """request_confidence_score returns a non-empty string."""
    text = request_confidence_score()
    assert isinstance(text, str)
    assert len(text) > 0


def test_request_confidence_score_contains_scale():
    """Confidence prompt mentions 1 and 5 (the scale boundaries)."""
    text = request_confidence_score()
    assert "1" in text and "5" in text


def test_request_confidence_score_mentions_confidence():
    """Confidence prompt contains the word 'confidence' or 'confident'."""
    text = request_confidence_score().lower()
    assert "confidence" in text or "confident" in text


# ---------------------------------------------------------------------------
# on_answer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_answer_streams_grade_and_confidence_prompt():
    """on_answer yields tokens that together mention confidence scale (1 and 5).

    Validates: Requirements 7.4, 7.5
    """
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("correct")),
    ):
        result = await _collect(on_answer(question, "Pages fix fragmentation."))

    assert len(result) > 0, "on_answer yielded no tokens"
    result_lower = result.lower()
    # Must contain confidence prompt with scale reference
    assert "1" in result and "5" in result, (
        f"Confidence scale not found in on_answer output:\n{result}"
    )
    assert "confidence" in result_lower or "confident" in result_lower, (
        f"'confidence' not found in on_answer output:\n{result}"
    )


@pytest.mark.asyncio
async def test_on_answer_correct_streams_positive_feedback():
    """on_answer with grade 'correct' streams positive feedback."""
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("correct")),
    ):
        result = await _collect(on_answer(question, "Correct answer here."))

    assert "correct" in result.lower() or "✓" in result


@pytest.mark.asyncio
async def test_on_answer_wrong_streams_incorrect_feedback():
    """on_answer with grade 'wrong' streams incorrect-answer feedback."""
    question = _make_question()
    with patch(
        "agents.interviewer.litellm.acompletion",
        new=AsyncMock(return_value=_make_non_streaming_response("wrong")),
    ):
        result = await _collect(on_answer(question, "Totally wrong."))

    assert "incorrect" in result.lower() or "✗" in result or "wrong" in result.lower()


# ---------------------------------------------------------------------------
# on_confidence_received tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_confidence_received_wrong_high_confidence_mentions_overconfidence():
    """Confidently-wrong answer (grade='wrong', confidence=4) mentions overconfidence.

    Validates: Requirements 7.6
    """
    result = await _collect(on_confidence_received(grade="wrong", confidence_score=4))
    result_lower = result.lower()

    assert "overconfident" in result_lower or "confident" in result_lower, (
        f"Expected overconfidence callout in output:\n{result}"
    )


@pytest.mark.asyncio
async def test_on_confidence_received_wrong_confidence_5_mentions_overconfidence():
    """Confidently-wrong answer (grade='wrong', confidence=5) mentions overconfidence.

    Validates: Requirements 7.6
    """
    result = await _collect(on_confidence_received(grade="wrong", confidence_score=5))
    result_lower = result.lower()

    assert "overconfident" in result_lower or "confident" in result_lower, (
        f"Expected overconfidence callout in output:\n{result}"
    )


@pytest.mark.asyncio
async def test_on_confidence_received_correct_no_penalty_mentioned():
    """Correct answer yields no-penalty feedback."""
    result = await _collect(on_confidence_received(grade="correct", confidence_score=3))
    result_lower = result.lower()
    # Should confirm no penalty
    assert "no penalty" in result_lower or "0.0" in result or "correct" in result_lower


@pytest.mark.asyncio
async def test_on_confidence_received_wrong_low_confidence_no_overconfidence():
    """Wrong answer with low confidence (1) should NOT call out overconfidence."""
    result = await _collect(on_confidence_received(grade="wrong", confidence_score=1))
    result_lower = result.lower()
    # May mention confidence but should not call out overconfidence for low confidence
    assert "overconfident" not in result_lower


@pytest.mark.asyncio
async def test_on_confidence_received_mentions_penalty_value_for_wrong_high():
    """Confidently-wrong answer streams include the computed penalty value."""
    result = await _collect(on_confidence_received(grade="wrong", confidence_score=5))
    # Penalty should be 1.0 * 2.0 = 2.0
    assert "2.0" in result or "2" in result


@pytest.mark.asyncio
async def test_on_confidence_received_streams_tokens():
    """on_confidence_received yields at least one token for any input."""
    result = await _collect(on_confidence_received(grade="partial", confidence_score=3))
    assert len(result) > 0
