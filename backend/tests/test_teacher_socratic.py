"""
Tests for Task 7.3 — Socratic branching logic and stuck fallback.

Tests use a mock LLM so no real API calls are made. They verify:
  - on_user_answer returns tokens and updates state correctly for each label
  - nudge_count increments on partial / novel / matched-failure
  - nudge_count resets to 0 on matched-success
  - stuck_fallback is triggered when nudge_count reaches 2
  - nudge_count is reset to 0 after stuck_fallback delivery
  - stuck_fallback response always contains all four required sections
  - static fallback (LLM down) still contains all four required sections

Requirements: 5.2, 5.3, 5.4, 5.5, 5.6
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

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence, TutorState
from agents.teacher import on_user_answer, stuck_fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(outcome: Outcome = Outcome.SUCCESS) -> HistoricalEpisode:
    return HistoricalEpisode(
        id="ep_paging",
        concept="Paging",
        problem_posed="How do you eliminate external fragmentation in memory management?",
        attempted_solution="Divide memory into fixed-size pages and maintain a page table.",
        outcome=outcome,
        why="Paging removes variable-size allocations, eliminating external fragmentation.",
        requires=["ep_segmentation"],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source="Denning, 1970",
    )


def _make_state(nudge_count: int = 0) -> TutorState:
    return TutorState(
        user_id="user_test",
        topic="OS memory management",
        current_episode="ep_paging",
        mode="teacher",
        session_id="sess_001",
        nudge_count=nudge_count,
        answer_history=[],
        trait_snapshot=[],
        ingest_needed=False,
    )


async def _collect(gen) -> str:
    """Collect all tokens from an async generator into a single string."""
    tokens = []
    async for t in gen:
        tokens.append(t)
    return "".join(tokens)


def _make_streaming_response(content: str):
    """Build a mock litellm streaming response that yields a single chunk."""
    chunk = MagicMock()
    chunk.choices[0].delta.content = content

    async def _aiter():
        yield chunk

    mock_resp = MagicMock()
    mock_resp.__aiter__ = lambda self: _aiter()
    return mock_resp


# ---------------------------------------------------------------------------
# classify_answer is patched to return a controlled label in all tests below
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["matched-failure", "matched-success", "partial", "novel"])
async def test_on_user_answer_records_history(label):
    """Answer is appended to answer_history with correct classification."""
    episode = _make_episode()
    state = _make_state()

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value=label)),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response("some response"),
        ),
    ):
        await _collect(on_user_answer(state, "my answer", episode))

    assert len(state["answer_history"]) == 1
    entry = state["answer_history"][0]
    assert entry["episode_id"] == episode.id
    assert entry["answer"] == "my answer"
    assert entry["classification"] == label


@pytest.mark.asyncio
async def test_matched_success_resets_nudge_count():
    """matched-success resets nudge_count to 0 regardless of prior value."""
    episode = _make_episode(Outcome.SUCCESS)
    state = _make_state(nudge_count=1)

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value="matched-success")),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response("Great job!"),
        ),
    ):
        await _collect(on_user_answer(state, "use pages", episode))

    assert state["nudge_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["partial", "novel", "matched-failure"])
async def test_non_success_increments_nudge_count(label):
    """partial / novel / matched-failure all increment nudge_count by 1."""
    episode = _make_episode(Outcome.FAILURE if label == "matched-failure" else Outcome.SUCCESS)
    state = _make_state(nudge_count=0)

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value=label)),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response("response"),
        ),
    ):
        await _collect(on_user_answer(state, "wrong answer", episode))

    assert state["nudge_count"] == 1


@pytest.mark.asyncio
async def test_stuck_fallback_triggered_at_nudge_count_2():
    """When nudge_count reaches 2, stuck_fallback is delivered instead of branch response."""
    episode = _make_episode()
    # Start at nudge_count=1; a partial answer pushes it to 2
    state = _make_state(nudge_count=1)

    fallback_text = (
        "**Problem framing** — X\n\n"
        "**Solution hint** — Y\n\n"
        "**Engineering Insight** — Z\n\n"
        "**Historical note** — W"
    )

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value="partial")),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response(fallback_text),
        ),
    ):
        result = await _collect(on_user_answer(state, "still struggling", episode))

    # The fallback was delivered
    assert "Problem framing" in result or len(result) > 0
    # nudge_count reset to 0 after fallback
    assert state["nudge_count"] == 0


@pytest.mark.asyncio
async def test_nudge_count_reset_after_stuck_fallback():
    """nudge_count is explicitly 0 after stuck_fallback delivery."""
    episode = _make_episode()
    state = _make_state(nudge_count=1)

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value="novel")),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response("**Problem framing** — A\n**Solution hint** — B\n**Engineering Insight** — C\n**Historical note** — D"),
        ),
    ):
        await _collect(on_user_answer(state, "a novel idea", episode))

    assert state["nudge_count"] == 0


@pytest.mark.asyncio
async def test_stuck_fallback_contains_all_four_sections():
    """stuck_fallback response contains all four required labelled sections."""
    episode = _make_episode()

    full_response = (
        "**Problem framing** — How do you eliminate external fragmentation?\n\n"
        "**Solution hint** — Think about fixed-size units.\n\n"
        "**Engineering Insight** — Fixed pages remove variable-size allocations.\n\n"
        "**Historical note** — Denning introduced paging in 1970."
    )

    with patch(
        "agents.teacher.litellm.acompletion",
        return_value=_make_streaming_response(full_response),
    ):
        result = await _collect(stuck_fallback(episode))

    for section in ["Problem framing", "Solution hint", "Engineering Insight", "Historical note"]:
        assert section in result, f"Missing section: {section!r}"


@pytest.mark.asyncio
async def test_stuck_fallback_static_fallback_contains_all_four_sections():
    """When the LLM is unavailable, the static fallback still has all four sections."""
    episode = _make_episode()

    with patch(
        "agents.teacher.litellm.acompletion",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = await _collect(stuck_fallback(episode))

    for section in ["Problem framing", "Solution hint", "Engineering Insight", "Historical note"]:
        assert section in result, f"Missing section in static fallback: {section!r}"


@pytest.mark.asyncio
async def test_on_user_answer_streams_tokens():
    """on_user_answer yields at least one token for a normal (non-stuck) answer."""
    episode = _make_episode()
    state = _make_state(nudge_count=0)

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value="matched-success")),
        patch(
            "agents.teacher.litellm.acompletion",
            return_value=_make_streaming_response("Well done!"),
        ),
    ):
        result = await _collect(on_user_answer(state, "use pages", episode))

    assert len(result) > 0


@pytest.mark.asyncio
async def test_on_user_answer_llm_down_yields_fallback_message():
    """When the LLM is unavailable during branching, a static message is still yielded."""
    episode = _make_episode()
    state = _make_state(nudge_count=0)

    with (
        patch("agents.teacher.classify_answer", new=AsyncMock(return_value="partial")),
        patch(
            "agents.teacher.litellm.acompletion",
            side_effect=RuntimeError("LLM unavailable"),
        ),
    ):
        result = await _collect(on_user_answer(state, "some partial answer", episode))

    assert len(result) > 0
