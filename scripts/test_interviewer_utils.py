#!/usr/bin/env python3
"""
test_interviewer_utils.py — Walkthrough demo for the Interviewer Agent utility functions.

Covers:
  1. select_questions()           — question generation preferring failure episodes
  2. grade_answer()               — LLM grader (correct / partial / wrong)
  3. request_confidence_score()   — fixed confidence prompt text
  4. compute_penalty()            — penalty math including HARSH_MULTIPLIER
  5. on_answer()                  — streaming grade feedback + confidence prompt
  6. on_confidence_received()     — streaming penalty feedback
  7. compute_misconception_diff() — pure-function session diff
  8. on_session_end()             — streaming diff summary

No real LLM calls are made unless MISTRAL_API_KEY is set in .env.
If no key is present, litellm raises and the heuristic / fallback paths kick in,
which is exactly what this demo exercises.

Run from the PROJECT ROOT:
    python scripts/test_interviewer_utils.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date

# ---------------------------------------------------------------------------
# Path + env setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

os.environ.setdefault("MISTRAL_API_KEY", "dummy")
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s — %(message)s")
logging.getLogger("agents.interviewer").setLevel(logging.DEBUG)

import config  # noqa: F401

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence, TutorState
from agents.interviewer import (
    select_questions,
    grade_answer,
    request_confidence_score,
    compute_penalty,
    on_answer,
    on_confidence_received,
    compute_misconception_diff,
    on_session_end,
    HARSH_MULTIPLIER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sep(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


def make_episode(
    id: str,
    concept: str,
    problem: str,
    solution: str,
    outcome: Outcome = Outcome.SUCCESS,
    why: str = "it worked",
    requires: list[str] | None = None,
    concurrent_with: list[str] | None = None,
    published_date: date | None = None,
    source: str | None = None,
    confidence: SourceConfidence = SourceConfidence.NAMED_REFERENCE,
) -> HistoricalEpisode:
    return HistoricalEpisode(
        id=id,
        concept=concept,
        problem_posed=problem,
        attempted_solution=solution,
        outcome=outcome,
        why=why,
        requires=requires or [],
        concurrent_with=concurrent_with or [],
        published_date=published_date,
        source=source,
        source_confidence=confidence,
    )


def make_state(
    user_id: str = "demo_user",
    topic: str = "demo_topic",
    trait_snapshot: list[str] | None = None,
    answer_history: list[dict] | None = None,
) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic=topic,
        current_episode="",
        mode="interviewer",
        session_id="demo_interview_session",
        nudge_count=0,
        answer_history=answer_history or [],
        trait_snapshot=trait_snapshot or [],
        ingest_needed=False,
    )


async def stream_to_str(gen) -> str:
    """Collect all tokens from an async generator into a single string."""
    tokens: list[str] = []
    async for token in gen:
        tokens.append(token)
    return "".join(tokens)


# ---------------------------------------------------------------------------
# 1. select_questions
# ---------------------------------------------------------------------------

async def demo_select_questions() -> None:
    sep("1. select_questions(weak_points, track_a_failure_episodes)")
    print("""
  WHAT IT DOES:
    Generates interview questions from Track A episodes, in priority order:
      1. Episodes whose outcome == FAILURE (Requirement 7.3) — probes WHY it failed
      2. Episodes that overlap with the learner's Track B weak points (Req 7.1)
      3. Any remaining episodes

    Each question targets EXACTLY ONE concept (Requirement 7.2).

  HOW IT DOES IT:
    - Sorts episodes: failure episodes first, then by weak-point overlap score.
    - Calls the LLM (or deterministic fallback) once per episode to generate
      a single-concept question.
    - Falls back to a template question when the LLM is unavailable.

  Example: three episodes (1 failure, 2 success), one weak point
""")

    ep_xor = make_episode(
        "xor_failure",
        "XOR Failure",
        "Can a single perceptron learn XOR?",
        "Minsky & Papert proved a single-layer perceptron CANNOT learn XOR.",
        outcome=Outcome.FAILURE,
        why="XOR is not linearly separable; a single perceptron can only draw a straight line.",
        source="Minsky & Papert, 1969",
        published_date=date(1969, 1, 1),
    )
    ep_mlp = make_episode(
        "mlp",
        "Multi-Layer Perceptron",
        "How do we learn non-linear functions?",
        "Add hidden layers; use a non-linear activation function.",
        outcome=Outcome.SUCCESS,
        why="Hidden layers can approximate any continuous function.",
    )
    ep_backprop = make_episode(
        "backprop",
        "Backpropagation",
        "How do we efficiently train a multi-layer network?",
        "Apply the chain rule of calculus layer-by-layer to propagate gradients.",
        outcome=Outcome.SUCCESS,
        why="Chain rule allows O(n) gradient computation.",
    )

    episodes = [ep_mlp, ep_backprop, ep_xor]  # deliberately not in failure-first order
    # Simulate a Track B weak point: learner struggles with XOR / linear separability
    weak_points_mock = type("Trait", (), {"concept": "XOR linear separability", "resolved": False})()

    print(f"  Episodes supplied : {[ep.id for ep in episodes]}")
    print(f"  Weak point        : 'XOR linear separability'")
    print(f"\n  Generating questions…\n")

    questions = await select_questions(weak_points_mock, episodes)

    print(f"  Questions generated: {len(questions)}\n")
    for i, q in enumerate(questions, 1):
        flag = " [FAILURE]" if q["from_failure"] else ""
        print(f"  Q{i}  concept={q['concept']!r}{flag}")
        print(f"       ep_id  ={q['episode_id']!r}")
        print(f"       q      ={q['question']!r}\n")

    # Verify failure episode sorted first
    if questions:
        first_is_failure = questions[0]["from_failure"]
        print(f"  Failure episode sorted first? {'✓ Yes' if first_is_failure else '✗ No — sorting bug'}")


# ---------------------------------------------------------------------------
# 2. grade_answer
# ---------------------------------------------------------------------------

async def demo_grade_answer() -> None:
    sep("2. grade_answer(question, answer)")
    print("""
  WHAT IT DOES:
    Grades a learner's answer against a question dict, returning exactly one of:
      "correct"   — substantially correct
      "partial"   — captures part of the concept but misses key elements
      "wrong"     — incorrect or shows a fundamental misconception

  HOW IT DOES IT:
    1. Calls the LLM with a zero-temperature, single-word-output prompt.
    2. Tries substring extraction if the LLM returns extra words.
    3. Falls back to: non-empty answer → "partial", empty → "wrong".

  Requirements: 7.5
""")
    question = {
        "episode_id": "xor_failure",
        "concept": "XOR Failure",
        "question": "Why could the perceptron of the 1960s not learn the XOR function?",
        "from_failure": True,
    }

    test_cases = [
        (
            "XOR is not linearly separable, so a single perceptron — which only draws a hyperplane — cannot separate the XOR classes.",
            "strong, accurate answer → expected: correct",
        ),
        (
            "The perceptron had some problem with non-linear stuff, I think it needed more layers.",
            "partially right — knows multi-layer helps but vague on why → expected: partial",
        ),
        (
            "The perceptron failed because its learning rate was too high.",
            "incorrect — learning rate is irrelevant here → expected: wrong",
        ),
        (
            "",
            "empty answer → expected: wrong (heuristic fallback)",
        ),
    ]

    print(f"  Question: {question['question']!r}\n")
    for answer, note in test_cases:
        grade = await grade_answer(question, answer)
        icon = {"correct": "✅", "partial": "🔶", "wrong": "✗"}.get(grade, "?")
        answer_display = repr(answer) if answer else "(empty)"
        print(f"  {icon} grade={grade!r}")
        print(f"     answer: {answer_display}")
        print(f"     note  : {note}\n")


# ---------------------------------------------------------------------------
# 3. request_confidence_score
# ---------------------------------------------------------------------------

def demo_request_confidence_score() -> None:
    sep("3. request_confidence_score()")
    print("""
  WHAT IT DOES:
    Returns the fixed prompt text asking the learner to rate their confidence
    on a 1–5 scale.  Pure function — no LLM, no I/O.

  Requirements: 7.4
""")
    prompt_text = request_confidence_score()
    print("  Prompt text returned:\n")
    for line in prompt_text.split("\n"):
        print(f"    {line}")


# ---------------------------------------------------------------------------
# 4. compute_penalty
# ---------------------------------------------------------------------------

def demo_compute_penalty() -> None:
    sep("4. compute_penalty(grade, confidence_score)")
    print(f"""
  WHAT IT DOES:
    Returns the numeric penalty for an answer:
      "correct"          → 0.0 (always)
      "partial", conf≤3  → 0.5
      "partial", conf∈{{4,5}} → 0.5 × {HARSH_MULTIPLIER:.0f} = {0.5 * HARSH_MULTIPLIER:.1f}  ← overconfidence penalty
      "wrong",   conf≤3  → 1.0
      "wrong",   conf∈{{4,5}} → 1.0 × {HARSH_MULTIPLIER:.0f} = {1.0 * HARSH_MULTIPLIER:.1f}  ← overconfidence penalty

  Property 14 guarantee:
    penalty("wrong", 4) > penalty("wrong", 1)   ← harshly penalises overconfidence
    penalty("wrong", 5) > penalty("wrong", 2)

  Requirements: 7.6
""")
    header = f"  {'grade':<10} {'conf':>5}  {'penalty':>8}  {'note'}"
    print(header)
    print("  " + "─" * 55)

    cases = [
        ("correct", 1, "no penalty"),
        ("correct", 5, "no penalty even if confident"),
        ("partial", 1, "low-confidence partial"),
        ("partial", 3, "moderate-confidence partial"),
        ("partial", 4, "HIGH confidence partial → harsh"),
        ("partial", 5, "HIGH confidence partial → harsh"),
        ("wrong",   1, "low-confidence wrong"),
        ("wrong",   3, "moderate-confidence wrong"),
        ("wrong",   4, "HIGH confidence wrong → harsh ⚠️"),
        ("wrong",   5, "HIGH confidence wrong → harsh ⚠️"),
    ]

    for grade, conf, note in cases:
        p = compute_penalty(grade, conf)
        print(f"  {grade:<10} {conf:>5}  {p:>8.1f}  {note}")

    # Verify Property 14
    print()
    p_wrong_4 = compute_penalty("wrong", 4)
    p_wrong_1 = compute_penalty("wrong", 1)
    p_wrong_5 = compute_penalty("wrong", 5)
    p_wrong_2 = compute_penalty("wrong", 2)

    ok_a = p_wrong_4 > p_wrong_1
    ok_b = p_wrong_5 > p_wrong_2
    print(f"  Property 14 check:")
    print(f"    penalty(wrong,4)={p_wrong_4:.1f} > penalty(wrong,1)={p_wrong_1:.1f}  → {'✓' if ok_a else '✗'}")
    print(f"    penalty(wrong,5)={p_wrong_5:.1f} > penalty(wrong,2)={p_wrong_2:.1f}  → {'✓' if ok_b else '✗'}")


# ---------------------------------------------------------------------------
# 5. on_answer
# ---------------------------------------------------------------------------

async def demo_on_answer() -> None:
    sep("5. on_answer(question, answer)")
    print("""
  WHAT IT DOES:
    Handles one answer turn: grades the answer and streams grade feedback
    + confidence prompt in the same response (Requirement 7.5).

  HOW IT DOES IT:
    1. Calls grade_answer() to get the label.
    2. Builds grade-feedback text.
    3. Appends request_confidence_score() prompt.
    4. Streams word-by-word for SSE.

  Requirements: 7.4, 7.5
""")
    question = {
        "episode_id": "vanishing_gradient",
        "concept": "Vanishing Gradient",
        "question": "Why did deep networks fail to train well before LSTM?",
        "from_failure": True,
    }

    cases = [
        "The gradient shrinks exponentially as it propagates backward through many sigmoid layers.",
        "Deep networks had some gradient problem, something about layers.",
        "I think the learning rate was too small.",
    ]

    for answer in cases:
        print(f"\n  Answer: {answer!r}")
        print("  Response stream:")
        print("  " + "·" * 55)
        response = await stream_to_str(on_answer(question, answer))
        for line in response.split("\n"):
            if line.strip():
                print(f"    {line}")
        print("  " + "·" * 55)


# ---------------------------------------------------------------------------
# 6. on_confidence_received
# ---------------------------------------------------------------------------

async def demo_on_confidence_received() -> None:
    sep("6. on_confidence_received(grade, confidence_score)")
    print(f"""
  WHAT IT DOES:
    Streams the penalty feedback after the learner reports their confidence.
    Explicitly calls out overconfidence when confidence ∈ {{4,5}} and grade ≠ correct.

  Requirements: 7.6
""")
    cases = [
        ("correct", 5, "confident and right → no penalty, positive reinforcement"),
        ("correct", 2, "right but unsure → encourage calibration"),
        ("partial",  4, f"partial, overconfident → harsh penalty ({0.5*HARSH_MULTIPLIER:.1f})"),
        ("wrong",    5, f"wrong, very overconfident → harshest penalty ({1.0*HARSH_MULTIPLIER:.1f})"),
        ("wrong",    1, "wrong but knew it → low penalty (1.0)"),
    ]

    for grade, conf, note in cases:
        print(f"\n  grade={grade!r}  confidence={conf}  ({note})")
        print("  " + "·" * 55)
        response = await stream_to_str(on_confidence_received(grade, conf))
        for line in response.split("\n"):
            if line.strip():
                print(f"    {line}")
        print("  " + "·" * 55)


# ---------------------------------------------------------------------------
# 7. compute_misconception_diff
# ---------------------------------------------------------------------------

def demo_compute_misconception_diff() -> None:
    sep("7. compute_misconception_diff(trait_snapshot, current_track_b)")
    print("""
  WHAT IT DOES:
    Pure function. Compares trait IDs from session start vs session end
    to produce three buckets:

      cleared    — in snapshot but NOT in current → resolved this session  ✅
      new        — in current but NOT in snapshot → surfaced this session  🆕
      persisted  — in both → still unresolved                              ⚠️

  Requirements: 7.7
""")
    snapshot = ["misconception:xor_linearity", "misconception:backprop_chain_rule", "misconception:lstm_gates"]
    current  = ["misconception:backprop_chain_rule", "misconception:lstm_gates", "misconception:attention_bias"]

    diff = compute_misconception_diff(snapshot, current)

    print(f"  Session start snapshot : {snapshot}")
    print(f"  Session end current    : {current}")
    print()
    print(f"  ✅ Cleared   : {diff['cleared']}")
    print(f"  🆕 New       : {diff['new']}")
    print(f"  ⚠️  Persisted : {diff['persisted']}")

    # Verify correctness
    assert "misconception:xor_linearity" in diff["cleared"],    "xor_linearity should be cleared"
    assert "misconception:attention_bias" in diff["new"],       "attention_bias should be new"
    assert "misconception:backprop_chain_rule" in diff["persisted"], "backprop should persist"
    print("\n  Diff correctness: ✓ all checks passed")

    # Edge cases
    print("\n  Edge cases:")
    empty_both = compute_misconception_diff([], [])
    print(f"    Both empty → {empty_both}")

    same = compute_misconception_diff(["x", "y"], ["x", "y"])
    print(f"    No change  → {same}")

    all_new = compute_misconception_diff([], ["a", "b"])
    print(f"    All new    → {all_new}")

    all_cleared = compute_misconception_diff(["a", "b"], [])
    print(f"    All cleared → {all_cleared}")


# ---------------------------------------------------------------------------
# 8. on_session_end
# ---------------------------------------------------------------------------

async def demo_on_session_end() -> None:
    sep("8. on_session_end(state)")
    print("""
  WHAT IT DOES:
    1. Calls cognee.recall() on the user's Track B graph to fetch current traits.
    2. Computes the diff against state["trait_snapshot"] (captured at session start).
    3. Streams a formatted diff summary token-by-token.

    If cognee.recall() fails (e.g. no API, test environment) it gracefully
    uses an empty current state — so the summary still streams.

  Requirements: 7.7, 7.9
""")
    # Simulate a snapshot captured at session start
    state = make_state(
        user_id="demo_user",
        topic="deep_learning",
        trait_snapshot=["misconception:xor_linearity", "misconception:vanishing_gradient"],
    )

    print(f"  Trait snapshot at session start: {state['trait_snapshot']}")
    print(f"  (cognee.recall() will likely return empty in test env — diff uses empty current state)\n")
    print("  Streaming session-end diff summary:\n")
    print("  " + "·" * 60)

    response = await stream_to_str(on_session_end(state))
    for line in response.split("\n"):
        print(f"  {line}")

    print("  " + "·" * 60)
    print(f"\n  Summary length: {len(response)} chars")
    print(f"  Contains section headers?")
    for header in ["Session Summary", "Cleared", "Newly surfaced", "Still open"]:
        present = header.lower() in response.lower()
        print(f"    {'✓' if present else '✗'}  {header!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    print("═" * 65)
    print("  INTERVIEWER AGENT — UTILITY FUNCTIONS WALKTHROUGH")
    print("═" * 65)

    await demo_select_questions()
    await demo_grade_answer()
    demo_request_confidence_score()
    demo_compute_penalty()
    await demo_on_answer()
    await demo_on_confidence_received()
    demo_compute_misconception_diff()
    await demo_on_session_end()

    print(f"\n{'─' * 65}")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
