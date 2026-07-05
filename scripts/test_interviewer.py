#!/usr/bin/env python3
"""
test_interviewer.py — Full interactive smoke-test for the Interviewer Agent.

WHAT IT DOES:
  Simulates a complete adversarial interview session:
    1. Loads seed episodes (from memory/seed.py) for a topic.
    2. Calls on_session_start() → generates questions preferring failure episodes
       and Track B weak points.
    3. Enters an interactive loop:
         - Presents one question at a time
         - Reads your answer from stdin
         - Streams grade feedback + confidence prompt (on_answer)
         - Reads your confidence score (1–5)
         - Streams penalty feedback (on_confidence_received)
         - Accumulates score across the session
    4. On 'end': calls on_session_end() → streams misconception diff summary.
    5. Exits when all questions are answered or you type 'quit'.

HOW IT DOES IT:
  - All routing flows through interviewer_agent() (the @cognee.agent_memory-
    decorated public wrapper) which delegates to _interviewer_agent_impl().
  - Special sentinels drive the state machine:
      "__session_start__"          — generates questions, streams first one
      "__session_end__"            — streams misconception diff
      "__confidence__:<grade>:<n>" — streams penalty feedback
  - Intermediate turns call on_answer() directly (answer → grade + conf prompt).
  - cognee.recall() is attempted for Track B; failures are silently swallowed.

USAGE:
  From the PROJECT ROOT:
    python scripts/test_interviewer.py
    python scripts/test_interviewer.py --topic os_memory
    python scripts/test_interviewer.py --topic deep_learning --user-id alice

  Topics available: deep_learning (default), os_memory

CONTROLS (during session):
    - Type your answer and press Enter
    - When prompted for confidence, type a number 1–5
    - Type 'end'  to finish early and see the session diff
    - Type 'skip' to skip current question with zero confidence
    - Type 'quit' to exit immediately
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

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
logging.getLogger("agents.interviewer").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("cognee").setLevel(logging.WARNING)

import config  # noqa: F401

from models.schemas import HistoricalEpisode, TutorState
from agents.interviewer import (
    on_session_start,
    on_session_end,
    on_answer,
    on_confidence_received,
    grade_answer,
    compute_penalty,
    compute_misconception_diff,
    interviewer_agent,
    HARSH_MULTIPLIER,
)
from memory.seed import OS_MEMORY_EPISODES, DEEP_LEARNING_EPISODES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 70


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _episodes_from_seed(topic: str) -> list[HistoricalEpisode]:
    return list(DEEP_LEARNING_EPISODES if topic == "deep_learning" else OS_MEMORY_EPISODES)


def _make_state(user_id: str, topic: str) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic=topic,
        current_episode="",
        mode="interviewer",
        session_id=f"interview_{user_id}_{topic}",
        nudge_count=0,
        answer_history=[],
        trait_snapshot=[],
        ingest_needed=False,
    )


async def _stream_and_print(gen, prefix: str = "  🤖 ") -> str:
    """Drain an async generator, printing tokens as they arrive."""
    print(prefix, end="", flush=True)
    collected: list[str] = []
    async for token in gen:
        print(token, end="", flush=True)
        collected.append(token)
    print()
    return "".join(collected)


def _read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  [interrupted]")
        return "quit"


def _parse_confidence(raw: str) -> int | None:
    """Return 1–5 int or None if invalid."""
    try:
        val = int(raw.strip())
        if 1 <= val <= 5:
            return val
    except ValueError:
        pass
    return None


def _penalty_label(grade: str, confidence: int) -> str:
    penalty = compute_penalty(grade, confidence)
    high = confidence in {4, 5}
    if grade == "correct":
        return "no penalty ✓"
    multiplier = f" × {HARSH_MULTIPLIER:.0f} HARSH" if high and grade != "correct" else ""
    return f"penalty = {penalty:.1f}{multiplier}"


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------

async def run_session(topic: str, user_id: str) -> None:
    episodes = _episodes_from_seed(topic)
    if not episodes:
        print(f"  No seed episodes found for topic {topic!r}.")
        return

    state = _make_state(user_id, topic)

    print(f"\n{'═' * 70}")
    print(f"  INTERVIEWER AGENT — ADVERSARIAL SESSION")
    print(f"  Topic   : {topic}")
    print(f"  User ID : {user_id}")
    print(f"  Episodes: {len(episodes)}")
    print(f"{'═' * 70}")
    print("""
  CONTROLS:
    - Type your answer and press Enter
    - When asked for confidence, type a number 1–5
    - Type 'end'  to finish early and see your session diff
    - Type 'skip' to skip with zero confidence (counts as wrong, confidence=1)
    - Type 'quit' to exit immediately
""")

    # ------------------------------------------------------------------
    # Generate questions via on_session_start
    # ------------------------------------------------------------------
    print("  Generating questions from Track A…")
    questions = await on_session_start(state, episodes)

    if not questions:
        print(
            "\n  No questions could be generated — ensure the topic has seed episodes.\n"
        )
        return

    print(f"  Generated {len(questions)} question(s).\n")

    # Track scores
    total_penalty = 0.0
    results_log: list[dict] = []

    # ------------------------------------------------------------------
    # Question loop
    # ------------------------------------------------------------------
    for q_idx, question in enumerate(questions):
        _header(
            f"QUESTION {q_idx + 1}/{len(questions)}"
            f"  —  {question['concept']}"
            + ("  [FAILURE EPISODE]" if question.get("from_failure") else "")
        )
        print(f"\n  {question['question']}\n")

        # --- Read answer ---
        answer = _read_line("  ✏️  Your answer (or 'skip' / 'end' / 'quit'): ")

        if answer.lower() == "quit":
            print("\n  [Session ended by user]")
            _print_session_summary(results_log, total_penalty, questions)
            return

        if answer.lower() == "end":
            break

        if answer.lower() == "skip":
            answer = ""
            print("  [Skipped — treated as wrong with confidence 1]")

        if not answer:
            # Empty → wrong with confidence 1
            grade = "wrong"
            confidence = 1
        else:
            # --- Stream grade + confidence prompt ---
            print()
            await _stream_and_print(on_answer(question, answer), prefix="  🤖 ")

            # --- Read confidence ---
            confidence = None
            while confidence is None:
                raw = _read_line("\n  🎯 Confidence (1–5): ")
                if raw.lower() == "quit":
                    print("\n  [Session ended by user]")
                    _print_session_summary(results_log, total_penalty, questions)
                    return
                confidence = _parse_confidence(raw)
                if confidence is None:
                    print("  Please enter a number between 1 and 5.")

            # Grade the answer (need the grade for penalty computation + routing)
            grade = await grade_answer(question, answer)

            # --- Stream penalty feedback ---
            print()
            await _stream_and_print(
                on_confidence_received(grade, confidence),
                prefix="  🤖 ",
            )

        # --- Compute penalty and log ---
        penalty = compute_penalty(grade, confidence)
        total_penalty += penalty
        results_log.append(
            {
                "q_idx": q_idx + 1,
                "concept": question["concept"],
                "episode_id": question["episode_id"],
                "from_failure": question.get("from_failure", False),
                "answer": answer,
                "grade": grade,
                "confidence": confidence,
                "penalty": penalty,
            }
        )

        high_conf_wrong = grade in {"wrong", "partial"} and confidence in {4, 5}
        status_icon = "✅" if grade == "correct" else ("🔶" if grade == "partial" else "✗")
        print(
            f"\n  {status_icon} grade={grade!r}  conf={confidence}  "
            f"{_penalty_label(grade, confidence)}"
            + ("  ⚠️  overconfidence!" if high_conf_wrong else "")
        )
        print(f"  Running total penalty: {total_penalty:.1f}")

    # ------------------------------------------------------------------
    # Session end — misconception diff
    # ------------------------------------------------------------------
    _header("SESSION END — Misconception Diff")
    print()
    await _stream_and_print(on_session_end(state), prefix="  ")

    _print_session_summary(results_log, total_penalty, questions)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_session_summary(
    log: list[dict],
    total_penalty: float,
    questions: list[dict],
) -> None:
    _header("SESSION SUMMARY")

    answered = len(log)
    total_q = len(questions)
    correct = sum(1 for r in log if r["grade"] == "correct")
    partial = sum(1 for r in log if r["grade"] == "partial")
    wrong = sum(1 for r in log if r["grade"] == "wrong")
    overconfident = sum(
        1 for r in log if r["grade"] in {"wrong", "partial"} and r["confidence"] in {4, 5}
    )

    print(f"  Questions answered : {answered} / {total_q}")
    print(f"  Correct            : {correct}")
    print(f"  Partial            : {partial}")
    print(f"  Wrong              : {wrong}")
    print(f"  Overconfident wrong: {overconfident}")
    print(f"  Total penalty      : {total_penalty:.1f}")

    if log:
        print(f"\n  Question log:")
        for r in log:
            icon = "✅" if r["grade"] == "correct" else ("🔶" if r["grade"] == "partial" else "✗")
            conf_flag = " ⚠️" if r["grade"] in {"wrong", "partial"} and r["confidence"] in {4, 5} else ""
            print(
                f"    {r['q_idx']:>2}. {icon}  grade={r['grade']:<7}  "
                f"conf={r['confidence']}  penalty={r['penalty']:.1f}{conf_flag}  "
                f"ep={r['episode_id']!r:<25}  "
                f"answer={r['answer'][:40]!r}"
            )

    grade_counts: dict[str, int] = {}
    for r in log:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
    if grade_counts:
        print(f"\n  Grade breakdown:")
        for grade, count in sorted(grade_counts.items()):
            bar = "█" * count
            print(f"    {grade:<10}: {bar} ({count})")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Interviewer Agent smoke-test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_interviewer.py
  python scripts/test_interviewer.py --topic os_memory
  python scripts/test_interviewer.py --topic deep_learning --user-id alice
        """,
    )
    parser.add_argument(
        "--topic",
        choices=["deep_learning", "os_memory"],
        default="deep_learning",
        help="Seed topic to use (default: deep_learning)",
    )
    parser.add_argument(
        "--user-id",
        default="demo_user",
        help="User ID for Track B recall (default: demo_user)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_session(topic=args.topic, user_id=args.user_id))
