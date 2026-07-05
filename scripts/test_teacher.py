#!/usr/bin/env python3
"""
test_teacher.py — Full interactive smoke-test for the Teacher Agent.

WHAT IT DOES:
  Simulates a complete Socratic learning session with the Teacher Agent:
    1. Loads seed episodes (from memory/seed.py) for a topic.
    2. Prints the first episode's problem_posed.
    3. Enters an interactive loop:
         - Read your answer from stdin
         - Classify it (matched-failure / matched-success / partial / novel)
         - Stream the Teacher's Socratic response token-by-token
         - Track nudge_count — if >= 2, deliver stuck_fallback (4 sections)
         - On matched-success: auto-advance to the next episode
    4. Exits when all episodes are resolved or you type 'quit'.

    Optionally: type 'digest' to enter Digest Mode — paste a transcript
    and see it summarised against Track A without advancing your position.

HOW IT DOES IT:
  - All logic flows through teacher_agent() (the @cognee.agent_memory-decorated
    public wrapper) which delegates to _teacher_agent_impl() (the async generator).
  - Mid-session turns call on_user_answer() directly via the state machine.
  - Digest turns call on_digest() directly.
  - cognee.recall() is attempted for Track B traits; failures are silently
    swallowed so the session continues without stored traits.

USAGE:
  From the PROJECT ROOT:
    python scripts/test_teacher.py
    python scripts/test_teacher.py --topic deep_learning
    python scripts/test_teacher.py --topic os_memory --user-id alice

  Topics available: deep_learning (default), os_memory
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
logging.getLogger("agents.teacher").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("cognee").setLevel(logging.WARNING)

import config  # noqa: F401

from models.schemas import HistoricalEpisode, TutorState
from agents.teacher import (
    classify_answer,
    stuck_fallback,
    select_next_episode,
    on_digest,
    on_user_answer,
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
    """Return the seed HistoricalEpisode objects for the given topic."""
    return list(DEEP_LEARNING_EPISODES if topic == "deep_learning" else OS_MEMORY_EPISODES)


def _make_state(user_id: str, topic: str, first_episode_id: str) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic=topic,
        current_episode=first_episode_id,
        mode="teacher",
        session_id=f"demo_{user_id}_{topic}",
        nudge_count=0,
        answer_history=[],
        trait_snapshot=[],
        ingest_needed=False,
    )


def _print_episode_banner(ep: HistoricalEpisode, index: int, total: int) -> None:
    _header(f"EPISODE {index}/{total}  —  {ep.concept}")
    print(f"\n  📖 PROBLEM POSED:\n")
    for line in ep.problem_posed.split(". "):
        if line.strip():
            print(f"     {line.strip()}.")
    source = f"  Source: {ep.source}" if ep.source else ""
    date_str = f"  Date: {ep.published_date}" if ep.published_date else ""
    if source or date_str:
        print(f"\n  [{ep.source_confidence.value}]{date_str}{source}")


async def _stream_and_print(gen, prefix: str = "  🤖 ") -> str:
    """Stream tokens from an async generator, printing them as they arrive."""
    print(prefix, end="", flush=True)
    collected = []
    async for token in gen:
        print(token, end="", flush=True)
        collected.append(token)
    print()  # newline after stream ends
    return "".join(collected)


def _read_answer(prompt: str = "\n  ✏️  Your answer (or 'digest' / 'skip' / 'quit'): ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  [interrupted]")
        return "quit"


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------

async def run_session(topic: str, user_id: str) -> None:
    episodes = _episodes_from_seed(topic)
    if not episodes:
        print(f"  No seed episodes found for topic {topic!r}.")
        return

    # Build episode lookup
    ep_map = {ep.id: ep for ep in episodes}
    state = _make_state(user_id, topic, episodes[0].id)

    print(f"\n{'═' * 70}")
    print(f"  TEACHER AGENT — INTERACTIVE SESSION")
    print(f"  Topic   : {topic}")
    print(f"  User ID : {user_id}")
    print(f"  Episodes: {len(episodes)}")
    print(f"{'═' * 70}")
    print("""
  CONTROLS:
    - Type your answer and press Enter
    - Type 'digest' to paste a transcript for Digest Mode
    - Type 'skip' to skip the current episode (counts as two nudges → fallback)
    - Type 'quit' to end the session

  The Teacher will:
    ✓ Classify your answer (matched-failure / matched-success / partial / novel)
    ✓ Stream a Socratic response
    ✓ Track nudge_count — after 2 stuck nudges, deliver a structured fallback
    ✓ Advance to the next episode when you succeed
    ✓ Show episode position in Track A at all times
""")

    resolved_ids: set[str] = set()
    session_log: list[dict] = []

    def _current_ep() -> HistoricalEpisode | None:
        return ep_map.get(state["current_episode"])

    episode_count = 0

    while True:
        current = _current_ep()
        if current is None:
            print("\n  🎉  All episodes resolved! Session complete.")
            break

        # Count episode visits (not the same as index in list)
        episode_count += 1
        resolved_count = len(resolved_ids)
        _print_episode_banner(current, resolved_count + 1, len(episodes))

        # --- Answer loop for this episode ---
        while True:
            answer = _read_answer()

            if answer.lower() == "quit":
                print("\n  [Session ended by user]")
                _print_session_summary(state, session_log, episodes)
                return

            if answer.lower() == "digest":
                print("\n  📼 DIGEST MODE")
                print("  Paste your transcript below. Type '---END---' on its own line when done:\n")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line.strip() == "---END---":
                        break
                    lines.append(line)
                transcript = "\n".join(lines)
                if not transcript.strip():
                    print("  (No transcript provided — skipping digest)")
                    continue

                old_ep = state["current_episode"]
                state["mode"] = "digest"
                print("\n  Digest summary:\n")
                print("  " + "·" * 60)
                await _stream_and_print(on_digest(state, transcript), prefix="  ")
                print("  " + "·" * 60)
                state["mode"] = "teacher"  # restore

                # Verify episode position unchanged (Requirement 6.2)
                assert state["current_episode"] == old_ep, \
                    f"BUG: digest mode changed current_episode from {old_ep!r} to {state['current_episode']!r}"
                print(f"\n  ✓ Episode position unchanged: {state['current_episode']!r}")
                continue

            if answer.lower() == "skip":
                # Force two nudges to trigger stuck_fallback
                state["nudge_count"] = 2
                print("\n  [Skipped — delivering stuck fallback]\n")
                print("  " + "·" * 60)
                await _stream_and_print(stuck_fallback(current), prefix="  ")
                print("  " + "·" * 60)
                state["nudge_count"] = 0
                # Don't advance; stay on same episode
                continue

            if not answer:
                print("  (Empty answer — please type something)")
                continue

            # --- Normal answer processing ---
            print()

            # 1. Classify
            label = await classify_answer(answer, current)
            classification_emoji = {
                "matched-success": "✅",
                "matched-failure": "⚠️ ",
                "partial": "🔶",
                "novel": "💡",
            }
            print(f"  {classification_emoji.get(label, '?')} Classification: {label!r}")
            print(f"  nudge_count before: {state['nudge_count']}")

            # 2. Log
            session_log.append({
                "episode_id": current.id,
                "concept": current.concept,
                "answer": answer,
                "label": label,
                "nudge_before": state["nudge_count"],
            })

            # 3. Stream branched response via on_user_answer
            print()
            response = await _stream_and_print(
                on_user_answer(state, answer, current),
                prefix="  🤖 ",
            )

            print(f"\n  nudge_count after : {state['nudge_count']}")

            # 4. If matched-success: advance to next episode
            if label == "matched-success":
                resolved_ids.add(current.id)
                print(f"\n  ✓ Episode '{current.concept}' resolved.")

                next_ep = await select_next_episode(state, episodes)
                if next_ep is None:
                    print("\n  🎉 All episodes resolved! Session complete.")
                    _print_session_summary(state, session_log, episodes)
                    return

                state["current_episode"] = next_ep.id
                print(f"  → Next episode: {next_ep.concept!r}")
                break  # exit inner loop → re-enter outer loop with new episode

            # Stay on same episode for partial / novel / matched-failure


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

def _print_session_summary(
    state: TutorState,
    log: list[dict],
    all_episodes: list[HistoricalEpisode],
) -> None:
    _header("SESSION SUMMARY")
    total = len(all_episodes)
    resolved = sum(
        1 for entry in state["answer_history"]
        if entry.get("classification") == "matched-success"
    )
    print(f"  Episodes resolved : {resolved} / {total}")
    print(f"  Total turns       : {len(log)}")

    if log:
        print(f"\n  Turn log:")
        for i, entry in enumerate(log, 1):
            print(
                f"    {i:>3}. [{entry['label']:<16}] "
                f"nudge={entry['nudge_before']}  "
                f"ep={entry['episode_id']!r:<25}  "
                f"answer={entry['answer'][:50]!r}"
            )

    label_counts: dict[str, int] = {}
    for entry in log:
        label_counts[entry["label"]] = label_counts.get(entry["label"], 0) + 1

    if label_counts:
        print(f"\n  Classification breakdown:")
        for label, count in sorted(label_counts.items()):
            bar = "█" * count
            print(f"    {label:<18}: {bar} ({count})")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Teacher Agent smoke-test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_teacher.py
  python scripts/test_teacher.py --topic os_memory
  python scripts/test_teacher.py --topic deep_learning --user-id alice
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
