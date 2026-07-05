"""
test_teacher_utils.py — Walkthrough demo for the Teacher Agent utility functions.

Covers:
  1. classify_answer()         — 4-label answer classifier
  2. stuck_fallback()          — structured fallback for stuck learners
  3. _heuristic_classify()     — offline fallback inside the classifier
  4. select_next_episode()     — episode selection with requires/concurrent_with
  5. on_digest()               — digest mode (no episode advance)

No real LLM calls are made unless MISTRAL_API_KEY is set.
If no key is present, litellm raises and the heuristic / fallback paths kick in,
which is exactly what this demo exercises.

Run from the PROJECT ROOT:
    python scripts/test_teacher_utils.py
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

# Load optional real key from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

# Suppress noisy third-party logs
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s — %(message)s")
logging.getLogger("agents.teacher").setLevel(logging.DEBUG)

import config  # noqa: F401

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence, TutorState
from agents.teacher import (
    classify_answer,
    stuck_fallback,
    select_next_episode,
    on_digest,
    _heuristic_classify,
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
    current_episode: str = "",
    answer_history: list[dict] | None = None,
    nudge_count: int = 0,
    user_id: str = "demo_user",
    mode: str = "teacher",
) -> TutorState:
    return TutorState(
        user_id=user_id,
        topic="demo_topic",
        current_episode=current_episode,
        mode=mode,
        session_id="demo_session",
        nudge_count=nudge_count,
        answer_history=answer_history or [],
        trait_snapshot=[],
        ingest_needed=False,
    )


async def stream_to_str(gen) -> str:
    """Collect all tokens from an async generator into a single string."""
    tokens = []
    async for token in gen:
        tokens.append(token)
    return "".join(tokens)


# ---------------------------------------------------------------------------
# 1. classify_answer
# ---------------------------------------------------------------------------

async def demo_classify_answer() -> None:
    sep("1. classify_answer(answer, episode)")
    print("""
  WHAT IT DOES:
    Classifies a learner's answer against a HistoricalEpisode into one of
    four mutually exclusive labels:

      matched-failure  — answer matches the historical approach that FAILED
      matched-success  — answer matches the historical approach that SUCCEEDED
      partial          — on the right track but incomplete
      novel            — proposes something not in the episode at all

  HOW IT DOES IT:
    1. Calls the Mistral LLM with a strict zero-temperature classification prompt.
    2. If the LLM is unavailable / returns garbage, falls back to
       _heuristic_classify() (word-overlap scoring, no LLM).

  Example episode: paging (OS memory management)
""")

    ep_paging = make_episode(
        id="paging",
        concept="Paging",
        problem="How do we give each process its own view of memory without wasting space?",
        solution="Divide memory into fixed-size pages; use a page table to translate virtual→physical addresses.",
        outcome=Outcome.SUCCESS,
        why="Pages eliminate external fragmentation by using uniform fixed-size blocks.",
        source="https://en.wikipedia.org/wiki/Paging",
    )

    test_cases = [
        ("Use fixed-size pages and a page table to map virtual to physical addresses.",
         "→ strong overlap with the SUCCESS solution  → expected: matched-success"),
        ("Divide memory into variable-size segments based on logical structure.",
         "→ describes segmentation (the step before paging, which partially failed)  → expected: partial or matched-failure"),
        ("Just give every process the full RAM and isolate with hardware checks.",
         "→ novel idea not in the episode  → expected: novel"),
        ("I have no idea.",
         "→ too vague  → expected: partial (heuristic: some overlap with problem words)"),
    ]

    for answer, note in test_cases:
        print(f"\n  Answer  : {answer!r}")
        print(f"  Note    : {note}")
        label = await classify_answer(answer, ep_paging)
        print(f"  Result  : {label!r}")


# ---------------------------------------------------------------------------
# 2. _heuristic_classify  (offline word-overlap fallback)
# ---------------------------------------------------------------------------

def demo_heuristic_classify() -> None:
    sep("2. _heuristic_classify(answer, episode)  ← offline fallback inside classifier")
    print("""
  WHAT IT DOES:
    A pure-Python word-overlap scorer used when the LLM is unavailable.
    No network, no API key required.

  HOW IT DOES IT:
    - Tokenises answer, attempted_solution, and problem_posed into word sets.
    - Removes common stop-words.
    - Computes overlap ratios:
        >= 35% overlap with solution → matched-failure / matched-success (by episode outcome)
        >= 15% overlap with solution or problem → partial
        < 15% → novel

  Example episode: backpropagation
""")
    ep_bp = make_episode(
        id="backprop",
        concept="Backpropagation",
        problem="How do we efficiently compute weight gradients in a multi-layer network?",
        solution="Apply the chain rule of calculus layer by layer, propagating error signals backward.",
        outcome=Outcome.SUCCESS,
        why="Chain rule allows O(n) gradient computation instead of O(n²) finite differences.",
    )

    cases = [
        ("Apply the chain rule backward through each layer to propagate gradients.",
         "~50% word overlap with solution → matched-success"),
        ("Use gradient descent to update weights somehow.",
         "~20% overlap → partial"),
        ("Train the network using evolutionary algorithms instead of calculus.",
         "<10% overlap with solution or problem → novel"),
    ]

    print(f"  Episode outcome: {ep_bp.outcome.value}")
    for answer, note in cases:
        label = _heuristic_classify(answer, ep_bp)
        print(f"\n  Answer : {answer!r}")
        print(f"  Note   : {note}")
        print(f"  Result : {label!r}")


# ---------------------------------------------------------------------------
# 3. stuck_fallback
# ---------------------------------------------------------------------------

async def demo_stuck_fallback() -> None:
    sep("3. stuck_fallback(episode)")
    print("""
  WHAT IT DOES:
    Delivers a four-section structured response when a learner has been stuck
    (nudge_count >= 2) on the same episode without progress.

    Required sections (ALWAYS present):
      **Problem framing**   — restates the core problem
      **Solution hint**     — gentle nudge toward the solution (no spoiler)
      **Engineering Insight** — the deeper principle behind the solution
      **Historical note**   — who solved it, when, why it mattered

  HOW IT DOES IT:
    1. Builds a strict prompt instructing the LLM to produce all four sections.
    2. Streams the response token-by-token (AsyncGenerator[str, None]).
    3. If the LLM fails, yields a static fallback that still contains all four
       sections.

  The caller is responsible for resetting nudge_count to 0 afterwards.

  Example: XOR failure episode (deep learning)
""")
    ep_xor = make_episode(
        id="xor_failure",
        concept="XOR Failure (Perceptron Limitation)",
        problem="Can a single perceptron learn the XOR function?",
        solution="Minsky & Papert showed a single-layer perceptron CANNOT learn XOR — it requires a non-linearly separable boundary.",
        outcome=Outcome.FAILURE,
        why="XOR is not linearly separable; a single perceptron only draws a straight line in input space.",
        source="Minsky & Papert, Perceptrons (1969)",
        published_date=date(1969, 1, 1),
    )

    print("  Streaming fallback response (may use LLM or static fallback):\n")
    print("  " + "·" * 60)
    response = await stream_to_str(stuck_fallback(ep_xor))
    # Print indented
    for line in response.split("\n"):
        print(f"  {line}")
    print("  " + "·" * 60)

    # Verify all four sections are present
    required_sections = ["Problem framing", "Solution hint", "Engineering Insight", "Historical note"]
    print("\n  Section presence check:")
    for section in required_sections:
        present = section.lower() in response.lower()
        print(f"    {'✓' if present else '✗'}  {section!r}")


# ---------------------------------------------------------------------------
# 4. select_next_episode
# ---------------------------------------------------------------------------

async def demo_select_next_episode() -> None:
    sep("4. select_next_episode(state, all_episodes)")
    print("""
  WHAT IT DOES:
    Picks the best next episode for the learner, in priority order:
      1. Unresolved mandatory prerequisites of the current episode
         (episodes listed in current_episode.requires that aren't resolved)
      2. Natural next-steps — episodes that list current_episode in THEIR requires
      3. Concurrent siblings — current_episode.concurrent_with peers
      4. Any other unresolved episode (fallback)
      5. None — all episodes resolved (session complete)

    Within each tier, episodes are ranked by misconception overlap score
    (how many Track B active misconceptions appear in the episode concept).
    cognee.recall() is called first to get Track B traits; if it fails the
    selection still works without trait weighting.

  HOW IT DOES IT:
    1. Calls cognee.recall(graph_name="user_{user_id}_traits") for Track B.
    2. Extracts active misconception concept strings.
    3. Builds resolved ID set from answer_history (classification==matched-success).
    4. Traverses requires/concurrent_with edges from current episode.
    5. Scores each candidate by misconception overlap, returns highest-scoring
       from the highest-priority tier.

  Example graph:
      A ← B ← C       (A must come before B, B before C)
              ↕ concurrent
              D        (D is concurrent with C, no ordering implied)
""")
    # Build episodes
    ep_a = make_episode("A", "Base+Limit Registers",
        "How do we isolate process memory cheaply?",
        "Use two registers: base (start) and limit (size) for each process.",
        outcome=Outcome.SUCCESS, requires=[])
    ep_b = make_episode("B", "Segmentation",
        "How do we map logical program structure to memory?",
        "Divide memory into variable-size segments matching code/data/stack.",
        outcome=Outcome.PARTIAL, requires=["A"])
    ep_c = make_episode("C", "External Fragmentation",
        "Why does segmentation lead to wasted memory holes?",
        "Variable segments create gaps that are too small to reuse.",
        outcome=Outcome.FAILURE, requires=["B"])
    ep_d = make_episode("D", "Internal Fragmentation (concurrent study)",
        "What about wasted space inside allocations?",
        "Fixed-size allocation units waste space at the end of each block.",
        outcome=Outcome.FAILURE, requires=["B"], concurrent_with=["C"])
    ep_e = make_episode("E", "Paging",
        "How do we eliminate external fragmentation?",
        "Use fixed-size pages; page table translates virtual→physical.",
        outcome=Outcome.SUCCESS, requires=["C"])

    all_eps = [ep_a, ep_b, ep_c, ep_d, ep_e]

    # --- Case A: prerequisite not yet resolved ---
    print("\n  Case A — current=B, A not resolved yet → must do A first (prerequisite)")
    state_a = make_state(current_episode="B", answer_history=[])
    next_ep = await select_next_episode(state_a, all_eps)
    print(f"    current_episode : B")
    print(f"    A resolved?     : No")
    print(f"    selected        : {next_ep.id if next_ep else 'None'}  (expected: A — unresolved prerequisite)")

    # --- Case B: natural next step ---
    print("\n  Case B — current=A, A resolved → natural next step is B")
    state_b = make_state(
        current_episode="A",
        answer_history=[{"episode_id": "A", "classification": "matched-success", "answer": "x"}],
    )
    next_ep = await select_next_episode(state_b, all_eps)
    print(f"    current_episode : A")
    print(f"    A resolved?     : Yes")
    print(f"    selected        : {next_ep.id if next_ep else 'None'}  (expected: B — natural next step requiring A)")

    # --- Case C: concurrent sibling ---
    print("\n  Case C — current=C, C+D concurrent, B+C resolved, no natural next yet → concurrent sibling D")
    state_c = make_state(
        current_episode="C",
        answer_history=[
            {"episode_id": "A", "classification": "matched-success", "answer": "x"},
            {"episode_id": "B", "classification": "matched-success", "answer": "x"},
            {"episode_id": "C", "classification": "matched-success", "answer": "x"},
        ],
    )
    # E requires C but C is already resolved; D is concurrent with C and not resolved
    next_ep = await select_next_episode(state_c, all_eps)
    print(f"    current_episode : C")
    print(f"    resolved        : A, B, C")
    print(f"    selected        : {next_ep.id if next_ep else 'None'}  (expected: E — natural next step after C)")

    # --- Case D: all resolved → None ---
    print("\n  Case D — all episodes resolved → None (session complete)")
    state_d = make_state(
        current_episode="E",
        answer_history=[
            {"episode_id": ep.id, "classification": "matched-success", "answer": "x"}
            for ep in all_eps
        ],
    )
    next_ep = await select_next_episode(state_d, all_eps)
    print(f"    resolved        : all")
    print(f"    selected        : {next_ep!r}  (expected: None)")


# ---------------------------------------------------------------------------
# 5. on_digest
# ---------------------------------------------------------------------------

async def demo_on_digest() -> None:
    sep("5. on_digest(state, transcript)")
    print("""
  WHAT IT DOES:
    Digest mode — the learner submits a video transcript (or any text they've
    already consumed) and the Teacher summarises it against the Track A
    knowledge graph, WITHOUT:
      - Posing any Socratic questions
      - Advancing state["current_episode"]

    This lets the system acknowledge content the learner has already seen and
    mark relevant episodes as contextually covered, without treating them as
    formally resolved via Socratic dialogue.

  HOW IT DOES IT:
    1. Uses the first 200 chars of the transcript to query cognee.recall()
       on "content_track" (Track A).
    2. Extracts concept names from matching episodes as context.
    3. Streams a structured summary via litellm against those concepts.
    4. state["current_episode"] is deliberately NOT updated.

  Requirements: 6.1, 6.2
""")
    transcript = """\
In this lecture we covered how neural networks learn: first the perceptron,
its single-layer limitation with XOR, then multi-layer perceptrons and how
backpropagation solves the credit assignment problem using the chain rule.
We also touched on vanishing gradients and why LSTM gates were introduced
to preserve long-range dependencies.
"""

    state = make_state(current_episode="xor_failure", mode="digest")
    print(f"  state['current_episode'] before digest: {state['current_episode']!r}")
    print(f"\n  Transcript (first 120 chars):\n    {transcript[:120].strip()!r}\n")
    print("  Streaming digest summary:\n")
    print("  " + "·" * 60)
    response = await stream_to_str(on_digest(state, transcript))
    for line in response.split("\n"):
        print(f"  {line}")
    print("  " + "·" * 60)

    print(f"\n  state['current_episode'] after digest : {state['current_episode']!r}")
    unchanged = state["current_episode"] == "xor_failure"
    print(f"  Episode position unchanged?          : {'✓ Yes' if unchanged else '✗ No — BUG'}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    print("═" * 65)
    print("  TEACHER AGENT — UTILITY FUNCTIONS WALKTHROUGH")
    print("═" * 65)

    await demo_classify_answer()
    demo_heuristic_classify()
    await demo_stuck_fallback()
    await demo_select_next_episode()
    await demo_on_digest()

    print(f"\n{'─' * 65}")
    print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
