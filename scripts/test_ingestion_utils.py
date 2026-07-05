"""
Toy demo for the three utility functions added in task 5.3:
  - tag_source_confidence()
  - narrative_sort()
  - fetch_with_retry()

Prints inputs, intermediates, and outputs so you can see exactly what each
function does. No LLM, no cognee, no network required.

Run from the project root:
    python scripts/test_ingestion_utils.py
"""
from __future__ import annotations

import asyncio
import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("MISTRAL_API_KEY", "dummy")
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

import config  # noqa: F401

from models.schemas import HistoricalEpisode, Outcome, SourceConfidence
from agents.ingestion import TransientFetchError, tag_source_confidence, narrative_sort, fetch_with_retry


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------

def ep(id, requires=None, source=None, confidence=SourceConfidence.REASONED, published_date=None):
    return HistoricalEpisode(
        id=id, concept=id, problem_posed="?", attempted_solution="...",
        outcome=Outcome.PARTIAL, why="toy",
        requires=requires or [], source=source,
        source_confidence=confidence, published_date=published_date,
    )

def conf_label(c: SourceConfidence) -> str:
    return c.value

def sep(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)

def show_episodes(label: str, episodes: list[HistoricalEpisode]) -> None:
    print(f"\n{label}:")
    for e in episodes:
        date_str = str(e.published_date) if e.published_date else "None"
        req_str  = e.requires if e.requires else "[]"
        src_str  = (e.source or "None")[:50]
        print(f"  id={e.id:<12}  confidence={conf_label(e.source_confidence):<16}"
              f"  source={src_str:<50}  requires={req_str}  date={date_str}")


# ---------------------------------------------------------------------------
# 1. tag_source_confidence
# ---------------------------------------------------------------------------

def demo_tag_source_confidence() -> None:
    sep("tag_source_confidence()")
    print("""
  Rule priority:
    1. source contains "arxiv"            → cited_source
    2. source contains "wikipedia"/"youtube" → named_reference
    3. source is None/empty               → reasoned
    4. anything else                      → keep whatever fetcher set
""")

    episodes = [
        ep("arxiv_paper",  source="https://arxiv.org/abs/1706.03762",
           confidence=SourceConfidence.REASONED),          # wrong tag coming in
        ep("wiki_page",    source="https://en.wikipedia.org/wiki/Backpropagation",
           confidence=SourceConfidence.REASONED),          # wrong tag coming in
        ep("yt_video",     source="https://www.youtube.com/watch?v=aircAruvnKk",
           confidence=SourceConfidence.REASONED),          # wrong tag coming in
        ep("no_source",    source=None,
           confidence=SourceConfidence.NAMED_REFERENCE),   # will be overridden → reasoned
        ep("blog_post",    source="https://example.com/blog",
           confidence=SourceConfidence.NAMED_REFERENCE),   # unknown → preserved
    ]

    show_episodes("INPUT  (note: some tags are deliberately wrong)", episodes)

    tag_source_confidence(episodes)

    show_episodes("OUTPUT (after tagging)", episodes)

    print("""
  What happened:
    arxiv_paper  → cited_source    (arxiv in URL)
    wiki_page    → named_reference (wikipedia in URL)
    yt_video     → named_reference (youtube in URL)
    no_source    → reasoned        (no source at all)
    blog_post    → named_reference (preserved — fetcher already set it)
""")


# ---------------------------------------------------------------------------
# 2. narrative_sort
# ---------------------------------------------------------------------------

def demo_narrative_sort() -> None:
    sep("narrative_sort()")
    print("""
  Episodes are sorted by topological order over `requires` edges.
  published_date is used only as a tiebreaker within the same level.
""")

    # --- Case A: linear chain (deliberately shuffled) ---
    print("  Case A — linear chain: C requires B, B requires A")
    a = ep("A", requires=[],    published_date=date(1950, 1, 1))
    b = ep("B", requires=["A"], published_date=date(1960, 1, 1))
    c = ep("C", requires=["B"], published_date=date(1970, 1, 1))
    shuffled = [c, a, b]
    print(f"  input order : {[e.id for e in shuffled]}")
    result = narrative_sort(shuffled)
    print(f"  output order: {[e.id for e in result]}  ← requires edges respected")

    # --- Case B: diamond with date tiebreaker ---
    print("\n  Case B — diamond: E and F both require D; G requires E and F")
    print("           F has an earlier date than E → F should appear before E at the same level")
    d  = ep("D",  requires=[],             published_date=date(1940, 1, 1))
    f2 = ep("F",  requires=["D"],          published_date=date(1945, 1, 1))  # earlier
    e2 = ep("E",  requires=["D"],          published_date=date(1955, 1, 1))  # later
    g  = ep("G",  requires=["E", "F"],     published_date=date(1965, 1, 1))
    shuffled2 = [g, e2, d, f2]
    print(f"  input order : {[e.id for e in shuffled2]}")
    result2 = narrative_sort(shuffled2)
    print(f"  output order: {[e.id for e in result2]}  ← F (1945) before E (1955) at same level")

    # --- Case C: None date sorts last among roots ---
    print("\n  Case C — two roots, one has no date")
    dated   = ep("with_date", published_date=date(1930, 1, 1))
    no_date = ep("no_date",   published_date=None)
    shuffled3 = [no_date, dated]
    print(f"  input order : {[e.id for e in shuffled3]}")
    result3 = narrative_sort(shuffled3)
    print(f"  output order: {[e.id for e in result3]}  ← None date goes last")

    # --- Case D: cycle (graceful fallback) ---
    print("\n  Case D — cycle: X requires Y, Y requires X (plus acyclic root Z)")
    x = ep("X", requires=["Y"])
    y = ep("Y", requires=["X"])
    z = ep("Z", requires=[])
    shuffled4 = [x, y, z]
    print(f"  input order : {[e.id for e in shuffled4]}")
    result4 = narrative_sort(shuffled4)
    print(f"  output order: {[e.id for e in result4]}  ← Z first (acyclic), X/Y appended with warning")


# ---------------------------------------------------------------------------
# 3. fetch_with_retry
# ---------------------------------------------------------------------------

async def demo_fetch_with_retry() -> None:
    sep("fetch_with_retry()")
    print("""
  Retries a coroutine on TransientFetchError with backoff: sleep(2**attempt).
  No tenacity. Uses only asyncio.sleep.
""")

    import agents.ingestion as _ing

    # Capture sleep calls without actually sleeping
    sleep_log: list[float] = []
    async def fake_sleep(s: float) -> None:
        sleep_log.append(s)

    _ing.asyncio.sleep = fake_sleep  # type: ignore[attr-defined]

    try:
        # --- Case A: succeeds immediately ---
        print("  Case A — fetch succeeds on first attempt")
        call_count = [0]
        async def succeed_immediately():
            call_count[0] += 1
            print(f"    → fetch_fn called (attempt {call_count[0]})")
            return "result_ok"

        sleep_log.clear()
        result = await fetch_with_retry(succeed_immediately)
        print(f"    sleeps      : {sleep_log}  (none — no retries needed)")
        print(f"    return value: {result!r}")

        # --- Case B: fails twice, succeeds on 3rd ---
        print("\n  Case B — fails twice, succeeds on third attempt")
        attempts = [0]
        async def fail_twice():
            attempts[0] += 1
            print(f"    → fetch_fn called (attempt {attempts[0]})", end="")
            if attempts[0] < 3:
                print("  ← raises TransientFetchError")
                raise TransientFetchError("transient failure")
            print("  ← succeeds")
            return "recovered"

        sleep_log.clear()
        result2 = await fetch_with_retry(fail_twice, max_attempts=3)
        print(f"    sleeps      : {sleep_log}  (2**0=1s before attempt 2, 2**1=2s before attempt 3)")
        print(f"    return value: {result2!r}")

        # --- Case C: exhausts all attempts ---
        print("\n  Case C — all 3 attempts fail → TransientFetchError re-raised")
        exhausted = [0]
        async def always_fails():
            exhausted[0] += 1
            print(f"    → fetch_fn called (attempt {exhausted[0]})  ← raises TransientFetchError")
            raise TransientFetchError("always broken")

        sleep_log.clear()
        raised = False
        try:
            await fetch_with_retry(always_fails, max_attempts=3)
        except TransientFetchError as exc:
            raised = True
            print(f"    sleeps      : {sleep_log}  (2**0=1s, 2**1=2s before retries; no sleep after last)")
            print(f"    exception   : TransientFetchError({exc!r}) re-raised after {exhausted[0]} attempts")

    finally:
        import asyncio as _asyncio
        _ing.asyncio.sleep = _asyncio.sleep  # restore


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    demo_tag_source_confidence()
    demo_narrative_sort()
    await demo_fetch_with_retry()
    print("\n" + "─"*60)
    print("  Done.")

if __name__ == "__main__":
    asyncio.run(main())
