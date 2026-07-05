#!/usr/bin/env python3
"""
test_ingestion.py — Manual smoke-test for the IngestionAgent.

Usage:
    python scripts/test_ingestion.py "transformer architecture"
    python scripts/test_ingestion.py "operating system scheduling"

Prints every intermediate step with clear section headers so you can
verify decomposition, per-source episode counts, tagging, sort order,
and the final episode list.

Run from the PROJECT ROOT (not backend/):
    python scripts/test_ingestion.py "<your topic>"
"""
from __future__ import annotations

import asyncio
import logging
import sys
import os
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Path setup — add backend/ so all imports resolve
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

# ---------------------------------------------------------------------------
# Logging — show INFO from our modules, suppress noisy third-party logs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("cognee").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Now import backend modules (config must come first)
# ---------------------------------------------------------------------------
import config  # noqa: F401 — sets up cognee + LiteLLM env vars

from agents.ingestion import (
    IngestionAgent,
    decompose_topic,
    fetch_wikipedia,
    fetch_arxiv,
    tag_source_confidence,
    narrative_sort,
)
from models.schemas import HistoricalEpisode


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 70

def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)

def _print_episode(ep: HistoricalEpisode, index: int) -> None:
    print(f"\n  [{index}] {ep.id}")
    print(f"      concept    : {ep.concept}")
    print(f"      outcome    : {ep.outcome.value}")
    print(f"      confidence : {ep.source_confidence.value}")
    print(f"      date       : {ep.published_date}")
    print(f"      requires   : {ep.requires}")
    print(f"      concurrent : {ep.concurrent_with}")
    print(f"      source     : {(ep.source or '')[:80]}")
    print(f"      problem    : {ep.problem_posed[:120]}")
    print(f"      solution   : {ep.attempted_solution[:120]}")
    print(f"      why        : {ep.why[:120]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(topic: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  INGESTION AGENT SMOKE TEST")
    print(f"  topic: {topic!r}")
    print(f"{'═' * 70}")

    # ------------------------------------------------------------------
    # Step 1: Decompose topic
    # ------------------------------------------------------------------
    _header("STEP 1 — Topic decomposition")
    subtopics = await decompose_topic(topic)
    print(f"  Decomposed into {len(subtopics)} subtopic(s):")
    for i, s in enumerate(subtopics, 1):
        print(f"    {i}. {s}")

    # ------------------------------------------------------------------
    # Step 2: Fetch per source (first subtopic only for speed in manual test)
    # ------------------------------------------------------------------
    _header("STEP 2 — Wikipedia fetch (first subtopic only)")
    first = subtopics[0]
    wiki_eps = await fetch_wikipedia(first)
    print(f"  Got {len(wiki_eps)} episode(s) from Wikipedia for {first!r}")
    for i, ep in enumerate(wiki_eps):
        _print_episode(ep, i)

    _header("STEP 3 — arXiv fetch (first subtopic only)")
    arxiv_eps = await fetch_arxiv(first)
    print(f"  Got {len(arxiv_eps)} episode(s) from arXiv for {first!r}")
    for i, ep in enumerate(arxiv_eps):
        _print_episode(ep, i)

    # ------------------------------------------------------------------
    # Step 3: Full pipeline via IngestionAgent
    # (skips cognee write — we just call the fetch+sort parts directly)
    # ------------------------------------------------------------------
    _header("STEP 4 — Full fetch across ALL subtopics (no cognee write)")
    all_episodes: list[HistoricalEpisode] = []
    for subtopic in subtopics:
        w = await fetch_wikipedia(subtopic)
        a = await fetch_arxiv(subtopic)
        all_episodes.extend(w)
        all_episodes.extend(a)
        print(f"  {subtopic!r}: wiki={len(w)}, arxiv={len(a)}")

    print(f"\n  Total raw episodes: {len(all_episodes)}")

    # ------------------------------------------------------------------
    # Step 4: Tag confidence
    # ------------------------------------------------------------------
    _header("STEP 5 — Source confidence tagging")
    tag_source_confidence(all_episodes)
    from collections import Counter
    counts = Counter(ep.source_confidence.value for ep in all_episodes)
    for tier, n in sorted(counts.items()):
        print(f"  {tier}: {n}")

    # ------------------------------------------------------------------
    # Step 5: Narrative sort
    # ------------------------------------------------------------------
    _header("STEP 6 — Narrative sort (topological + date tiebreak)")
    sorted_eps = narrative_sort(all_episodes)
    print(f"  Sorted {len(sorted_eps)} episodes. Order:")
    for i, ep in enumerate(sorted_eps):
        deps = f"  ← {ep.requires}" if ep.requires else ""
        conc = f"  ∥ {ep.concurrent_with}" if ep.concurrent_with else ""
        print(f"    {i+1:>3}. [{ep.source_confidence.value[:6]}] {ep.concept}{deps}{conc}")

    # ------------------------------------------------------------------
    # Step 6: Full details of sorted episodes
    # ------------------------------------------------------------------
    _header("STEP 7 — Full episode details (sorted order)")
    for i, ep in enumerate(sorted_eps):
        _print_episode(ep, i + 1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _header("SUMMARY")
    print(f"  Topic         : {topic!r}")
    print(f"  Subtopics     : {len(subtopics)}")
    print(f"  Total episodes: {len(sorted_eps)}")
    has_requires = sum(1 for ep in sorted_eps if ep.requires)
    has_concurrent = sum(1 for ep in sorted_eps if ep.concurrent_with)
    print(f"  With requires : {has_requires}")
    print(f"  With concurrent: {has_concurrent}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_ingestion.py \"<topic>\"")
        print('Example: python scripts/test_ingestion.py "transformer architecture"')
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    asyncio.run(run(topic))
