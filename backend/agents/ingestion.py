"""
Ingestion Agent — graph-builder-driven discovery pipeline.

IMPORTANT: backend.config must be imported first (before any cognee import).

PUBLIC API (preserved):
- ingestion_agent.run(topic: str, video_ids: list[str] | None = None) -> list[HistoricalEpisode]

Pipeline:
1. graph_builder.generate_dependency_graph(topic)
   → OpenAlex paper discovery + recursive reference expansion
   → LLM builds dependency graph from paper evidence
   → returns {nodes: [ordered concepts], edges: [[prereq, concept], ...]}
2. For each concept node (in order): fetch Wikipedia + arXiv in parallel
3. Extract HistoricalEpisode from each source using LLM
4. Wire up requires[] using graph edges
5. narrative_sort to finalize order
6. Write to memory / return
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict, deque
from datetime import date
from typing import Callable, Optional

# --- config MUST come before cognee ---
import config  # noqa: F401

import litellm

from memory.gateway import AgentRole, MemoryGateway
from models.schemas import HistoricalEpisode, Outcome, SourceConfidence

logger = logging.getLogger(__name__)


class TransientFetchError(Exception):
    """Raised when a fetch operation fails transiently and may be retried."""


_LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/mistral-small-latest")

_OUTCOME_MAP = {
    "success": Outcome.SUCCESS,
    "failure": Outcome.FAILURE,
    "partial": Outcome.PARTIAL,
}


def _parse_outcome(raw: str) -> Outcome:
    return _OUTCOME_MAP.get(raw.lower().strip(), Outcome.PARTIAL)


def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Graph-based topic discovery (wraps graph_builder.py)
# ---------------------------------------------------------------------------

async def discover_curriculum(topic: str) -> tuple[list[str], dict[str, list[str]]]:
    """
    Use graph_builder to discover the dependency graph via OpenAlex + LLM.

    Returns:
        nodes   — ordered list of concept strings (foundational → advanced)
        prereqs — dict mapping concept → list of prerequisite concepts
    """
    try:
        from agents.graph_builder import generate_dependency_graph
    except ImportError:
        from backend.agents.graph_builder import generate_dependency_graph

    logger.info("=== GRAPH BUILDER: discovering curriculum for '%s' ===", topic)

    # Run the blocking graph_builder pipeline in a thread
    graph = await asyncio.to_thread(generate_dependency_graph, topic)

    nodes: list[str] = graph["nodes"]
    edges: list[list[str]] = graph["edges"]

    # Build prereq map: concept → [prerequisites]
    prereqs: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        if len(edge) == 2:
            src, dst = edge[0], edge[1]
            if dst in prereqs:
                prereqs[dst].append(src)

    logger.info(
        "Graph builder returned %d concepts, %d edges: %s",
        len(nodes), len(edges),
        nodes
    )
    return nodes, prereqs


# ---------------------------------------------------------------------------
# Raw text collectors (no LLM — just gather source material)
# ---------------------------------------------------------------------------

async def fetch_wikipedia(subtopic: str) -> list[tuple[str, str]]:
    """Return up to 5 (url, text) pairs from Wikipedia for a concept."""
    import wikipediaapi

    def _get_pages(title: str) -> list[tuple[str, str]]:
        wiki = wikipediaapi.Wikipedia(user_agent="FirstPrinciple/1.0", language="en")
        page = wiki.page(title)
        if not page.exists():
            return []
        candidates = [(page.fullurl, (page.summary or page.text or "")[:2000])]
        for linked_title in list(page.links.keys())[:40]:
            linked = wiki.page(linked_title)
            text = linked.summary or linked.text or ""
            if linked.exists() and text:
                candidates.append((linked.fullurl, text[:2000]))
            if len(candidates) >= 5:
                break
        return candidates

    try:
        pages = await asyncio.to_thread(_get_pages, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"Wikipedia fetch failed for '{subtopic}': {exc}") from exc

    if not pages:
        logger.info("Wikipedia page not found for: %r", subtopic)
    return pages


async def fetch_arxiv(subtopic: str) -> list[tuple[str, str]]:
    """Return up to 5 (title, abstract) pairs from arXiv for a concept."""
    import arxiv

    def _search(query: str) -> list:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=5)
        return list(client.results(search))

    try:
        results = await asyncio.to_thread(_search, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"arXiv fetch failed for '{subtopic}': {exc}") from exc

    papers = []
    for r in results[:5]:
        abstract = (r.summary or "").strip()[:2000]
        if abstract:
            papers.append((r.title or r.entry_id, abstract))
    return papers


# ---------------------------------------------------------------------------
# Synthesize exactly 2 episodes per concept from all gathered sources
# ---------------------------------------------------------------------------

async def synthesize_concept_episodes(
    concept: str,
    wiki_sources: list[tuple[str, str]],
    arxiv_sources: list[tuple[str, str]],
) -> list[HistoricalEpisode]:
    """
    Given raw text from up to 5 Wikipedia pages and 5 arXiv papers,
    make a single LLM call that synthesises exactly 2 rich episodes.
    """
    slug = concept.lower().replace(" ", "_")

    # Build a compact evidence block
    sections: list[str] = []
    for i, (url, text) in enumerate(wiki_sources, 1):
        sections.append(f"[Wikipedia {i}] {url}\n{text}")
    for i, (title, abstract) in enumerate(arxiv_sources, 1):
        sections.append(f"[arXiv {i}] {title}\n{abstract}")

    evidence = "\n\n".join(sections)

    system_prompt = (
        "You are a scientific knowledge synthesiser. "
        "Given multiple source excerpts about a concept, produce EXACTLY 2 problem-solving episodes "
        "that capture the most important intellectual milestones, in chronological order.\n\n"
        "Respond with ONLY a JSON array of exactly 2 objects (no markdown). Each object:\n"
        '  "concept": str  — specific name of the concept or milestone\n'
        '  "problem_posed": str  — the core research problem addressed (1-2 sentences)\n'
        '  "attempted_solution": str  — the method or approach proposed (1-2 sentences)\n'
        '  "outcome": "success" | "failure" | "partial"\n'
        '  "why": str  — impact, limitations, or reason for outcome (1-2 sentences)\n'
        '  "published_date": "YYYY-MM-DD" or null\n'
        '  "requires": []\n'
        '  "concurrent_with": []\n'
        "\nDraw on ALL provided sources. Be specific and factual."
    )
    user_prompt = (
        f"Concept: {concept}\n\n"
        f"Sources:\n{evidence}\n\n"
        "Synthesise exactly 2 episodes as a JSON array:"
    )

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content: str = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        raw = json.loads(content)
        if not isinstance(raw, list):
            raise ValueError("Expected JSON array")

        # Enforce exactly 2
        raw = raw[:2]

        episodes: list[HistoricalEpisode] = []
        # pick best source label for attribution
        best_source = arxiv_sources[0][0] if arxiv_sources else (wiki_sources[0][0] if wiki_sources else concept)
        confidence = SourceConfidence.CITED_SOURCE if arxiv_sources else SourceConfidence.NAMED_REFERENCE

        for i, ep in enumerate(raw):
            episodes.append(
                HistoricalEpisode(
                    id=f"{slug}_{i}",
                    concept=ep.get("concept", concept),
                    problem_posed=ep.get("problem_posed", ""),
                    attempted_solution=ep.get("attempted_solution", ""),
                    outcome=_parse_outcome(ep.get("outcome", "partial")),
                    why=ep.get("why", ""),
                    requires=[],
                    concurrent_with=[],
                    source_confidence=confidence,
                    source=best_source,
                    published_date=_parse_date(ep.get("published_date")),
                )
            )
        return episodes

    except Exception as exc:
        logger.warning("Synthesis failed for %r: %s", concept, exc)
        # Fallback: one bare-bones episode
        fallback_text = (arxiv_sources[0][1] if arxiv_sources else (wiki_sources[0][1] if wiki_sources else ""))[:400]
        return [
            HistoricalEpisode(
                id=f"{slug}_0",
                concept=concept,
                problem_posed=f"Understanding {concept}",
                attempted_solution=fallback_text,
                outcome=Outcome.PARTIAL,
                why="LLM synthesis unavailable.",
                source_confidence=SourceConfidence.NAMED_REFERENCE,
                source=concept,
            )
        ]


# ---------------------------------------------------------------------------
# Source confidence tagging
# ---------------------------------------------------------------------------

def tag_source_confidence(episodes: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
    for ep in episodes:
        src = (ep.source or "").lower()
        if "arxiv" in src:
            ep.source_confidence = SourceConfidence.CITED_SOURCE
        elif "wikipedia" in src or "youtube" in src:
            ep.source_confidence = SourceConfidence.NAMED_REFERENCE
        elif not src:
            ep.source_confidence = SourceConfidence.REASONED
    return episodes


# ---------------------------------------------------------------------------
# Narrative sort (topological + date tiebreak)
# ---------------------------------------------------------------------------

def narrative_sort(episodes: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
    ep_by_id: dict[str, HistoricalEpisode] = {ep.id: ep for ep in episodes}
    ids = list(ep_by_id.keys())

    in_degree: dict[str, int] = {eid: 0 for eid in ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    for ep in episodes:
        for req in ep.requires:
            if req in ep_by_id:
                in_degree[ep.id] += 1
                dependents[req].append(ep.id)

    def _date_key(eid: str):
        d = ep_by_id[eid].published_date
        return (1, date.min) if d is None else (0, d)

    queue: deque[str] = deque(
        sorted([eid for eid in ids if in_degree[eid] == 0], key=_date_key)
    )
    sorted_ids: list[str] = []

    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        newly_free = []
        for dep in dependents[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                newly_free.append(dep)
        if newly_free:
            combined = sorted(list(deque(newly_free)) + list(queue), key=_date_key)
            queue = deque(combined)

    cyclic_ids = [eid for eid in ids if eid not in set(sorted_ids)]
    if cyclic_ids:
        cyclic_ids.sort(key=_date_key)
        sorted_ids.extend(cyclic_ids)

    return [ep_by_id[eid] for eid in sorted_ids]


async def fetch_with_retry(fetch_fn: Callable, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return await fetch_fn()
        except TransientFetchError:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------

class IngestionAgent:
    """
    Graph-builder-driven ingestion.

    Step 1: graph_builder discovers concepts via OpenAlex + LLM
    Step 2: fetch Wikipedia + arXiv per concept (parallel)
    Step 3: extract episodes
    Step 4: wire requires[] from graph edges
    Step 5: narrative_sort
    Step 6: write to memory
    """

    def __init__(self) -> None:
        self.gateway = MemoryGateway(role=AgentRole.INGESTION)

    async def run(
        self,
        topic: str,
        video_ids: list[str] | None = None,
    ) -> list[HistoricalEpisode]:
        import cognee  # noqa: PLC0415

        logger.info("=" * 70)
        logger.info("IngestionAgent.run: topic=%r", topic)
        logger.info("=" * 70)

        # ------------------------------------------------------------------
        # Step 1: discover curriculum via graph_builder
        # ------------------------------------------------------------------
        logger.info("STEP 1: Graph builder — discovering dependency graph")
        try:
            nodes, prereqs = await discover_curriculum(topic)
        except Exception as exc:
            logger.warning("Graph builder failed (%s); falling back to topic only", exc)
            nodes = [topic]
            prereqs = {topic: []}

        logger.info("Curriculum order (%d concepts): %s", len(nodes), nodes)

        # ------------------------------------------------------------------
        # Step 2: fetch Wikipedia + arXiv per concept in parallel
        # ------------------------------------------------------------------
        logger.info("STEP 2: Fetching Wikipedia + arXiv for %d concepts", len(nodes))

        async def fetch_concept(concept: str) -> list[HistoricalEpisode]:
            wiki_task = fetch_wikipedia(concept)
            arxiv_task = fetch_arxiv(concept)
            wiki_sources, arxiv_sources = await asyncio.gather(wiki_task, arxiv_task, return_exceptions=True)
            if not isinstance(wiki_sources, list):
                wiki_sources = []
            if not isinstance(arxiv_sources, list):
                arxiv_sources = []
            eps = await synthesize_concept_episodes(concept, wiki_sources, arxiv_sources)
            logger.info("  %s → %d episodes (wiki=%d, arxiv=%d)", concept, len(eps), len(wiki_sources), len(arxiv_sources))
            return eps

        # Fetch all concepts concurrently (but throttle to avoid overwhelming APIs)
        semaphore = asyncio.Semaphore(3)

        async def throttled_fetch(concept: str) -> list[HistoricalEpisode]:
            async with semaphore:
                return await fetch_concept(concept)

        results = await asyncio.gather(*[throttled_fetch(c) for c in nodes])

        # Flatten: map concept → its episodes (first episode = canonical for wiring)
        all_episodes: list[HistoricalEpisode] = []
        concept_to_first_ep_id: dict[str, str] = {}

        for concept, eps in zip(nodes, results):
            if eps:
                concept_to_first_ep_id[concept] = eps[0].id
            all_episodes.extend(eps)

        logger.info("Total episodes before wiring: %d", len(all_episodes))

        # ------------------------------------------------------------------
        # Step 3: wire requires[] from graph edges
        # ------------------------------------------------------------------
        logger.info("STEP 3: Wiring requires[] from graph edges")
        for concept, prereq_concepts in prereqs.items():
            # Find episodes belonging to this concept
            concept_eps = [ep for ep in all_episodes if ep.concept.lower() == concept.lower()]
            prereq_ids = [
                concept_to_first_ep_id[p]
                for p in prereq_concepts
                if p in concept_to_first_ep_id
            ]
            for ep in concept_eps:
                if prereq_ids:
                    ep.requires = prereq_ids
            if prereq_ids:
                logger.info("  '%s' requires %s", concept, prereq_ids)

        # ------------------------------------------------------------------
        # Step 4: tag + sort
        # ------------------------------------------------------------------
        tag_source_confidence(all_episodes)
        sorted_episodes = narrative_sort(all_episodes)

        logger.info("=" * 70)
        logger.info("CURRICULUM ORDER (%d episodes):", len(sorted_episodes))
        for i, ep in enumerate(sorted_episodes, 1):
            req_str = f" ← {len(ep.requires)} prereq(s)" if ep.requires else ""
            logger.info("  %2d. [%s] %s%s", i, ep.source_confidence.value[:6], ep.concept, req_str)
        logger.info("=" * 70)

        # ------------------------------------------------------------------
        # Step 5: write to memory (skip on Cognee error)
        # ------------------------------------------------------------------
        logger.info("STEP 5: Writing to memory (skipping on error)")
        try:
            await self.gateway.add_data_points(sorted_episodes, temporal_cognify=True)
            await cognee.consolidate_entity_descriptions_pipeline()
        except Exception as exc:
            logger.warning("Memory write skipped: %s", exc)

        return sorted_episodes

    async def self_check_recall(self, topic: str) -> bool:
        try:
            import cognee
            results = await cognee.recall(topic)
            return bool(results)
        except Exception as exc:
            logger.warning("self_check_recall failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

ingestion_agent = IngestionAgent()
