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
# Wikipedia fetch
# ---------------------------------------------------------------------------

async def fetch_wikipedia(subtopic: str) -> list[HistoricalEpisode]:
    import wikipediaapi

    def _get_pages(title: str):
        wiki = wikipediaapi.Wikipedia(user_agent="FirstPrinciple/1.0", language="en")
        page = wiki.page(title)
        if not page.exists():
            return []
        # collect up to 5 pages: the main page + related linked pages
        candidates = [(page.fullurl, page.summary or page.text or "")]
        for linked_title in list(page.links.keys())[:30]:
            linked = wiki.page(linked_title)
            if linked.exists() and (linked.summary or linked.text):
                candidates.append((linked.fullurl, linked.summary or linked.text or ""))
            if len(candidates) >= 5:
                break
        return candidates

    try:
        pages = await asyncio.to_thread(_get_pages, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"Wikipedia fetch failed for '{subtopic}': {exc}") from exc

    if not pages:
        logger.info("Wikipedia page not found for: %r", subtopic)
        return []

    all_episodes: list[HistoricalEpisode] = []
    for url, text in pages:
        text = text[:3000]
        if not text:
            continue
        try:
            eps = await _extract_episodes_from_text(subtopic, text, url)
            all_episodes.extend(eps)
        except Exception as exc:
            logger.warning("Wikipedia extraction failed for %r: %s", url, exc)

    return all_episodes


# ---------------------------------------------------------------------------
# arXiv fetch
# ---------------------------------------------------------------------------

async def fetch_arxiv(subtopic: str) -> list[HistoricalEpisode]:
    import arxiv

    def _search(query: str) -> list:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=5)
        return list(client.results(search))

    try:
        results = await asyncio.to_thread(_search, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"arXiv fetch failed for '{subtopic}': {exc}") from exc

    if not results:
        return []

    all_episodes: list[HistoricalEpisode] = []
    for result in results[:5]:
        abstract = (result.summary or "").strip()[:3000]
        if not abstract:
            continue

        source_label = result.title or result.entry_id
        try:
            episodes = await _extract_episodes_from_text(subtopic, abstract, source_label)
            for ep in episodes:
                ep.source_confidence = SourceConfidence.CITED_SOURCE
                ep.source = source_label
            all_episodes.extend(episodes)
        except Exception as exc:
            logger.warning("arXiv extraction failed for %r: %s", result.title, exc)

    return all_episodes


# ---------------------------------------------------------------------------
# Episode extraction from text
# ---------------------------------------------------------------------------

async def _extract_episodes_from_text(
    subtopic: str, text: str, source_url: str
) -> list[HistoricalEpisode]:
    system_prompt = (
        "You are a historical episode extractor. "
        "Given a text excerpt about a technical or scientific topic, "
        "extract 1-3 key problem-solving episodes IN CHRONOLOGICAL ORDER.\n\n"
        "Respond with ONLY a JSON array (no markdown, no extra text) where each element has:\n"
        '  "concept": str  — specific concept name (e.g. "Backpropagation", "Attention Mechanism")\n'
        '  "problem_posed": str  — the core research problem or open question addressed\n'
        '  "attempted_solution": str  — the approach, method, or technique proposed\n'
        '  "outcome": "success" | "failure" | "partial"\n'
        '  "why": str  — why it succeeded/failed/was partial; limitations or impact\n'
        '  "published_date": "YYYY-MM-DD" or null\n'
        '  "requires": []\n'
        '  "concurrent_with": []\n'
        "\nBe specific and factual. Each field should be 1-2 sentences max."
    )
    user_prompt = f"Topic: {subtopic}\n\nExcerpt:\n{text}\n\nExtract episodes as JSON array:"

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content: str = response.choices[0].message.content or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        raw_episodes = json.loads(content)
        if not isinstance(raw_episodes, list):
            raise ValueError("Expected a JSON array")

        slug = subtopic.lower().replace(" ", "_")
        episodes: list[HistoricalEpisode] = []
        for i, ep in enumerate(raw_episodes):
            episodes.append(
                HistoricalEpisode(
                    id=f"{slug}_{i}",
                    concept=ep.get("concept", subtopic),
                    problem_posed=ep.get("problem_posed", ""),
                    attempted_solution=ep.get("attempted_solution", ""),
                    outcome=_parse_outcome(ep.get("outcome", "partial")),
                    why=ep.get("why", ""),
                    requires=[],  # wired later from graph edges
                    concurrent_with=[],
                    source_confidence=SourceConfidence.NAMED_REFERENCE,
                    source=source_url,
                    published_date=_parse_date(ep.get("published_date")),
                )
            )
        return episodes

    except Exception as exc:
        logger.warning("LLM extraction failed for %r: %s", subtopic, exc)
        return [
            HistoricalEpisode(
                id=f"{subtopic.lower().replace(' ', '_')}_0",
                concept=subtopic,
                problem_posed=f"Understanding {subtopic}",
                attempted_solution=text[:400],
                outcome=Outcome.PARTIAL,
                why="Extracted from source; LLM structuring unavailable.",
                source_confidence=SourceConfidence.NAMED_REFERENCE,
                source=source_url,
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
            wiki_eps, arxiv_eps = await asyncio.gather(wiki_task, arxiv_task, return_exceptions=True)
            eps = []
            if isinstance(wiki_eps, list):
                eps.extend(wiki_eps)
            if isinstance(arxiv_eps, list):
                eps.extend(arxiv_eps)
            logger.info("  %s → %d episodes", concept, len(eps))
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
