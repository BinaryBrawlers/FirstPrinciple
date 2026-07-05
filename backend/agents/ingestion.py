"""
Ingestion Agent — decomposes a topic into subtopics, fetches Wikipedia content,
and synthesises HistoricalEpisode objects from each subtopic.

IMPORTANT: backend.config must be imported first (before any cognee import)
to apply the LiteLLM patch and set COGNEE_SKIP_CONNECTION_TEST.
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

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TransientFetchError(Exception):
    """Raised when a fetch operation fails transiently and may be retried."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# decompose_topic
# ---------------------------------------------------------------------------


async def decompose_topic(topic: str) -> list[str]:
    """
    Use the configured LLM to decompose *topic* into 4-8 constituent subtopics.

    Falls back to ``[topic]`` if the LLM call or parsing fails.
    """
    system_prompt = (
        "You are a knowledge decomposition assistant. "
        "When given a topic, respond with ONLY a JSON array of 4 to 8 strings, "
        "each string being a distinct constituent subtopic of that topic. "
        "Do not include any explanation or markdown fences — only the raw JSON array."
    )
    user_prompt = f"Decompose the following topic into constituent subtopics:\n\n{topic}"

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content: str = response.choices[0].message.content or ""
        # Strip markdown fences if the LLM adds them despite instructions
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        subtopics = json.loads(content)
        if isinstance(subtopics, list) and subtopics:
            return [str(s).strip() for s in subtopics if str(s).strip()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("decompose_topic failed, using fallback: %s", exc)

    return [topic]


# ---------------------------------------------------------------------------
# fetch_wikipedia
# ---------------------------------------------------------------------------


async def fetch_wikipedia(subtopic: str) -> list[HistoricalEpisode]:
    """
    Fetch Wikipedia content for *subtopic* and synthesise HistoricalEpisode
    objects using the LLM.

    Returns an empty list if the page does not exist.
    Falls back to a single synthesised episode if LLM extraction fails.
    """
    import wikipediaapi  # imported here to avoid module-level side-effects

    def _get_page(title: str):
        wiki = wikipediaapi.Wikipedia(user_agent="FirstPrinciple/1.0", language="en")
        return wiki.page(title)

    # Run the blocking wikipediaapi call in a thread to avoid blocking the event loop
    try:
        page = await asyncio.to_thread(_get_page, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"Wikipedia fetch failed for '{subtopic}': {exc}") from exc

    if not page.exists():
        logger.info("Wikipedia page not found for subtopic: %r", subtopic)
        return []

    # Truncate to keep LLM context manageable
    text = page.summary[:4000] if page.summary else (page.text[:4000] if page.text else "")
    if not text:
        logger.info("Wikipedia page for %r has no content", subtopic)
        return []

    episodes = await _extract_episodes_from_text(subtopic, text, page.fullurl)
    return episodes


async def _extract_episodes_from_text(
    subtopic: str, text: str, source_url: str
) -> list[HistoricalEpisode]:
    """Ask the LLM to extract structured episode data from Wikipedia text."""
    system_prompt = (
        "You are a historical episode extractor. "
        "Given a Wikipedia excerpt about a technical or scientific topic, "
        "extract the key problem-solving episodes from the history of that topic. "
        "Respond with ONLY a JSON array (no markdown, no extra text) where each element has:\n"
        '  "concept": str — the core concept or idea\n'
        '  "problem_posed": str — the original problem that was being solved\n'
        '  "attempted_solution": str — how researchers/engineers tried to solve it\n'
        '  "outcome": "success" | "failure" | "partial"\n'
        '  "why": str — why the outcome occurred (causal explanation)\n'
        '  "published_date": "YYYY-MM-DD" or null\n'
        "Return 1–3 episodes maximum. Keep each field concise (1–2 sentences)."
    )
    user_prompt = (
        f"Topic: {subtopic}\n\nWikipedia excerpt:\n{text}\n\n"
        "Extract historical episodes as a JSON array."
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

        episodes: list[HistoricalEpisode] = []
        slug = subtopic.lower().replace(" ", "_")
        for i, ep in enumerate(raw_episodes):
            episodes.append(
                HistoricalEpisode(
                    id=f"wiki_{slug}_{i}",
                    concept=ep.get("concept", subtopic),
                    problem_posed=ep.get("problem_posed", ""),
                    attempted_solution=ep.get("attempted_solution", ""),
                    outcome=_parse_outcome(ep.get("outcome", "partial")),
                    why=ep.get("why", ""),
                    source_confidence=SourceConfidence.NAMED_REFERENCE,
                    source=source_url,
                    published_date=_parse_date(ep.get("published_date")),
                )
            )
        return episodes

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM episode extraction failed for %r, using text-based fallback: %s",
            subtopic,
            exc,
        )
        # Graceful fallback: synthesise a single episode from the raw text
        return [
            HistoricalEpisode(
                id=f"wiki_{subtopic.lower().replace(' ', '_')}_0",
                concept=subtopic,
                problem_posed=f"Understanding the fundamentals of {subtopic}",
                attempted_solution=text[:500],
                outcome=Outcome.PARTIAL,
                why="Extracted from Wikipedia summary; LLM structuring was unavailable.",
                source_confidence=SourceConfidence.NAMED_REFERENCE,
                source=source_url,
            )
        ]


# ---------------------------------------------------------------------------
# fetch_arxiv
# ---------------------------------------------------------------------------


async def fetch_arxiv(subtopic: str) -> list[HistoricalEpisode]:
    """
    Search arXiv for papers related to *subtopic*, extract HistoricalEpisode
    objects from each paper's abstract, and tag them with
    ``source_confidence=SourceConfidence.CITED_SOURCE``.

    Only the abstract (``result.summary``) is used — no full-paper PDF parsing.
    Returns an empty list if no papers are found or all extractions fail.
    Raises :exc:`TransientFetchError` on network-level failures.
    """
    import arxiv  # imported here to avoid module-level side-effects

    def _search(query: str) -> list:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=5)
        return list(client.results(search))

    try:
        results = await asyncio.to_thread(_search, subtopic)
    except Exception as exc:
        raise TransientFetchError(f"arXiv fetch failed for '{subtopic}': {exc}") from exc

    if not results:
        logger.info("arXiv: no results found for subtopic: %r", subtopic)
        return []

    all_episodes: list[HistoricalEpisode] = []
    for result in results:
        abstract = (result.summary or "").strip()
        if not abstract:
            logger.info("arXiv: skipping paper with empty abstract: %r", result.title)
            continue

        # Use abstract as the text for episode extraction
        text = abstract[:4000]
        source_label = result.title or result.entry_id

        try:
            episodes = await _extract_episodes_from_text(subtopic, text, source_label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("arXiv: episode extraction failed for %r: %s", result.title, exc)
            continue

        # Override source_confidence to CITED_SOURCE and set source to paper title/ID
        for ep in episodes:
            ep.source_confidence = SourceConfidence.CITED_SOURCE
            ep.source = source_label

        all_episodes.extend(episodes)
        logger.info(
            "arXiv: extracted %d episode(s) from paper %r", len(episodes), result.title
        )

    return all_episodes


# ---------------------------------------------------------------------------
# fetch_youtube
# ---------------------------------------------------------------------------


async def fetch_youtube(video_ids: list[str]) -> list[HistoricalEpisode]:
    """
    Fetch YouTube transcripts for *video_ids*, extract HistoricalEpisode objects
    from each transcript, and tag them with
    ``source_confidence=SourceConfidence.NAMED_REFERENCE``.

    Videos with unavailable transcripts are skipped with a warning.
    Raises :exc:`TransientFetchError` on network-level failures.
    Returns an empty list if all transcripts are unavailable or extraction fails.
    """
    from youtube_transcript_api import (  # imported here to avoid module-level side-effects
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    all_episodes: list[HistoricalEpisode] = []

    for video_id in video_ids:
        # Fetch transcript in a thread to avoid blocking the event loop
        def _get_transcript(vid: str):
            return YouTubeTranscriptApi.get_transcript(vid)

        try:
            transcript_data = await asyncio.to_thread(_get_transcript, video_id)
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
            logger.warning(
                "fetch_youtube: transcript unavailable for video %r: %s", video_id, exc
            )
            continue
        except Exception as exc:
            raise TransientFetchError(
                f"YouTube transcript fetch failed for video '{video_id}': {exc}"
            ) from exc

        # Join transcript segments into a single string
        text = " ".join(segment.get("text", "") for segment in transcript_data).strip()
        if not text:
            logger.info("fetch_youtube: empty transcript for video %r, skipping", video_id)
            continue

        source_label = f"https://www.youtube.com/watch?v={video_id}"

        # Truncate to keep LLM context manageable
        text = text[:4000]

        try:
            episodes = await _extract_episodes_from_text(video_id, text, source_label)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fetch_youtube: episode extraction failed for video %r: %s", video_id, exc
            )
            continue

        # Override source_confidence to NAMED_REFERENCE and set source to YouTube URL
        for ep in episodes:
            ep.source_confidence = SourceConfidence.NAMED_REFERENCE
            ep.source = source_label

        all_episodes.extend(episodes)
        logger.info(
            "fetch_youtube: extracted %d episode(s) from video %r", len(episodes), video_id
        )

    return all_episodes


# ---------------------------------------------------------------------------
# tag_source_confidence
# ---------------------------------------------------------------------------


def tag_source_confidence(
    episodes: list[HistoricalEpisode],
) -> list[HistoricalEpisode]:
    """
    Apply three-tier source-confidence tagging rules to *episodes* in-place
    and return the same list.

    Rules (applied in priority order):
    1. ``cited_source``    — source contains "arxiv" (case-insensitive)
    2. ``named_reference`` — source contains "wikipedia" or "youtube"
    3. ``reasoned``        — source is None/empty, or none of the above match
                             and current tag is already REASONED; otherwise
                             the fetcher-assigned tag is preserved.
    """
    for ep in episodes:
        src = (ep.source or "").lower()
        if "arxiv" in src:
            ep.source_confidence = SourceConfidence.CITED_SOURCE
        elif "wikipedia" in src or "youtube" in src:
            ep.source_confidence = SourceConfidence.NAMED_REFERENCE
        elif not src:
            # No source at all → reasoned
            ep.source_confidence = SourceConfidence.REASONED
        # else: non-empty source that doesn't match any known pattern →
        #       leave whatever the fetcher already set (may be REASONED, etc.)
    return episodes


# ---------------------------------------------------------------------------
# narrative_sort
# ---------------------------------------------------------------------------


def narrative_sort(
    episodes: list[HistoricalEpisode],
) -> list[HistoricalEpisode]:
    """
    Sort *episodes* into narrative order using Kahn's topological sort over
    ``requires`` edges.  ``published_date`` is used only as a tiebreaker
    within the same topological level (earlier dates first; ``None`` last).

    Cycles are handled gracefully: any episode involved in a cycle is
    appended after the acyclic portion, sorted by ``published_date`` only.
    """
    ep_by_id: dict[str, HistoricalEpisode] = {ep.id: ep for ep in episodes}
    ids = list(ep_by_id.keys())

    # Build adjacency list and in-degree map restricted to known IDs
    in_degree: dict[str, int] = {eid: 0 for eid in ids}
    dependents: dict[str, list[str]] = defaultdict(list)  # id → list of ids that require it

    for ep in episodes:
        for req in ep.requires:
            if req in ep_by_id:
                in_degree[ep.id] += 1
                dependents[req].append(ep.id)

    def _date_key(eid: str):
        d = ep_by_id[eid].published_date
        # None dates sort last
        return (1, date.min) if d is None else (0, d)

    # Seed queue with zero-in-degree nodes, ordered by published_date as tiebreaker
    queue: deque[str] = deque(
        sorted([eid for eid in ids if in_degree[eid] == 0], key=_date_key)
    )

    sorted_ids: list[str] = []
    while queue:
        # Take the next node; tiebreaking is already baked in via sorted seeding
        # and re-insertion below.
        current = queue.popleft()
        sorted_ids.append(current)

        # Collect newly freed nodes and insert them sorted by date
        newly_free = []
        for dep in dependents[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                newly_free.append(dep)

        if newly_free:
            # Sort new free nodes by date and merge into the front of the queue
            newly_free.sort(key=_date_key)
            # Rebuild queue: prepend newly_free, respecting existing ordering
            new_queue: deque[str] = deque(newly_free)
            # Re-sort the combined front for correct tiebreaking
            combined = sorted(list(new_queue) + list(queue), key=_date_key)
            queue = deque(combined)

    # Any remaining nodes with in_degree > 0 are part of a cycle
    cyclic_ids = [eid for eid in ids if eid not in set(sorted_ids)]
    if cyclic_ids:
        logger.warning(
            "narrative_sort: cycle detected among episode IDs %s; "
            "falling back to date-only sort for these nodes",
            cyclic_ids,
        )
        cyclic_ids.sort(key=_date_key)
        sorted_ids.extend(cyclic_ids)

    return [ep_by_id[eid] for eid in sorted_ids]


# ---------------------------------------------------------------------------
# fetch_with_retry
# ---------------------------------------------------------------------------


async def fetch_with_retry(fetch_fn: Callable, max_attempts: int = 3):
    """
    Call ``await fetch_fn()`` up to *max_attempts* times, retrying only on
    :exc:`TransientFetchError`.

    Backoff schedule (0-indexed attempt *k*): ``asyncio.sleep(2 ** k)``
    — i.e., 1 s before attempt 1, 2 s before attempt 2.
    No third-party retry libraries (tenacity etc.) are used.
    """
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
    Orchestrates topic decomposition, source fetching, Track A ingestion,
    and self-check recall to produce a list of HistoricalEpisode objects.
    """

    def __init__(self) -> None:
        self.gateway = MemoryGateway(role=AgentRole.INGESTION)

    async def run(
        self,
        topic: str,
        video_ids: list[str] | None = None,
    ) -> list[HistoricalEpisode]:
        """
        Full pipeline:
        1. decompose_topic → subtopics
        2. For each subtopic: fetch_wikipedia + fetch_arxiv
        3. Optionally fetch_youtube
        4. tag_source_confidence(all_episodes)
        5. narrative_sort(all_episodes) → sorted_episodes
        6. Retry loop (max 3 attempts):
           a. gateway.add_data_points(sorted_episodes, temporal_cognify=True)
           b. cognee.consolidate_entity_descriptions_pipeline()
           c. self_check_recall(topic) → if True, break; else continue
        7. If all 3 attempts fail: reasoned_fallback(subtopics)
        8. Return final episodes list
        """
        import cognee  # noqa: PLC0415 — imported here; config already applied above

        logger.info("IngestionAgent.run: topic=%r", topic)
        subtopics = await decompose_topic(topic)
        logger.info("Decomposed %r into %d subtopics: %s", topic, len(subtopics), subtopics)

        all_episodes: list[HistoricalEpisode] = []
        for subtopic in subtopics:
            wiki_episodes = await fetch_wikipedia(subtopic)
            all_episodes.extend(wiki_episodes)
            logger.info(
                "Wikipedia returned %d episode(s) for subtopic %r",
                len(wiki_episodes),
                subtopic,
            )

            arxiv_episodes = await fetch_arxiv(subtopic)
            all_episodes.extend(arxiv_episodes)
            logger.info(
                "arXiv returned %d episode(s) for subtopic %r",
                len(arxiv_episodes),
                subtopic,
            )

        if video_ids:
            youtube_episodes = await fetch_youtube(video_ids)
            all_episodes.extend(youtube_episodes)
            logger.info(
                "YouTube returned %d episode(s) for %d video(s)",
                len(youtube_episodes),
                len(video_ids),
            )

        # Tag and sort
        tag_source_confidence(all_episodes)
        sorted_episodes = narrative_sort(all_episodes)

        # Retry loop — up to 3 total attempts to write and verify recall
        _MAX_ATTEMPTS = 3
        recalled = False
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            logger.info(
                "IngestionAgent.run: write attempt %d/%d for topic %r",
                attempt, _MAX_ATTEMPTS, topic,
            )
            await self.gateway.add_data_points(sorted_episodes, temporal_cognify=True)

            try:
                await cognee.consolidate_entity_descriptions_pipeline()
            except AttributeError:
                logger.warning(
                    "cognee.consolidate_entity_descriptions_pipeline() is not available "
                    "in the installed version; skipping."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "consolidate_entity_descriptions_pipeline raised an unexpected error: %s",
                    exc,
                )

            recalled = await self.self_check_recall(topic)
            if recalled:
                logger.info(
                    "IngestionAgent.run: recall verified on attempt %d for topic %r",
                    attempt, topic,
                )
                break
            else:
                logger.warning(
                    "IngestionAgent.run: recall check failed on attempt %d for topic %r",
                    attempt, topic,
                )

        if not recalled:
            logger.warning(
                "IngestionAgent.run: all %d recall attempts failed for topic %r; "
                "running reasoned_fallback",
                _MAX_ATTEMPTS, topic,
            )
            fallback_episodes = await self.reasoned_fallback(subtopics)
            sorted_episodes = sorted_episodes + fallback_episodes

        return sorted_episodes

    # ------------------------------------------------------------------
    # self_check_recall
    # ------------------------------------------------------------------

    async def self_check_recall(self, topic: str) -> bool:
        """Call cognee.recall() to verify ingested episodes are surfaced."""
        try:
            import cognee  # noqa: PLC0415
            results = await cognee.recall(topic)
            return bool(results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("self_check_recall failed for %r: %s", topic, exc)
            return False

    # ------------------------------------------------------------------
    # reasoned_fallback
    # ------------------------------------------------------------------

    async def reasoned_fallback(
        self, subtopics: list[str]
    ) -> list[HistoricalEpisode]:
        """
        Generate ``reasoned``-tier HistoricalEpisode objects for subtopics
        that have no existing reasoned coverage in cognee.

        For each subtopic:
        1. Pre-check: if cognee.recall(subtopic) returns any results whose
           source_confidence is REASONED, skip that subtopic (avoid duplication).
        2. Otherwise, use the LLM to generate 1-3 reasoned episodes.
        3. Tag all generated episodes with source_confidence=SourceConfidence.REASONED.
        4. Write the batch to Track A via gateway.add_data_points.
        5. Return the full list of generated episodes.
        """
        import cognee  # noqa: PLC0415

        generated: list[HistoricalEpisode] = []

        for subtopic in subtopics:
            # Pre-check: skip if reasoned episodes already exist
            try:
                existing = await cognee.recall(subtopic)
                has_reasoned = any(
                    getattr(item, "source_confidence", None) == SourceConfidence.REASONED
                    or getattr(item, "source_confidence", None) == "reasoned"
                    for item in (existing or [])
                )
                if has_reasoned:
                    logger.info(
                        "reasoned_fallback: reasoned episodes already exist for %r; skipping",
                        subtopic,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reasoned_fallback: pre-check recall failed for %r: %s; proceeding anyway",
                    subtopic, exc,
                )

            # Generate episodes via LLM
            system_prompt = (
                "You are a first-principles reasoning assistant. "
                "Given a technical or scientific subtopic, reason from first principles "
                "to construct plausible historical problem-solving episodes. "
                "Respond with ONLY a JSON array (no markdown, no extra text) of 1–3 objects, "
                "each with these fields:\n"
                '  "concept": str\n'
                '  "problem_posed": str\n'
                '  "attempted_solution": str\n'
                '  "outcome": "success" | "failure" | "partial"\n'
                '  "why": str\n'
                '  "published_date": "YYYY-MM-DD" or null\n'
                "These are reasoned reconstructions, not sourced facts."
            )
            user_prompt = (
                f"Subtopic: {subtopic}\n\n"
                "Generate 1–3 first-principles problem-solving episodes as a JSON array."
            )

            try:
                response = await litellm.acompletion(
                    model=_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
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
                for i, ep in enumerate(raw_episodes):
                    episode = HistoricalEpisode(
                        id=f"reasoned_{slug}_{uuid.uuid4().hex[:8]}_{i}",
                        concept=ep.get("concept", subtopic),
                        problem_posed=ep.get("problem_posed", ""),
                        attempted_solution=ep.get("attempted_solution", ""),
                        outcome=_parse_outcome(ep.get("outcome", "partial")),
                        why=ep.get("why", ""),
                        source_confidence=SourceConfidence.REASONED,
                        source=None,
                        published_date=_parse_date(ep.get("published_date")),
                    )
                    generated.append(episode)
                    logger.info(
                        "reasoned_fallback: generated episode %r for subtopic %r",
                        episode.id, subtopic,
                    )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reasoned_fallback: LLM generation failed for subtopic %r: %s",
                    subtopic, exc,
                )
                # Minimal fallback episode so the pipeline never returns empty-handed
                generated.append(
                    HistoricalEpisode(
                        id=f"reasoned_{subtopic.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
                        concept=subtopic,
                        problem_posed=f"Understanding {subtopic} from first principles",
                        attempted_solution=(
                            f"Reasoned reconstruction of the foundational challenges in {subtopic}."
                        ),
                        outcome=Outcome.PARTIAL,
                        why="Generated via reasoned fallback; LLM was unavailable.",
                        source_confidence=SourceConfidence.REASONED,
                        source=None,
                    )
                )

        if generated:
            try:
                await self.gateway.add_data_points(generated, temporal_cognify=True)
                logger.info(
                    "reasoned_fallback: wrote %d reasoned episode(s) to Track A",
                    len(generated),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "reasoned_fallback: failed to write episodes to Track A: %s", exc
                )

        return generated


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

ingestion_agent = IngestionAgent()
