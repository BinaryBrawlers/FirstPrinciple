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
from datetime import date
from typing import Optional

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
# IngestionAgent
# ---------------------------------------------------------------------------


class IngestionAgent:
    """
    Orchestrates topic decomposition and source fetching to produce a list of
    HistoricalEpisode objects.

    This is a skeleton implementation for tasks 5.1-5.2; cognee writes
    are added in task 5.5.
    """

    def __init__(self) -> None:
        self.gateway = MemoryGateway(role=AgentRole.INGESTION)

    async def run(
        self,
        topic: str,
        video_ids: list[str] | None = None,
    ) -> list[HistoricalEpisode]:
        """
        Decompose *topic* into subtopics, fetch Wikipedia content for each,
        and return the collected HistoricalEpisode objects.

        cognee writes (Track A) will be added in task 5.5.
        arXiv and YouTube fetching will be added in task 5.2.
        """
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

        return all_episodes


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

ingestion_agent = IngestionAgent()
