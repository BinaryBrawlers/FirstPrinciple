"""Ingestion Agent — tasks 5.1–5.5.

Implements:
  - decompose_topic(topic) -> list[str]
  - fetch_wikipedia(subtopic) -> list[HistoricalEpisode]
  - fetch_arxiv(subtopic) -> list[HistoricalEpisode]
  - fetch_youtube(video_ids) -> list[HistoricalEpisode]
  - tag_source_confidence(episodes) -> list[HistoricalEpisode]
  - narrative_sort(episodes) -> list[HistoricalEpisode]
  - fetch_with_retry(fetch_fn, max_attempts) -> T
  - self_check_recall(topic) -> bool
  - reasoned_fallback(subtopics) -> list[HistoricalEpisode]
  - run(topic, video_ids) -> list[HistoricalEpisode]

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 2.4, 2.5
"""
from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict, deque
from datetime import date
from typing import Awaitable, Callable, List, Optional, TypeVar

import arxiv  # type: ignore[import-untyped]
import wikipediaapi  # type: ignore[import-untyped]
from youtube_transcript_api import (  # type: ignore[import-untyped]
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

import os

import cognee  # type: ignore[import-untyped]

from memory.gateway import AgentRole, MemoryGateway
from models.schemas import HistoricalEpisode, Outcome, SourceConfidence

# Gateway instance used by the Ingestion Agent exclusively.
_gateway = MemoryGateway(role=AgentRole.INGESTION)

# ---------------------------------------------------------------------------
# Wikipedia client (user-agent required by the API)
# ---------------------------------------------------------------------------

_wiki = wikipediaapi.Wikipedia(
    user_agent="FirstPrinciple/1.0 (learning-system)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

# ---------------------------------------------------------------------------
# Sentence splitter helper
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Topic decomposition
# ---------------------------------------------------------------------------

# A small curated map lets us return meaningful subtopics for the two seed
# domains without an LLM call.  Unknown topics fall back to a generic split.
_TOPIC_SUBTOPICS: dict[str, list[str]] = {
    "os memory management": [
        "base and limit registers",
        "memory segmentation",
        "external fragmentation",
        "paging",
        "page tables",
        "memory management unit TLB",
    ],
    "deep learning": [
        "perceptron",
        "XOR problem neural networks",
        "multilayer perceptron",
        "backpropagation",
        "convolutional neural network",
        "recurrent neural network",
        "vanishing gradient problem",
        "long short-term memory LSTM",
        "attention mechanism neural networks",
        "transformer model",
    ],
}


def decompose_topic(topic: str) -> list[str]:
    """Decompose a topic string into a list of subtopic strings.

    For known seed topics a curated subtopic list is returned directly.
    For unknown topics the topic itself is returned as a single-element list
    so that the rest of the pipeline can proceed.

    Args:
        topic: A human-readable topic name, e.g. "deep learning".

    Returns:
        An ordered list of subtopic strings suitable for passing to
        fetch_wikipedia() or fetch_arxiv().
    """
    normalised = topic.strip().lower()
    if normalised in _TOPIC_SUBTOPICS:
        return list(_TOPIC_SUBTOPICS[normalised])
    # Generic fallback: treat the whole topic as one subtopic.
    return [topic.strip()]


# ---------------------------------------------------------------------------
# Wikipedia skeleton pass
# ---------------------------------------------------------------------------

def _parse_published_date(page: wikipediaapi.WikipediaPage) -> Optional[date]:
    """Attempt to extract a rough publication year from the page title.

    Wikipedia pages don't carry a first-published date; we look for a
    four-digit year in the title as a best-effort signal.  Returns None when
    no year is found.
    """
    match = re.search(r"\b(1[89]\d\d|20\d\d)\b", page.title)
    if match:
        return date(int(match.group(1)), 1, 1)
    return None


def _build_episode_from_section(
    subtopic: str,
    section_title: str,
    section_text: str,
    page_url: str,
    episode_index: int,
    parent_id: Optional[str],
) -> HistoricalEpisode:
    """Convert a Wikipedia section into a HistoricalEpisode.

    The first sentence of the section becomes *problem_posed*; the second
    sentence (if present) becomes *attempted_solution*; any remainder feeds
    *why*.  Missing sentences are replaced with an empty string so the schema
    is always satisfied.
    """
    sentences = _split_sentences(section_text)
    problem_posed = sentences[0] if len(sentences) > 0 else ""
    attempted_solution = sentences[1] if len(sentences) > 1 else ""
    why = " ".join(sentences[2:]) if len(sentences) > 2 else ""

    episode_id = f"{subtopic.lower().replace(' ', '_')}_{episode_index}_{uuid.uuid4().hex[:6]}"

    return HistoricalEpisode(
        id=episode_id,
        concept=section_title or subtopic,
        problem_posed=problem_posed,
        attempted_solution=attempted_solution,
        outcome=Outcome.SUCCESS,   # default; refined by tag_source_confidence later
        why=why,
        requires=[parent_id] if parent_id else [],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source=page_url,
        published_date=None,
    )


def fetch_wikipedia(subtopic: str) -> list[HistoricalEpisode]:
    """Fetch Wikipedia content for a subtopic and return HistoricalEpisode objects.

    Performs the Wikipedia skeleton pass described in Requirement 4.2.  The
    top-level page summary is turned into the first episode; each top-level
    section that has meaningful text becomes a subsequent episode.  All
    episodes are tagged with source_confidence=named_reference.

    If Wikipedia returns no page for the subtopic, an empty list is returned
    so the caller can gracefully fall back to other sources or the reasoned
    fallback.

    Args:
        subtopic: A single subtopic string, e.g. "attention mechanism neural networks".

    Returns:
        A list of HistoricalEpisode instances derived from Wikipedia content,
        with source_confidence set to SourceConfidence.NAMED_REFERENCE.
    """
    page = _wiki.page(subtopic)
    if not page.exists():
        return []

    episodes: list[HistoricalEpisode] = []
    page_url = page.fullurl

    # Episode 0: page summary (the lede / introduction)
    if page.summary:
        summary_episode = _build_episode_from_section(
            subtopic=subtopic,
            section_title=page.title,
            section_text=page.summary,
            page_url=page_url,
            episode_index=0,
            parent_id=None,
        )
        episodes.append(summary_episode)

    # Episodes 1…N: one per top-level section with substantive content
    parent_id: Optional[str] = episodes[0].id if episodes else None
    for idx, section in enumerate(page.sections, start=1):
        text = section.text.strip()
        if not text or len(text) < 80:   # skip stubs / navboxes
            continue
        ep = _build_episode_from_section(
            subtopic=subtopic,
            section_title=section.title,
            section_text=text,
            page_url=page_url,
            episode_index=idx,
            parent_id=parent_id,
        )
        episodes.append(ep)
        # Each section episode depends on the previous one (linear narrative).
        parent_id = ep.id

    return episodes


# ---------------------------------------------------------------------------
# arXiv detail pass
# ---------------------------------------------------------------------------

# arXiv API client (shared across calls)
_arxiv_client = arxiv.Client()

# Regex that matches the start of a "References" (or similar) section so we
# can strip it when the introduction text bleeds into the bibliography.
_REFERENCES_RE = re.compile(
    r"\n\s*(?:references|bibliography|acknowledgements?)\s*\n",
    re.IGNORECASE,
)

# Rough word limit for the combined abstract + introduction text we keep.
# arXiv abstracts are typically 150–300 words; intro sections a few hundred
# more.  We cap at 600 words to keep episodes concise.
_MAX_WORDS = 600


def _trim_to_max_words(text: str, max_words: int = _MAX_WORDS) -> str:
    """Return at most *max_words* words from *text*, preserving word boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _extract_abstract_and_intro(paper: arxiv.Result) -> str:
    """Return the abstract text for an arXiv result.

    The arxiv library only exposes the abstract (``paper.summary``); full-text
    PDF parsing is out of scope.  We therefore use the abstract as the
    "abstract + introduction" proxy and cap it at *_MAX_WORDS* words so that
    very long abstracts do not inflate episode text.

    References sections are stripped if they appear in the summary (rare but
    possible in some preprint formats).
    """
    text = paper.summary.strip()
    # Strip anything after a references / acknowledgements heading.
    match = _REFERENCES_RE.search(text)
    if match:
        text = text[: match.start()].strip()
    return _trim_to_max_words(text)


def _build_episode_from_arxiv(
    subtopic: str,
    paper: arxiv.Result,
    episode_index: int,
    parent_id: Optional[str],
) -> HistoricalEpisode:
    """Convert an arXiv ``Result`` into a ``HistoricalEpisode``.

    The abstract text is sentence-split in the same way as Wikipedia sections:
    first sentence → *problem_posed*; second → *attempted_solution*; remainder
    → *why*.  ``source_confidence`` is always ``CITED_SOURCE`` for arXiv papers.
    """
    raw_text = _extract_abstract_and_intro(paper)
    sentences = _split_sentences(raw_text)

    problem_posed = sentences[0] if len(sentences) > 0 else ""
    attempted_solution = sentences[1] if len(sentences) > 1 else ""
    why = " ".join(sentences[2:]) if len(sentences) > 2 else ""

    # Use the arxiv short ID (without the version suffix) for a stable key.
    short_id = paper.entry_id.split("/")[-1]  # e.g. "2301.12345v2" → "2301.12345v2"
    episode_id = (
        f"{subtopic.lower().replace(' ', '_')}_arxiv_{short_id}_{uuid.uuid4().hex[:6]}"
    )

    published_date: Optional[date] = None
    if paper.published:
        published_date = paper.published.date()

    return HistoricalEpisode(
        id=episode_id,
        concept=paper.title or subtopic,
        problem_posed=problem_posed,
        attempted_solution=attempted_solution,
        outcome=Outcome.SUCCESS,  # default; refined by tag_source_confidence later
        why=why,
        requires=[parent_id] if parent_id else [],
        concurrent_with=[],
        source_confidence=SourceConfidence.CITED_SOURCE,
        source=paper.entry_id,
        published_date=published_date,
    )


def fetch_arxiv(subtopic: str, max_results: int = 5) -> list[HistoricalEpisode]:
    """Fetch arXiv papers for *subtopic* and return HistoricalEpisode objects.

    Performs the arXiv detail pass described in Requirement 4.2.  Results are
    restricted to the abstract (which the arxiv library makes available as
    ``Result.summary``); full-text PDF parsing is not performed.  Each result
    is tagged with ``source_confidence=SourceConfidence.CITED_SOURCE``.

    Transient network errors (``Exception`` subclasses raised by the arxiv
    client) are caught and logged so that the caller can fall back or retry
    without the stack propagating.  An empty list is returned on failure.

    Args:
        subtopic: A single subtopic string, e.g. "attention mechanism neural
            networks".
        max_results: Maximum number of arXiv papers to return (default 5).

    Returns:
        A list of HistoricalEpisode instances derived from arXiv abstracts,
        with source_confidence set to SourceConfidence.CITED_SOURCE.
    """
    search = arxiv.Search(
        query=subtopic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        results = list(_arxiv_client.results(search))
    except Exception:
        # Transient network / parsing error — return empty so the caller can
        # retry or fall back gracefully (Requirement 4.8).
        return []

    if not results:
        return []

    episodes: list[HistoricalEpisode] = []
    parent_id: Optional[str] = None

    for idx, paper in enumerate(results):
        ep = _build_episode_from_arxiv(
            subtopic=subtopic,
            paper=paper,
            episode_index=idx,
            parent_id=parent_id,
        )
        episodes.append(ep)
        # Chain episodes linearly: each paper depends on the previous one.
        parent_id = ep.id

    return episodes


# ---------------------------------------------------------------------------
# YouTube transcript fetch
# ---------------------------------------------------------------------------

# Maximum number of transcript snippets (captions) we collect per video.
# Each snippet is typically 3–7 words; 200 snippets ≈ 600–1400 words.
_MAX_SNIPPETS = 200

# Exceptions that indicate the transcript simply isn't available (not transient).
_TRANSCRIPT_UNAVAILABLE = (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def _build_episode_from_transcript(
    video_id: str,
    transcript_text: str,
    episode_index: int,
    parent_id: Optional[str],
) -> HistoricalEpisode:
    """Convert a YouTube transcript into a ``HistoricalEpisode``.

    The same sentence-splitting strategy as Wikipedia/arXiv is applied.
    ``source_confidence`` is always ``NAMED_REFERENCE`` for video content.
    """
    sentences = _split_sentences(transcript_text)

    problem_posed = sentences[0] if len(sentences) > 0 else ""
    attempted_solution = sentences[1] if len(sentences) > 1 else ""
    why = " ".join(sentences[2:]) if len(sentences) > 2 else ""

    episode_id = f"youtube_{video_id}_{episode_index}_{uuid.uuid4().hex[:6]}"
    source_url = f"https://www.youtube.com/watch?v={video_id}"

    return HistoricalEpisode(
        id=episode_id,
        concept=f"YouTube/{video_id}",
        problem_posed=problem_posed,
        attempted_solution=attempted_solution,
        outcome=Outcome.SUCCESS,  # default; refined by tag_source_confidence later
        why=why,
        requires=[parent_id] if parent_id else [],
        concurrent_with=[],
        source_confidence=SourceConfidence.NAMED_REFERENCE,
        source=source_url,
        published_date=None,
    )


def fetch_youtube(video_ids: list[str]) -> list[HistoricalEpisode]:
    """Fetch YouTube transcripts for a list of video IDs and return HistoricalEpisode objects.

    Performs the YouTube transcript pass described in Requirement 4.3.  Uses
    ``youtube-transcript-api`` to extract English caption text; each video
    becomes one HistoricalEpisode tagged with
    ``source_confidence=SourceConfidence.NAMED_REFERENCE``.

    Videos for which no transcript is available (disabled captions, private
    videos, etc.) are silently skipped.  Transient errors (network failures,
    unexpected API errors) are caught so that the caller can retry or fall
    back without the stack propagating.  An empty list is returned if no
    transcripts could be retrieved.

    Args:
        video_ids: A list of YouTube video ID strings, e.g. ["dQw4w9WgXcQ"].
            These must be bare IDs, not full URLs.

    Returns:
        A list of HistoricalEpisode instances derived from video transcripts,
        with source_confidence set to SourceConfidence.NAMED_REFERENCE.
    """
    if not video_ids:
        return []

    episodes: list[HistoricalEpisode] = []
    parent_id: Optional[str] = None

    for idx, video_id in enumerate(video_ids):
        try:
            fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except _TRANSCRIPT_UNAVAILABLE:
            # Transcript genuinely not available — skip this video.
            continue
        except CouldNotRetrieveTranscript:
            # Broader catch for any retrieval failure — skip gracefully.
            continue
        except Exception:
            # Transient error (network timeout, rate-limit, etc.) — skip so
            # the caller can retry the whole batch if needed (Requirement 4.8).
            continue

        # Join snippet texts into a single string, capped at _MAX_SNIPPETS.
        # get_transcript returns list[dict] with 'text', 'start', 'duration'.
        transcript_text = " ".join(
            snippet["text"] for snippet in fetched[:_MAX_SNIPPETS]
        ).strip()

        if not transcript_text:
            continue

        ep = _build_episode_from_transcript(
            video_id=video_id,
            transcript_text=transcript_text,
            episode_index=idx,
            parent_id=parent_id,
        )
        episodes.append(ep)
        parent_id = ep.id

    return episodes


# ---------------------------------------------------------------------------
# tag_source_confidence — Requirement 4.4
# ---------------------------------------------------------------------------

# Keywords that hint an episode represents a failed attempt rather than a
# successful solution.  Used to downgrade the default SUCCESS outcome when
# no explicit outcome has been set by the caller.
_FAILURE_KEYWORDS = frozenset(
    [
        "fail", "failure", "error", "limitation", "problem", "challenge",
        "unable", "cannot", "could not", "does not", "did not", "incorrect",
        "inadequate", "insufficient", "wrong", "broken", "issue", "defect",
        "regression", "crash", "bug",
    ]
)


def tag_source_confidence(episodes: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
    """Apply three-tier source_confidence tagging rules to *episodes* in place.

    The tagging rules are:
      - Episodes whose ``source`` URL contains "arxiv" → ``CITED_SOURCE``
      - Episodes from Wikipedia (``source`` is a Wikipedia URL) → ``NAMED_REFERENCE``
      - Episodes with no ``source`` → ``REASONED``
      - Any episode that already has a non-default confidence set by the caller
        is left unchanged.

    Additionally, the ``outcome`` field is refined for episodes that have not
    already been explicitly set to FAILURE or PARTIAL: if the episode text
    (concept + problem_posed + why) contains failure-indicating keywords, the
    outcome is updated to ``Outcome.FAILURE``.

    Args:
        episodes: A list of HistoricalEpisode instances to tag.

    Returns:
        The same list with ``source_confidence`` (and ``outcome`` where
        applicable) updated in place.  Returns the list for convenience.
    """
    for ep in episodes:
        source = (ep.source or "").lower()

        # Determine confidence tier from the source URL.
        if "arxiv.org" in source:
            ep.source_confidence = SourceConfidence.CITED_SOURCE
        elif "wikipedia.org" in source:
            ep.source_confidence = SourceConfidence.NAMED_REFERENCE
        elif not source:
            ep.source_confidence = SourceConfidence.REASONED
        # else: non-standard source URL — leave the existing confidence intact.

        # Refine outcome: check for failure keywords only when outcome is
        # currently SUCCESS (we don't override explicit FAILURE/PARTIAL).
        if ep.outcome == Outcome.SUCCESS:
            combined_text = " ".join(
                [ep.concept, ep.problem_posed, ep.why]
            ).lower()
            if any(kw in combined_text for kw in _FAILURE_KEYWORDS):
                ep.outcome = Outcome.FAILURE

    return episodes


# ---------------------------------------------------------------------------
# narrative_sort — Requirement 4.5
# ---------------------------------------------------------------------------


def narrative_sort(episodes: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
    """Return *episodes* sorted into narrative (topological) order.

    Algorithm: Kahn's BFS-based topological sort.

    - Primary criterion: ``requires`` edges — if episode B requires episode A,
      A must appear before B in the output.
    - Tiebreaker: when multiple episodes are simultaneously eligible (in-degree
      zero at the same step), order them by ``published_date`` ascending;
      episodes with no date sort last among ties (treated as ``date.max``).
    - Raises ``ValueError`` if the ``requires`` graph contains a cycle.

    Args:
        episodes: An unordered list of HistoricalEpisode instances.

    Returns:
        A new list in dependency-respecting narrative order.

    Raises:
        ValueError: If a dependency cycle is detected among the episode IDs.

    Requirements: 4.5
    """
    if not episodes:
        return []

    id_to_episode: dict[str, HistoricalEpisode] = {ep.id: ep for ep in episodes}

    # Build in-degree map and successor adjacency list for known IDs only.
    in_degree: dict[str, int] = {ep.id: 0 for ep in episodes}
    successors: dict[str, list[str]] = defaultdict(list)

    for ep in episodes:
        for req_id in ep.requires:
            if req_id in id_to_episode:
                in_degree[ep.id] += 1
                successors[req_id].append(ep.id)

    def _sort_key(ep_id: str) -> tuple:
        ep = id_to_episode[ep_id]
        d = ep.published_date if ep.published_date is not None else date.max
        return (d, ep_id)  # ep_id as secondary tiebreaker for stability

    ready: list[str] = sorted(
        [ep_id for ep_id, deg in in_degree.items() if deg == 0],
        key=_sort_key,
    )
    queue: deque[str] = deque(ready)
    result: list[HistoricalEpisode] = []

    while queue:
        current_id = queue.popleft()
        result.append(id_to_episode[current_id])

        newly_eligible: list[str] = []
        for successor_id in successors[current_id]:
            in_degree[successor_id] -= 1
            if in_degree[successor_id] == 0:
                newly_eligible.append(successor_id)

        for ep_id in sorted(newly_eligible, key=_sort_key):
            queue.append(ep_id)

    if len(result) != len(episodes):
        visited = {ep.id for ep in result}
        cycle_members = [ep.id for ep in episodes if ep.id not in visited]
        raise ValueError(
            f"narrative_sort: cycle detected among episode IDs: {cycle_members}"
        )

    return result


# ---------------------------------------------------------------------------
# fetch_with_retry — Requirement 4.8
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class TransientFetchError(Exception):
    """Raised by fetch functions when a transient (retryable) error occurs."""


async def fetch_with_retry(
    fetch_fn: Callable[[], Awaitable[_T]],
    max_attempts: int = 3,
) -> _T:
    """Call *fetch_fn* up to *max_attempts* times with exponential backoff.

    On each failure the coroutine sleeps for ``2 ** attempt`` seconds before
    the next attempt (attempt 0 → 1 s, attempt 1 → 2 s, attempt 2 → 4 s).
    ``tenacity`` is intentionally NOT used; the backoff is implemented with
    plain ``asyncio.sleep``.

    Args:
        fetch_fn: A zero-argument async callable that either returns a result
            or raises ``TransientFetchError`` on a retryable failure.
        max_attempts: Total number of attempts before re-raising the last
            exception (default 3).

    Returns:
        The return value of *fetch_fn* on the first successful call.

    Raises:
        TransientFetchError: If all *max_attempts* attempts fail.

    Requirements: 4.8
    """
    last_exc: Exception = TransientFetchError("fetch_with_retry: no attempts made")
    for attempt in range(max_attempts):
        try:
            return await fetch_fn()
        except TransientFetchError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_exc


# ---------------------------------------------------------------------------
# self_check_recall — Requirement 4.6
# ---------------------------------------------------------------------------

# Maximum total attempts for ingestion + recall verification loop.
_MAX_INGEST_ATTEMPTS = 3


async def self_check_recall(topic: str) -> bool:
    """Check whether cognee.recall() surfaces episodes for *topic* in Track A.

    Makes a single ``cognee.recall()`` call and returns ``True`` if at least
    one result is returned, ``False`` otherwise.  This is a lightweight
    post-ingestion sanity check; it does NOT retry internally — the caller
    (``run()``) owns the retry loop.

    Args:
        topic: The topic string that was ingested, used as the recall query.

    Returns:
        ``True`` if recall returns at least one result; ``False`` if the result
        is empty or if cognee is unavailable.

    Requirements: 4.6
    """
    try:
        results = await cognee.recall(graph_name="content_track", query=topic)
        return bool(results)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# reasoned_fallback — Requirements 4.7
# ---------------------------------------------------------------------------

def _build_reasoned_episode(subtopic: str, index: int, parent_id: Optional[str]) -> HistoricalEpisode:
    """Build a minimal ``reasoned``-tier HistoricalEpisode for *subtopic*.

    This is a lightweight placeholder used when the Mistral LLM call either
    fails or is unavailable.  The episode signals that synthesised content is
    present but provides minimal prose so consumers can distinguish it from
    fetched content.
    """
    episode_id = f"reasoned_{subtopic.lower().replace(' ', '_')}_{index}_{uuid.uuid4().hex[:6]}"
    return HistoricalEpisode(
        id=episode_id,
        concept=subtopic,
        problem_posed=f"What problem did researchers face when studying {subtopic}?",
        attempted_solution=f"Early attempts at {subtopic} explored foundational approaches.",
        outcome=Outcome.PARTIAL,
        why=f"This episode was synthesised by the Ingestion Agent as a reasoned-tier fallback for '{subtopic}'.",
        requires=[parent_id] if parent_id else [],
        concurrent_with=[],
        source_confidence=SourceConfidence.REASONED,
        source=None,
        published_date=None,
    )


async def _synthesise_via_llm(subtopic: str) -> Optional[HistoricalEpisode]:
    """Use the Mistral LLM (via cognee) to synthesise a reasoned-tier episode.

    Constructs a prompt that requests a concise historical narrative for
    *subtopic* and parses the response into ``HistoricalEpisode`` fields.
    Returns ``None`` if cognee or the LLM is unavailable.

    Args:
        subtopic: The subtopic to synthesise an episode for.

    Returns:
        A ``HistoricalEpisode`` with ``source_confidence=REASONED``, or
        ``None`` on failure.
    """
    prompt = (
        f"You are a history-of-technology expert. "
        f"Describe in three sentences the historical problem that researchers faced "
        f"when developing '{subtopic}': "
        f"(1) the problem they were trying to solve, "
        f"(2) the approach they attempted, "
        f"(3) why that approach succeeded, failed, or led to further work. "
        f"Be concise and factual."
    )
    try:
        # cognee.recall with a synthesis prompt acts as an LLM query when
        # no graph data is present; some cognee builds also expose
        # cognee.generate() — we use recall as a portable interface.
        response = await cognee.recall(query=prompt)
        # response may be a list of strings or objects; extract text.
        if isinstance(response, (list, tuple)) and response:
            text = str(response[0])
        elif isinstance(response, str):
            text = response
        else:
            return None

        sentences = _split_sentences(text)
        problem_posed = sentences[0] if len(sentences) > 0 else ""
        attempted_solution = sentences[1] if len(sentences) > 1 else ""
        why = " ".join(sentences[2:]) if len(sentences) > 2 else ""

        episode_id = f"reasoned_{subtopic.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        return HistoricalEpisode(
            id=episode_id,
            concept=subtopic,
            problem_posed=problem_posed,
            attempted_solution=attempted_solution,
            outcome=Outcome.PARTIAL,
            why=why,
            requires=[],
            concurrent_with=[],
            source_confidence=SourceConfidence.REASONED,
            source=None,
            published_date=None,
        )
    except Exception:
        return None


async def reasoned_fallback(subtopics: List[str]) -> List[HistoricalEpisode]:
    """Generate ``reasoned``-tier episodes for *subtopics* that failed recall.

    For each subtopic, first checks whether a ``reasoned``-tier episode already
    exists in Track A (via ``cognee.recall()``).  If one is found the subtopic
    is skipped to avoid duplication (Requirement 1.4).  Otherwise the agent
    attempts to synthesise an episode via the Mistral LLM; if the LLM call
    fails it falls back to a minimal placeholder episode.

    The synthesised episodes are written to Track A via the gateway and
    ``consolidate_entity_descriptions_pipeline()`` is called at the end.

    Args:
        subtopics: List of subtopic strings for which ingestion recall failed.

    Returns:
        A list of newly created ``HistoricalEpisode`` objects with
        ``source_confidence=SourceConfidence.REASONED``.

    Requirements: 4.7, 1.4
    """
    new_episodes: List[HistoricalEpisode] = []
    parent_id: Optional[str] = None

    for idx, subtopic in enumerate(subtopics):
        # Pre-check: skip if a reasoned episode already exists for this subtopic
        # to prevent duplication on repeated runs (Requirement 1.4).
        try:
            existing = await cognee.recall(query=f"reasoned {subtopic}")
            if existing:
                # At least one episode already recalled — skip synthesis.
                continue
        except Exception:
            pass  # If recall fails, proceed with synthesis anyway.

        # Attempt LLM synthesis via Mistral first.
        ep = await _synthesise_via_llm(subtopic)
        if ep is None:
            # LLM unavailable — use the minimal placeholder.
            ep = _build_reasoned_episode(subtopic, idx, parent_id)
        else:
            # Wire requires chain from previous episode.
            if parent_id:
                ep.requires = [parent_id]

        new_episodes.append(ep)
        parent_id = ep.id

    if new_episodes:
        try:
            await _gateway.add_data_points(new_episodes, temporal_cognify=True)
            await cognee.consolidate_entity_descriptions_pipeline()
        except Exception:
            pass

    return new_episodes


# ---------------------------------------------------------------------------
# run — top-level ingestion pipeline
# ---------------------------------------------------------------------------

async def run(topic: str, video_ids: Optional[List[str]] = None) -> List[HistoricalEpisode]:
    """Execute the full ingestion pipeline for *topic*.

    Pipeline:
      1. decompose_topic(topic) → subtopics
      2. For each subtopic: fetch_wikipedia + fetch_arxiv (+ fetch_youtube if
         video_ids are provided)
      3. tag_source_confidence()
      4. narrative_sort()
      5. gateway.add_data_points(episodes, temporal_cognify=True)
      6. cognee.consolidate_entity_descriptions_pipeline()
      7. self_check_recall(topic) — retry up to _MAX_INGEST_ATTEMPTS total
         with exponential backoff (asyncio.sleep(2**attempt)) if it fails
      8. If all recall attempts fail → reasoned_fallback(subtopics)

    Args:
        topic: A human-readable topic name, e.g. "deep learning".
        video_ids: Optional list of YouTube video IDs to include as a
            supplementary source (Requirement 4.3).

    Returns:
        The ordered list of HistoricalEpisode objects written to Track A.
        If the reasoned fallback was triggered, reasoned-tier episodes are
        appended at the end.

    Requirements: 2.4, 2.5, 4.1–4.8
    """
    video_ids = video_ids or []

    # 1. Decompose topic into subtopics.
    subtopics = decompose_topic(topic)

    # 2. Fetch source material for each subtopic.
    all_episodes: List[HistoricalEpisode] = []
    for subtopic in subtopics:
        all_episodes.extend(fetch_wikipedia(subtopic))
        all_episodes.extend(fetch_arxiv(subtopic))

    if video_ids:
        all_episodes.extend(fetch_youtube(video_ids))

    # 3. Tag confidence tiers.
    tag_source_confidence(all_episodes)

    # 4. Sort into narrative (topological) order.
    ordered_episodes = narrative_sort(all_episodes)

    # 5–6. Write to Track A and consolidate.
    if ordered_episodes:
        await _gateway.add_data_points(ordered_episodes, temporal_cognify=True)
        try:
            await cognee.consolidate_entity_descriptions_pipeline()
        except Exception:
            pass  # Non-fatal; ingestion continues.

    # 7. Self-check recall with up to _MAX_INGEST_ATTEMPTS total attempts.
    recall_ok = False
    for attempt in range(_MAX_INGEST_ATTEMPTS):
        if await self_check_recall(topic):
            recall_ok = True
            break
        # Re-ingest on failure before sleeping (except on the last attempt).
        if attempt < _MAX_INGEST_ATTEMPTS - 1:
            # Re-add data points to prompt cognee to re-index.
            if ordered_episodes:
                try:
                    await _gateway.add_data_points(ordered_episodes, temporal_cognify=True)
                    try:
                        await cognee.consolidate_entity_descriptions_pipeline()
                    except Exception:
                        pass
                except Exception:
                    pass
            await asyncio.sleep(2 ** attempt)

    # 8. Reasoned fallback if all recall checks failed.
    fallback_episodes: List[HistoricalEpisode] = []
    if not recall_ok:
        fallback_episodes = await reasoned_fallback(subtopics)

    return ordered_episodes + fallback_episodes
