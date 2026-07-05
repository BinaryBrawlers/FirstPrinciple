"""Unit tests for ingestion.py — fetch_arxiv and fetch_youtube (task 5.2).

These tests use mocking so no real network calls are made.  They verify:
  - Return type is list[HistoricalEpisode]
  - source_confidence is correctly assigned per function
  - Empty/error cases return empty lists gracefully
  - Episode fields satisfy the HistoricalEpisode schema requirements
  - requires chain is built correctly across multiple results
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make sure the backend package root is on sys.path when running from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, "backend")

from agents.ingestion import fetch_arxiv, fetch_youtube
from models.schemas import HistoricalEpisode, SourceConfidence


# ===========================================================================
# Helpers
# ===========================================================================

def _make_arxiv_result(
    title: str = "Test Paper",
    summary: str = "First sentence. Second sentence. Third sentence.",
    entry_id: str = "https://arxiv.org/abs/2301.00001v1",
    published: datetime | None = None,
) -> MagicMock:
    """Return a minimal mock of an arxiv.Result."""
    result = MagicMock()
    result.title = title
    result.summary = summary
    result.entry_id = entry_id
    result.published = published or datetime(2023, 1, 1, tzinfo=timezone.utc)
    return result


def _make_snippet(text: str) -> dict:
    """Return a transcript snippet dict matching YouTubeTranscriptApi.get_transcript output."""
    return {"text": text, "start": 0.0, "duration": 5.0}


def _make_fetched_transcript(snippets: list[str]) -> list[dict]:
    return [_make_snippet(t) for t in snippets]


# ===========================================================================
# fetch_arxiv tests
# ===========================================================================

class TestFetchArxiv:
    """Tests for fetch_arxiv()."""

    def test_returns_list_of_historical_episodes(self):
        """fetch_arxiv returns a list of HistoricalEpisode objects."""
        mock_result = _make_arxiv_result()
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([mock_result])
            episodes = fetch_arxiv("attention mechanism")

        assert isinstance(episodes, list)
        assert len(episodes) == 1
        assert isinstance(episodes[0], HistoricalEpisode)

    def test_source_confidence_is_cited_source(self):
        """All episodes produced by fetch_arxiv must be tagged CITED_SOURCE."""
        mock_results = [_make_arxiv_result(f"Paper {i}") for i in range(3)]
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter(mock_results)
            episodes = fetch_arxiv("backpropagation", max_results=3)

        assert len(episodes) == 3
        for ep in episodes:
            assert ep.source_confidence == SourceConfidence.CITED_SOURCE

    def test_episode_fields_populated(self):
        """Core HistoricalEpisode fields are populated from arxiv result."""
        mock_result = _make_arxiv_result(
            title="Attention Is All You Need",
            summary="We propose a new architecture. It uses attention only. It is faster.",
            entry_id="https://arxiv.org/abs/1706.03762v5",
        )
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([mock_result])
            episodes = fetch_arxiv("transformer model")

        ep = episodes[0]
        assert ep.concept == "Attention Is All You Need"
        assert ep.problem_posed == "We propose a new architecture."
        assert ep.attempted_solution == "It uses attention only."
        assert "It is faster." in ep.why
        assert ep.source == "https://arxiv.org/abs/1706.03762v5"

    def test_published_date_extracted(self):
        """published_date is set from paper.published."""
        published = datetime(2017, 6, 12, tzinfo=timezone.utc)
        mock_result = _make_arxiv_result(published=published)
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([mock_result])
            episodes = fetch_arxiv("transformer")

        from datetime import date
        assert episodes[0].published_date == date(2017, 6, 12)

    def test_requires_chain_built_across_results(self):
        """Each episode (after the first) should require the preceding one."""
        mock_results = [_make_arxiv_result(f"Paper {i}") for i in range(3)]
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter(mock_results)
            episodes = fetch_arxiv("neural networks", max_results=3)

        assert episodes[0].requires == []
        assert episodes[1].requires == [episodes[0].id]
        assert episodes[2].requires == [episodes[1].id]

    def test_no_results_returns_empty_list(self):
        """fetch_arxiv returns [] when no arXiv results are found."""
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([])
            episodes = fetch_arxiv("nonexistent topic xyz123")

        assert episodes == []

    def test_network_error_returns_empty_list(self):
        """fetch_arxiv returns [] on transient network errors, does not raise."""
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.side_effect = ConnectionError("network failure")
            episodes = fetch_arxiv("backpropagation")

        assert episodes == []

    def test_episode_id_is_string(self):
        """Episode IDs must be non-empty strings."""
        mock_result = _make_arxiv_result()
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([mock_result])
            episodes = fetch_arxiv("paging")

        assert isinstance(episodes[0].id, str)
        assert len(episodes[0].id) > 0

    def test_references_section_stripped_from_summary(self):
        """Text after a 'References' heading in the summary is stripped."""
        summary_with_refs = (
            "First sentence of abstract. Second sentence.\n\nReferences\n\n[1] Smith et al."
        )
        mock_result = _make_arxiv_result(summary=summary_with_refs)
        with patch("agents.ingestion._arxiv_client") as mock_client:
            mock_client.results.return_value = iter([mock_result])
            episodes = fetch_arxiv("test")

        ep = episodes[0]
        assert "[1] Smith et al." not in ep.problem_posed
        assert "[1] Smith et al." not in ep.attempted_solution
        assert "[1] Smith et al." not in ep.why


# ===========================================================================
# fetch_youtube tests
# ===========================================================================

class TestFetchYoutube:
    """Tests for fetch_youtube()."""

    def _patch_api_fetch(self, snippets: list[str]):
        """Context manager that patches YouTubeTranscriptApi.get_transcript to return snippets.

        Each call returns a fresh list of dicts so multiple video IDs all
        produce transcripts without exhausting a shared iterator.
        """
        transcript_dicts = [_make_snippet(t) for t in snippets]
        return patch(
            "agents.ingestion.YouTubeTranscriptApi.get_transcript",
            side_effect=lambda *a, **kw: list(transcript_dicts),
        )

    def test_returns_list_of_historical_episodes(self):
        """fetch_youtube returns a list of HistoricalEpisode objects."""
        snippets = ["Hello world.", "This is a transcript.", "Third line."]
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube(["dQw4w9WgXcQ"])

        assert isinstance(episodes, list)
        assert len(episodes) == 1
        assert isinstance(episodes[0], HistoricalEpisode)

    def test_source_confidence_is_named_reference(self):
        """All episodes produced by fetch_youtube must be tagged NAMED_REFERENCE."""
        snippets = ["Intro.", "Body.", "Conclusion."]
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube(["vid1", "vid2"])

        assert len(episodes) == 2
        for ep in episodes:
            assert ep.source_confidence == SourceConfidence.NAMED_REFERENCE

    def test_episode_source_url_contains_video_id(self):
        """The source URL must embed the YouTube video ID."""
        snippets = ["Some text."]
        video_id = "abc123XYZ"
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube([video_id])

        assert video_id in episodes[0].source
        assert "youtube.com" in episodes[0].source

    def test_episode_fields_populated(self):
        """problem_posed, attempted_solution, and why are populated from transcript."""
        snippets = ["First sentence text.", "Second sentence text.", "Third sentence text."]
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube(["test123"])

        ep = episodes[0]
        assert "First sentence text." in ep.problem_posed
        assert "Second sentence text." in ep.attempted_solution

    def test_requires_chain_across_multiple_videos(self):
        """Each episode (after the first) should require the preceding one."""
        snippets = ["A line."]
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube(["vid1", "vid2", "vid3"])

        assert episodes[0].requires == []
        assert episodes[1].requires == [episodes[0].id]
        assert episodes[2].requires == [episodes[1].id]

    def test_empty_video_ids_returns_empty_list(self):
        """fetch_youtube returns [] when given an empty list."""
        episodes = fetch_youtube([])
        assert episodes == []

    def test_transcripts_disabled_skips_video(self):
        """Videos with disabled transcripts are silently skipped."""
        from youtube_transcript_api import TranscriptsDisabled

        with patch("agents.ingestion.YouTubeTranscriptApi.get_transcript", side_effect=TranscriptsDisabled("vid1")):
            episodes = fetch_youtube(["vid1"])

        assert episodes == []

    def test_no_transcript_found_skips_video(self):
        """Videos with no English transcript are silently skipped."""
        from youtube_transcript_api import NoTranscriptFound

        with patch("agents.ingestion.YouTubeTranscriptApi.get_transcript", side_effect=NoTranscriptFound("vid1", ["en"], {})):
            episodes = fetch_youtube(["vid1"])

        assert episodes == []

    def test_video_unavailable_skips_video(self):
        """Unavailable videos are silently skipped."""
        from youtube_transcript_api import VideoUnavailable

        with patch("agents.ingestion.YouTubeTranscriptApi.get_transcript", side_effect=VideoUnavailable("vid1")):
            episodes = fetch_youtube(["vid1"])

        assert episodes == []

    def test_transient_exception_skips_video(self):
        """Transient errors (e.g. ConnectionError) cause video to be skipped, not raised."""
        with patch("agents.ingestion.YouTubeTranscriptApi.get_transcript", side_effect=ConnectionError("timeout")):
            episodes = fetch_youtube(["vid1"])

        assert episodes == []

    def test_partial_failure_still_returns_successful_videos(self):
        """If one video fails and another succeeds, only the successful one appears."""
        from youtube_transcript_api import TranscriptsDisabled

        good_snippet = [_make_snippet("Hello world.")]

        def side_effect(video_id, **kwargs):
            if video_id == "bad_vid":
                raise TranscriptsDisabled(video_id)
            return good_snippet

        with patch("agents.ingestion.YouTubeTranscriptApi.get_transcript", side_effect=side_effect):
            episodes = fetch_youtube(["bad_vid", "good_vid"])

        assert len(episodes) == 1
        assert "good_vid" in episodes[0].source

    def test_episode_id_is_string(self):
        """Episode IDs must be non-empty strings."""
        snippets = ["Some text."]
        with self._patch_api_fetch(snippets):
            episodes = fetch_youtube(["vidtest"])

        assert isinstance(episodes[0].id, str)
        assert len(episodes[0].id) > 0


# ===========================================================================
# Tests for self_check_recall, reasoned_fallback, and run (task 5.5)
# ===========================================================================

import asyncio
from unittest.mock import AsyncMock, patch, call


def _run(coro):
    """Run a coroutine synchronously for test purposes."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# self_check_recall
# ---------------------------------------------------------------------------

class TestSelfCheckRecall:
    """Tests for self_check_recall()."""

    def test_returns_true_when_recall_returns_results(self):
        """self_check_recall returns True when cognee.recall returns non-empty."""
        from agents.ingestion import self_check_recall

        with patch("agents.ingestion.cognee") as mock_cognee:
            mock_cognee.recall = AsyncMock(return_value=["some episode"])
            result = _run(self_check_recall("deep learning"))

        assert result is True

    def test_returns_false_when_recall_returns_empty(self):
        """self_check_recall returns False when cognee.recall returns empty list."""
        from agents.ingestion import self_check_recall

        with patch("agents.ingestion.cognee") as mock_cognee:
            mock_cognee.recall = AsyncMock(return_value=[])
            result = _run(self_check_recall("unknown topic"))

        assert result is False

    def test_returns_false_when_recall_raises(self):
        """self_check_recall returns False (does not propagate) when recall raises."""
        from agents.ingestion import self_check_recall

        with patch("agents.ingestion.cognee") as mock_cognee:
            mock_cognee.recall = AsyncMock(side_effect=RuntimeError("cognee down"))
            result = _run(self_check_recall("deep learning"))

        assert result is False

    def test_returns_false_when_cognee_is_none(self):
        """self_check_recall returns False when cognee.recall raises (simulating unavailability)."""
        from agents.ingestion import self_check_recall

        with patch("agents.ingestion.cognee") as mock_cognee:
            mock_cognee.recall = AsyncMock(side_effect=Exception("unavailable"))
            result = _run(self_check_recall("any topic"))

        assert result is False


# ---------------------------------------------------------------------------
# reasoned_fallback
# ---------------------------------------------------------------------------

class TestReasonedFallback:
    """Tests for reasoned_fallback()."""

    def test_returns_list_of_historical_episodes(self):
        """reasoned_fallback returns HistoricalEpisode instances."""
        from agents.ingestion import reasoned_fallback

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw:
            mock_cognee.recall = AsyncMock(return_value=[])
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            episodes = _run(reasoned_fallback(["paging", "page tables"]))

        assert isinstance(episodes, list)
        assert len(episodes) == 2
        for ep in episodes:
            assert isinstance(ep, HistoricalEpisode)

    def test_all_episodes_have_reasoned_confidence(self):
        """All episodes from reasoned_fallback have REASONED source_confidence."""
        from agents.ingestion import reasoned_fallback
        from models.schemas import SourceConfidence

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw:
            mock_cognee.recall = AsyncMock(return_value=[])
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            episodes = _run(reasoned_fallback(["attention mechanism"]))

        for ep in episodes:
            assert ep.source_confidence == SourceConfidence.REASONED

    def test_no_duplication_when_recall_returns_existing(self):
        """reasoned_fallback skips subtopics already present in Track A."""
        from agents.ingestion import reasoned_fallback

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw:
            mock_cognee.recall = AsyncMock(return_value=["existing episode"])
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            episodes = _run(reasoned_fallback(["paging", "segmentation"]))

        assert episodes == []
        mock_gw.add_data_points.assert_not_called()

    def test_gateway_add_data_points_called_with_temporal_cognify(self):
        """reasoned_fallback calls gateway.add_data_points with temporal_cognify=True."""
        from agents.ingestion import reasoned_fallback

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw:
            mock_cognee.recall = AsyncMock(return_value=[])
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            _run(reasoned_fallback(["backpropagation"]))

        mock_gw.add_data_points.assert_called_once()
        _, kwargs = mock_gw.add_data_points.call_args
        assert kwargs.get("temporal_cognify") is True

    def test_requires_chain_built_across_episodes(self):
        """Each reasoned episode (after the first) requires the preceding one."""
        from agents.ingestion import reasoned_fallback

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw:
            mock_cognee.recall = AsyncMock(return_value=[])
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            episodes = _run(reasoned_fallback(["a", "b", "c"]))

        assert episodes[0].requires == []
        assert episodes[1].requires == [episodes[0].id]
        assert episodes[2].requires == [episodes[1].id]

    def test_empty_subtopics_returns_empty_list(self):
        """reasoned_fallback returns [] for an empty subtopics list."""
        from agents.ingestion import reasoned_fallback

        with patch("agents.ingestion._gateway") as mock_gw:
            mock_gw.add_data_points = AsyncMock()
            episodes = _run(reasoned_fallback([]))

        assert episodes == []
        mock_gw.add_data_points.assert_not_called()


# ---------------------------------------------------------------------------
# run() — top-level ingestion pipeline
# ---------------------------------------------------------------------------

class TestIngestionRun:
    """Tests for the run() ingestion pipeline."""

    def _make_episode(self, id_suffix: str = "ep1"):
        """Create a minimal HistoricalEpisode for use in mocks."""
        from models.schemas import HistoricalEpisode, Outcome, SourceConfidence
        return HistoricalEpisode(
            id=f"test_{id_suffix}",
            concept="Test Concept",
            problem_posed="Problem.",
            attempted_solution="Solution.",
            outcome=Outcome.SUCCESS,
            why="Because.",
            requires=[],
            concurrent_with=[],
            source_confidence=SourceConfidence.NAMED_REFERENCE,
            source="https://en.wikipedia.org/wiki/Test",
        )

    def test_returns_list_of_historical_episodes(self):
        """run() returns a list of HistoricalEpisode objects."""
        from agents.ingestion import run

        ep = self._make_episode()
        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.decompose_topic", return_value=["subtopic1"]), \
             patch("agents.ingestion.fetch_wikipedia", return_value=[ep]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=True)):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            result = _run(run("deep learning"))

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_gateway_add_data_points_called_with_temporal_cognify(self):
        """run() calls gateway.add_data_points(episodes, temporal_cognify=True)."""
        from agents.ingestion import run

        ep = self._make_episode()
        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.decompose_topic", return_value=["subtopic1"]), \
             patch("agents.ingestion.fetch_wikipedia", return_value=[ep]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=True)):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            _run(run("deep learning"))

        mock_gw.add_data_points.assert_called()
        _, kwargs = mock_gw.add_data_points.call_args
        assert kwargs.get("temporal_cognify") is True

    def test_reasoned_fallback_triggered_when_all_recalls_fail(self):
        """run() triggers reasoned_fallback when self_check_recall always returns False."""
        from agents.ingestion import run
        from models.schemas import SourceConfidence

        ep = self._make_episode()
        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.decompose_topic", return_value=["subtopic1"]), \
             patch("agents.ingestion.fetch_wikipedia", return_value=[ep]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=False)), \
             patch("agents.ingestion.asyncio.sleep", new=AsyncMock()):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_cognee.recall = AsyncMock(return_value=[])
            mock_gw.add_data_points = AsyncMock()
            result = _run(run("some unknown topic"))

        reasoned_eps = [e for e in result if e.source_confidence == SourceConfidence.REASONED]
        assert len(reasoned_eps) >= 1

    def test_no_reasoned_fallback_when_recall_succeeds(self):
        """run() does NOT trigger reasoned_fallback when self_check_recall succeeds."""
        from agents.ingestion import run

        ep = self._make_episode()
        fallback_called = []

        async def fake_fallback(subtopics):
            fallback_called.append(True)
            return []

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.decompose_topic", return_value=["subtopic1"]), \
             patch("agents.ingestion.fetch_wikipedia", return_value=[ep]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=True)), \
             patch("agents.ingestion.reasoned_fallback", side_effect=fake_fallback):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            _run(run("deep learning"))

        assert fallback_called == [], "reasoned_fallback should not be called when recall succeeds"

    def test_retry_uses_exponential_backoff_sleeps(self):
        """run() sleeps with exponential backoff between recall retry attempts."""
        from agents.ingestion import run

        ep = self._make_episode()
        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.decompose_topic", return_value=["subtopic1"]), \
             patch("agents.ingestion.fetch_wikipedia", return_value=[ep]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=False)), \
             patch("agents.ingestion.asyncio.sleep", side_effect=fake_sleep), \
             patch("agents.ingestion.reasoned_fallback", new=AsyncMock(return_value=[])):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            _run(run("failing topic"))

        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0

    def test_no_add_data_points_called_when_no_episodes(self):
        """run() does not call gateway.add_data_points when all fetchers return empty."""
        from agents.ingestion import run

        with patch("agents.ingestion.cognee") as mock_cognee, \
             patch("agents.ingestion._gateway") as mock_gw, \
             patch("agents.ingestion.fetch_wikipedia", return_value=[]), \
             patch("agents.ingestion.fetch_arxiv", return_value=[]), \
             patch("agents.ingestion.self_check_recall", new=AsyncMock(return_value=True)):
            mock_cognee.consolidate_entity_descriptions_pipeline = AsyncMock()
            mock_gw.add_data_points = AsyncMock()
            _run(run("empty topic"))

        mock_gw.add_data_points.assert_not_called()
