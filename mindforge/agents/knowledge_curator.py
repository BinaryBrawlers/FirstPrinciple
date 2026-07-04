"""Knowledge Curator Agent for MindForge.

Responsible for ingesting educational content, extracting concepts and
prerequisite relationships using an LLM, and persisting everything via
Cognee's memory platform.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 15.1, 16.1
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import List

import httpx

from mindforge.config import settings
from mindforge.models import Concept, Relationship
from mindforge.protocol import IngestionResult
from mindforge.resilience import safe_forget, safe_remember

logger = logging.getLogger("mindforge.agents.knowledge_curator")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CONCEPT_EXTRACTION_PROMPT = """\
You are the Knowledge Curator for MindForge.
Extract educational concepts from the provided text.
Return ONLY valid JSON (no markdown fences) in this exact schema:
{
  "concepts": [
    {
      "id": "<slug e.g. gradient_descent>",
      "name": "<Human-readable name>",
      "definition": "<1-3 sentence definition>",
      "difficulty": "<beginner|intermediate|advanced>",
      "prerequisites": ["<concept_id>", ...]
    }
  ]
}
Extract between 1 and 20 atomic concepts. Prerequisites must reference other concept IDs in the same list.\
"""

_RELATIONSHIP_EXTRACTION_PROMPT = """\
You are the Knowledge Curator for MindForge.
Given this list of concept IDs and names, identify prerequisite relationships.
Return ONLY valid JSON (no markdown fences):
{
  "relationships": [
    {"from": "<concept_id>", "to": "<concept_id>"}
  ]
}
Only include relationships where concept A must be understood before concept B.\
"""


class KnowledgeCuratorAgent:
    """Agent that ingests content, extracts knowledge, and stores it in Cognee."""

    def __init__(self) -> None:
        # Strip the "mistral/" prefix that Cognee/LiteLLM needs but the
        # Mistral SDK does not understand.
        raw_model = settings.llm_model
        self._model = raw_model.removeprefix("mistral/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_content(
        self,
        content: str | bytes,
        dataset: str,
        source_metadata: dict,
    ) -> IngestionResult:
        """Ingest educational content, extract concepts & relationships, and persist.

        Args:
            content:         Raw text/markdown, a URL string, or raw bytes.
            dataset:         Cognee dataset scope to store the knowledge in.
            source_metadata: Dict with optional keys: title, author, year, url.

        Returns:
            IngestionResult with counts and extracted concept IDs.
        """
        # Resolve source_url from metadata or content (if it looks like a URL)
        source_url: str = source_metadata.get("url", "")

        # ── Decode bytes to str ──────────────────────────────────────────────
        if isinstance(content, bytes):
            text: str = content.decode("utf-8", errors="replace")
        else:
            text = content

        # ── Fetch URL content ────────────────────────────────────────────────
        if text.startswith("http://") or text.startswith("https://"):
            if not source_url:
                source_url = text
            text = await self._fetch_url(text)

        # ── LLM extraction ───────────────────────────────────────────────────
        self._source_metadata = source_metadata  # make available to _extract_concepts
        concepts: List[Concept] = await self._extract_concepts(text)
        relationships: List[Relationship] = await self._extract_relationships(concepts)

        # ── Persist via Cognee ───────────────────────────────────────────────
        # Installed Cognee only accepts str/file data, not dict — serialize first.
        import json as _json
        payload_str = _json.dumps({
            "content": text,
            "concepts": [dataclasses.asdict(c) for c in concepts],
            "relationships": [dataclasses.asdict(r) for r in relationships],
            "metadata": source_metadata,
        }, ensure_ascii=False)
        await safe_remember(
            data=payload_str,
            dataset=dataset,
            self_improvement=True,
        )

        return IngestionResult(
            concepts_count=len(concepts),
            relationships_count=len(relationships),
            concepts=[c.id for c in concepts],
            source_url=source_url,
            dataset=dataset,
        )

    async def remove_topic(self, dataset: str) -> None:
        """Remove an entire topic dataset from Cognee memory.

        Args:
            dataset: The dataset name to delete from Cognee.

        Requirements: 8.2
        """
        await safe_forget(dataset=dataset)

    async def remove_item(self, data_item: str) -> None:
        """Remove a specific data item from Cognee memory.

        Args:
            data_item: Identifier of the specific item to remove.

        Requirements: 8.3
        """
        await safe_forget(data_item=data_item)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_url(self, url: str) -> str:
        """Fetch the text body of *url* using an async HTTP client.

        Args:
            url: The URL to fetch.

        Returns:
            Response text.

        Raises:
            RuntimeError: If the HTTP request fails.
        """
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch URL %s: %s", url, exc)
            raise RuntimeError(f"Could not fetch content from URL '{url}': {exc}") from exc

    async def _extract_concepts(self, content: str) -> List[Concept]:
        """Call the LLM to extract educational concepts from *content*.

        Args:
            content: Raw text to analyse.

        Returns:
            List of Concept dataclass instances (may be empty on parse failure).
        """
        source_metadata: dict = getattr(self, "_source_metadata", {})

        try:
            from mistralai import Mistral  # local import to allow testing without the package

            client = Mistral(api_key=settings.mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CONCEPT_EXTRACTION_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            raw: str = response.choices[0].message.content
        except Exception as exc:
            logger.warning("LLM call for concept extraction failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)
            raw_concepts: list = parsed.get("concepts", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse concept extraction JSON response: %s", exc)
            return []

        concepts: List[Concept] = []
        for item in raw_concepts:
            try:
                concept = Concept(
                    id=item["id"],
                    name=item["name"],
                    definition=item["definition"],
                    difficulty=item.get("difficulty", "beginner"),
                    prerequisites=item.get("prerequisites", []),
                    # Source attribution from metadata
                    source_title=source_metadata.get("title", ""),
                    source_author=source_metadata.get("author", ""),
                    source_year=int(source_metadata.get("year", 0) or 0),
                    source_url=source_metadata.get("url", ""),
                )
                concepts.append(concept)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed concept entry %s: %s", item, exc)

        logger.info("Extracted %d concepts from content.", len(concepts))
        return concepts

    async def _extract_relationships(self, concepts: List[Concept]) -> List[Relationship]:
        """Call the LLM to identify prerequisite relationships among *concepts*.

        Args:
            concepts: Concepts extracted in the current batch.

        Returns:
            List of Relationship dataclass instances (may be empty on parse failure).
        """
        if not concepts:
            return []

        concept_list = "\n".join(
            f"- {c.id}: {c.name}" for c in concepts
        )

        try:
            from mistralai import Mistral

            client = Mistral(api_key=settings.mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _RELATIONSHIP_EXTRACTION_PROMPT},
                    {"role": "user", "content": concept_list},
                ],
            )
            raw: str = response.choices[0].message.content
        except Exception as exc:
            logger.warning("LLM call for relationship extraction failed: %s", exc)
            return []

        try:
            parsed = json.loads(raw)
            raw_relationships: list = parsed.get("relationships", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse relationship extraction JSON response: %s", exc)
            return []

        # Build a set of valid concept IDs to guard against hallucinated IDs
        valid_ids = {c.id for c in concepts}

        relationships: List[Relationship] = []
        for item in raw_relationships:
            from_id = item.get("from", "")
            to_id = item.get("to", "")
            if from_id in valid_ids and to_id in valid_ids and from_id != to_id:
                relationships.append(
                    Relationship(
                        from_concept=from_id,
                        to_concept=to_id,
                        relationship_type="prerequisite",
                    )
                )
            else:
                logger.warning(
                    "Skipping relationship with unknown concept IDs: %s → %s",
                    from_id,
                    to_id,
                )

        logger.info("Extracted %d relationships from concepts.", len(relationships))
        return relationships
