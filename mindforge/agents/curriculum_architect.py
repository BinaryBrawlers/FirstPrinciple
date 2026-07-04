"""Curriculum Architect Agent for MindForge.

Responsible for generating personalised, topologically-sorted learning paths
from the concept graph stored in Cognee, taking into account the learner's
mastery and feedback history.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.3, 7.4
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from mindforge.config import settings
from mindforge.models import Concept, ConceptStep, LearnerProfile, LearningPath, Relationship
from mindforge.resilience import safe_recall

logger = logging.getLogger("mindforge.agents.curriculum_architect")

# ---------------------------------------------------------------------------
# System prompt — LLM fallback when Cognee returns no concepts
# ---------------------------------------------------------------------------

_FALLBACK_CONCEPT_PROMPT = """\
You are the Curriculum Architect for MindForge.
Generate a concise ordered list of educational concepts required to learn the following goal.
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
Include between 3 and 10 concepts ordered from foundational to advanced.
Prerequisites must reference other concept IDs in the same list.\
"""

# Hours estimated per concept step (per design spec)
_HOURS_PER_CONCEPT: float = 2.0


class CurriculumArchitectAgent:
    """Agent that generates personalised learning paths for a learner and goal.

    The generation pipeline:
    1. Recall concept graph from Cognee (with LLM fallback).
    2. Recall learner profile from Cognee.
    3. Filter out already-mastered concepts.
    4. Topologically sort remaining concepts (Kahn's algorithm).
    5. Re-order by feedback weights so weak concepts come first.
    6. Package into a ``LearningPath`` and return.
    """

    def __init__(self) -> None:
        raw_model = settings.llm_model
        # Strip the "mistral/" prefix — the Mistral SDK doesn't accept it.
        self._model = raw_model.removeprefix("mistral/")
        # LLM provider attribute for potential future use / feature parity with other agents.
        self.llm_provider: str = settings.llm_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_learning_path(
        self,
        goal: str,
        learner_id: str,
        dataset: str,
    ) -> LearningPath:
        """Generate an ordered learning path for *learner_id* towards *goal*.

        Args:
            goal:       Natural-language description of what the learner wants to learn.
            learner_id: Identifier of the learner.
            dataset:    Cognee dataset scope to query for concept graph data.

        Returns:
            A ``LearningPath`` instance with ordered ``ConceptStep`` entries.
        """
        logger.info(
            "Generating learning path for learner=%s, goal=%r, dataset=%s",
            learner_id,
            goal,
            dataset,
        )

        # ── Step 1: Recall concept graph ────────────────────────────────────
        raw_graph = await safe_recall(
            query_text=f"all concepts and prerequisites for {goal}",
            dataset=dataset,
            limit=100,
        )
        concepts, relationships = self._parse_graph_results(raw_graph)

        # ── LLM fallback if Cognee returned nothing ──────────────────────────
        if not concepts:
            logger.warning(
                "Cognee returned no concepts for goal=%r; falling back to LLM.",
                goal,
            )
            concepts, relationships = await self._llm_fallback_concepts(goal)

        logger.info(
            "Retrieved %d concepts and %d relationships for goal=%r.",
            len(concepts),
            len(relationships),
            goal,
        )

        # ── Step 2: Recall learner profile ───────────────────────────────────
        raw_profile = await safe_recall(
            query_text=f"learner profile for {learner_id}",
            limit=1,
        )
        learner_profile = self._parse_learner_profile(raw_profile, learner_id)

        logger.info(
            "Learner %s has %d mastered concepts.",
            learner_id,
            len(learner_profile.mastered_concepts),
        )

        # ── Step 3: Filter mastered concepts ────────────────────────────────
        unmastered = [
            c for c in concepts
            if c.id not in learner_profile.mastered_concepts
            and c.mastery_percentage < 100.0
        ]
        logger.info(
            "After filtering mastered, %d concepts remain.",
            len(unmastered),
        )

        # ── Step 4: Topological sort ─────────────────────────────────────────
        sorted_concepts = self._topological_sort(unmastered, relationships)

        # ── Step 5: Apply feedback weights ───────────────────────────────────
        weighted_concepts = self._apply_feedback_weights(sorted_concepts, learner_profile)

        # ── Step 6: Build LearningPath ───────────────────────────────────────
        steps: List[ConceptStep] = [
            ConceptStep(
                concept_id=c.id,
                title=c.name,
                estimated_hours=_HOURS_PER_CONCEPT,
                prerequisites=list(c.prerequisites),
                difficulty=c.difficulty,
                order=idx + 1,
            )
            for idx, c in enumerate(weighted_concepts)
        ]

        total_hours = len(steps) * _HOURS_PER_CONCEPT
        path = LearningPath(
            learner_id=learner_id,
            goal=goal,
            concepts=steps,
            total_concepts=len(steps),
            estimated_hours=total_hours,
            generated_at=datetime.utcnow(),
        )

        logger.info(
            "Learning path generated: %d steps, %.1f estimated hours.",
            path.total_concepts,
            path.estimated_hours,
        )
        return path

    # ------------------------------------------------------------------
    # Graph parsing helpers
    # ------------------------------------------------------------------

    def _parse_graph_results(
        self,
        raw_results: list,
    ) -> tuple[List[Concept], List[Relationship]]:
        """Parse Cognee recall results into Concept and Relationship objects.

        Cognee can return dicts, JSON strings, or opaque result objects.
        This method handles all three cases defensively.

        Args:
            raw_results: Raw list returned by ``safe_recall``.

        Returns:
            A ``(concepts, relationships)`` tuple.
        """
        concepts: List[Concept] = []
        relationships: List[Relationship] = []

        for item in raw_results:
            data = self._coerce_to_dict(item)
            if data is None:
                continue

            # The item might be a wrapper containing "concepts" / "relationships"
            if "concepts" in data:
                for raw_c in data["concepts"]:
                    c = self._parse_concept(raw_c)
                    if c:
                        concepts.append(c)

            if "relationships" in data:
                for raw_r in data["relationships"]:
                    r = self._parse_relationship(raw_r)
                    if r:
                        relationships.append(r)

            # Or the item itself might be a concept (has "id" + "name")
            if "id" in data and "name" in data and "definition" in data:
                c = self._parse_concept(data)
                if c:
                    concepts.append(c)

            # Or a relationship (has "from_concept" / "to_concept" OR "from" / "to")
            if ("from_concept" in data and "to_concept" in data) or (
                "from" in data and "to" in data
            ):
                r = self._parse_relationship(data)
                if r:
                    relationships.append(r)

        # Deduplicate by id
        seen_concepts: Dict[str, Concept] = {}
        for c in concepts:
            seen_concepts.setdefault(c.id, c)

        seen_rels: set[tuple[str, str]] = set()
        unique_rels: List[Relationship] = []
        for r in relationships:
            key = (r.from_concept, r.to_concept)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(r)

        return list(seen_concepts.values()), unique_rels

    def _coerce_to_dict(self, item: Any) -> Optional[Dict[str, Any]]:
        """Try to coerce a recall result item to a plain dict.

        Handles:
        - Already a dict → return as-is.
        - A JSON string → parse.
        - An object with a ``__dict__`` → use that.
        - Anything else → return None.
        """
        if isinstance(item, dict):
            return item
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return None
        # Cognee result object — try __dict__
        if hasattr(item, "__dict__"):
            return vars(item)
        return None

    def _parse_concept(self, data: Any) -> Optional[Concept]:
        """Build a Concept from a dict, ignoring malformed entries."""
        d = self._coerce_to_dict(data)
        if d is None:
            return None
        try:
            return Concept(
                id=str(d["id"]),
                name=str(d["name"]),
                definition=str(d.get("definition", "")),
                difficulty=str(d.get("difficulty", "beginner")),
                prerequisites=list(d.get("prerequisites", [])),
                source_title=str(d.get("source_title", "")),
                source_author=str(d.get("source_author", "")),
                source_year=int(d.get("source_year", 0) or 0),
                source_url=str(d.get("source_url", "")),
                mastery_percentage=float(d.get("mastery_percentage", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed concept entry: %s (%s)", data, exc)
            return None

    def _parse_relationship(self, data: Any) -> Optional[Relationship]:
        """Build a Relationship from a dict, ignoring malformed entries."""
        d = self._coerce_to_dict(data)
        if d is None:
            return None
        try:
            from_id = str(d.get("from_concept") or d.get("from", ""))
            to_id = str(d.get("to_concept") or d.get("to", ""))
            if not from_id or not to_id or from_id == to_id:
                return None
            return Relationship(
                from_concept=from_id,
                to_concept=to_id,
                relationship_type=str(d.get("relationship_type", "prerequisite")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed relationship entry: %s (%s)", data, exc)
            return None

    # ------------------------------------------------------------------
    # Learner profile parsing
    # ------------------------------------------------------------------

    def _parse_learner_profile(
        self,
        raw_results: list,
        learner_id: str,
    ) -> LearnerProfile:
        """Parse Cognee recall results into a LearnerProfile.

        Returns a blank profile for *learner_id* if nothing useful was found.
        """
        for item in raw_results:
            d = self._coerce_to_dict(item)
            if d is None:
                continue
            if "learner_id" in d or "mastered_concepts" in d or "feedback_weights" in d:
                try:
                    return LearnerProfile(
                        learner_id=str(d.get("learner_id", learner_id)),
                        mastered_concepts=list(d.get("mastered_concepts", [])),
                        feedback_weights={
                            str(k): float(v)
                            for k, v in d.get("feedback_weights", {}).items()
                        },
                        session_history=list(d.get("session_history", [])),
                        overall_mastery=float(d.get("overall_mastery", 0.0)),
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Could not parse learner profile entry for %s: %s",
                        learner_id,
                        exc,
                    )

        logger.info(
            "No existing profile found for learner %s; using empty profile.",
            learner_id,
        )
        return LearnerProfile(learner_id=learner_id)

    # ------------------------------------------------------------------
    # Core algorithm: topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(
        self,
        concepts: List[Concept],
        relationships: List[Relationship],
    ) -> List[Concept]:
        """Sort *concepts* in dependency order using Kahn's algorithm.

        Concepts with no prerequisites come first.  If the graph contains a
        cycle, the remaining cycle members are appended in their original
        stable order.

        Args:
            concepts:      Concepts to sort.
            relationships: Directed prerequisite edges (from → to means
                           "from" must be learned before "to").

        Returns:
            Concepts in topological (dependency-first) order.
        """
        if not concepts:
            return []

        concept_map: Dict[str, Concept] = {c.id: c for c in concepts}
        concept_ids = set(concept_map.keys())

        # Build adjacency: prerequisite → dependents
        # Only include edges where both endpoints are in the current set
        in_degree: Dict[str, int] = {cid: 0 for cid in concept_ids}
        adjacency: Dict[str, List[str]] = {cid: [] for cid in concept_ids}

        # Also incorporate prerequisites encoded directly on Concept objects
        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id in concept_ids:
                    adjacency[prereq_id].append(concept.id)
                    in_degree[concept.id] += 1

        # Add edges from explicit Relationship objects
        for rel in relationships:
            if rel.from_concept in concept_ids and rel.to_concept in concept_ids:
                # Avoid double-counting if already captured via prerequisites
                if rel.to_concept not in adjacency[rel.from_concept]:
                    adjacency[rel.from_concept].append(rel.to_concept)
                    in_degree[rel.to_concept] += 1

        # Kahn's BFS
        # Seed with zero-in-degree nodes, preserving original list order as a
        # stable tiebreaker so the result is deterministic.
        original_order = {c.id: idx for idx, c in enumerate(concepts)}
        queue: deque[str] = deque(
            sorted(
                (cid for cid, deg in in_degree.items() if deg == 0),
                key=lambda cid: original_order[cid],
            )
        )

        sorted_ids: List[str] = []
        while queue:
            cid = queue.popleft()
            sorted_ids.append(cid)
            for neighbour in sorted(adjacency[cid], key=lambda n: original_order[n]):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        # Handle cycles: append remaining nodes in original stable order
        if len(sorted_ids) < len(concept_ids):
            remaining = sorted(
                (cid for cid in concept_ids if cid not in set(sorted_ids)),
                key=lambda cid: original_order[cid],
            )
            logger.warning(
                "Cycle detected in concept graph; appending %d remaining nodes in stable order.",
                len(remaining),
            )
            sorted_ids.extend(remaining)

        return [concept_map[cid] for cid in sorted_ids if cid in concept_map]

    # ------------------------------------------------------------------
    # Core algorithm: feedback weight ordering
    # ------------------------------------------------------------------

    def _apply_feedback_weights(
        self,
        concepts: List[Concept],
        profile: LearnerProfile,
    ) -> List[Concept]:
        """Re-order *concepts* so weak (low-weight) concepts come first.

        Within the same weight tier, the original topological order is
        preserved via a stable secondary sort key.

        Args:
            concepts: Topologically sorted list of concepts.
            profile:  Learner profile containing feedback weights.

        Returns:
            Concepts sorted ascending by feedback weight (weakest first),
            with topological order as a tiebreaker.
        """
        if not concepts:
            return []

        # Use enumerate index as stable tiebreaker to preserve topological order
        return sorted(
            concepts,
            key=lambda c: (
                profile.feedback_weights.get(c.id, 0.0),  # primary: ascending weight
                concepts.index(c),                          # secondary: topological order
            ),
        )

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    async def _llm_fallback_concepts(
        self,
        goal: str,
    ) -> tuple[List[Concept], List[Relationship]]:
        """Call the LLM to generate a default concept list for *goal*.

        Used when Cognee returns empty results (graceful degradation).

        Args:
            goal: The learning goal to generate concepts for.

        Returns:
            A ``(concepts, relationships)`` tuple parsed from the LLM response.
        """
        try:
            from mistralai import Mistral  # local import — not required for all paths

            client = Mistral(api_key=settings.mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _FALLBACK_CONCEPT_PROMPT},
                    {"role": "user", "content": f"Generate a learning path for: {goal}"},
                ],
            )
            raw: str = response.choices[0].message.content
        except Exception as exc:
            logger.warning("LLM fallback call failed for goal=%r: %s", goal, exc)
            return [], []

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM fallback response JSON: %s", exc)
            return [], []

        concepts: List[Concept] = []
        for item in parsed.get("concepts", []):
            c = self._parse_concept(item)
            if c:
                concepts.append(c)

        # Derive relationships from the prerequisites embedded in each concept
        concept_ids = {c.id for c in concepts}
        relationships: List[Relationship] = []
        seen: set[tuple[str, str]] = set()
        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id in concept_ids:
                    key = (prereq_id, concept.id)
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            Relationship(
                                from_concept=prereq_id,
                                to_concept=concept.id,
                                relationship_type="prerequisite",
                            )
                        )

        logger.info(
            "LLM fallback produced %d concepts and %d relationships for goal=%r.",
            len(concepts),
            len(relationships),
            goal,
        )
        return concepts, relationships
