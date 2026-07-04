"""Teacher Agent for MindForge.

Delivers Socratic explanations of educational concepts using Cognee memory
for concept retrieval and session history, and a Mistral LLM for generating
chunked explanations with probing questions.

Requirements: 4.1, 4.2, 15.1, 15.2
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from mindforge.config import settings
from mindforge.protocol import EvaluationResult, TeachingResponse
from mindforge.resilience import safe_recall, safe_remember

logger = logging.getLogger("mindforge.agents.teacher")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_TEACHER_SYSTEM_PROMPT = """\
You are a Socratic tutor for MindForge. Your role is to teach concepts through \
guided discovery rather than direct lecturing.

Rules:
1. Present information in small chunks — 2 to 3 sentences maximum per explanation.
2. Always follow your explanation with a single probing question that checks \
   understanding or invites the learner to connect ideas.
3. Adapt your language to the learner's apparent level:
   - poor understanding: use simple analogies and everyday language
   - partial understanding: build on what they know, add one new idea
   - good understanding: introduce nuance, edge cases, or deeper implications
4. Always include source attribution in the explanation when available.
5. Return ONLY valid JSON (no markdown fences) in this exact schema:
{
  "explanation": "<2-3 sentence explanation with source attribution if available>",
  "question": "<single probing question>",
  "source": "<Title, Author (Year)> or empty string if unavailable"
}

Example output:
{
  "explanation": "Gradient descent is an optimisation algorithm that iteratively \
adjusts model parameters in the direction that reduces the loss function. \
Think of it as always taking a small step downhill on a hilly landscape. \
[Source: Deep Learning, Goodfellow et al. (2016)]",
  "question": "What do you think would happen if the step size (learning rate) \
were set too large?",
  "source": "Deep Learning, Goodfellow et al. (2016)"
}\
"""


_EVALUATOR_SYSTEM_PROMPT = """\
You are an evaluator for MindForge. Your role is to assess a learner's response \
against an expected concept and score their understanding.

Rules:
1. Score the response as "poor", "partial", or "good":
   - poor:    The response is incorrect, missing key ideas, or shows fundamental misunderstanding.
   - partial: The response shows some understanding but is incomplete or has minor errors.
   - good:    The response is accurate, complete, and demonstrates clear understanding.
2. Provide brief, constructive feedback (1-2 sentences) explaining the score.
3. Return ONLY valid JSON (no markdown fences) in this exact schema:
{
  "score": <float between 0.0 and 1.0>,
  "level": "<poor|partial|good>",
  "feedback": "<1-2 sentence explanation of the evaluation>"
}

Score mapping guide:
  poor:    0.0 – 0.39
  partial: 0.40 – 0.74
  good:    0.75 – 1.0\
"""


class TeacherAgent:
    """Agent that teaches concepts via the Socratic method using LLM + Cognee memory."""

    def __init__(self) -> None:
        # llm_model_name strips the "mistral/" prefix that Cognee/LiteLLM
        # needs but the Mistral SDK does not understand.
        self._model = settings.llm_model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def teach_concept(
        self,
        concept_id: str,
        learner_id: str,
        session_id: str,
        dataset: str,
    ) -> TeachingResponse:
        """Retrieve concept knowledge and generate a Socratic explanation.

        Args:
            concept_id:  The concept to teach (e.g. "gradient_descent").
            learner_id:  Identifier of the learner (for future personalisation).
            session_id:  Active session ID used to fetch recent dialogue history.
            dataset:     Cognee dataset scope containing the concept definitions.

        Returns:
            TeachingResponse with explanation, probing question, and source attribution.

        Requirements: 4.1, 4.2, 15.1, 15.2
        """
        # ── 1. Retrieve concept definition from Cognee ───────────────────────
        concept_data: list = await safe_recall(
            query_text=f"concept definition for {concept_id}",
            dataset=dataset,
        )
        logger.debug(
            "safe_recall returned %d concept result(s) for '%s'.",
            len(concept_data),
            concept_id,
        )

        # ── 2. Retrieve recent session history ───────────────────────────────
        session_history: list = await safe_recall(
            query_text="recent teaching interactions",
            session_id=session_id,
            limit=5,
        )
        logger.debug(
            "safe_recall returned %d session history result(s) for session '%s'.",
            len(session_history),
            session_id,
        )

        # ── 3. Build user prompt ─────────────────────────────────────────────
        user_content = self._build_user_prompt(concept_id, concept_data, session_history)

        # ── 4. Call Mistral LLM ──────────────────────────────────────────────
        explanation: str = f"Let's explore {concept_id}."
        question: str = f"What do you already know about {concept_id}?"
        source: str = ""

        try:
            from mistralai import Mistral  # local import — allows testing without SDK

            client = Mistral(api_key=settings.effective_mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw: str = response.choices[0].message.content

            # ── 5. Parse structured JSON response ───────────────────────────
            try:
                parsed = json.loads(raw)
                explanation = parsed.get("explanation", explanation)
                question = parsed.get("question", question)
                source = parsed.get("source", "")
            except (json.JSONDecodeError, AttributeError) as parse_exc:
                logger.warning(
                    "Failed to parse LLM JSON response for concept '%s'; "
                    "using raw text as explanation. Error: %s",
                    concept_id,
                    parse_exc,
                )
                explanation = raw
                question = ""

        except Exception as llm_exc:
            logger.error(
                "LLM call failed for concept '%s'; using default response. Error: %s",
                concept_id,
                llm_exc,
            )
            # Fall back to defaults set above.

        # ── 6. Source attribution fallback ───────────────────────────────────
        if not source:
            source = self._build_source_attribution(concept_data)

        return TeachingResponse(
            concept_id=concept_id,
            explanation=explanation,
            question=question,
            source=source,
            turn=1,
        )

    async def evaluate_response(
        self,
        learner_response: str,
        expected_concept: str,
        session_id: str,
    ) -> EvaluationResult:
        """Evaluate a learner's response against an expected concept.

        Retrieves concept context from memory, uses the LLM to score the response,
        persists the evaluation, and returns a structured result.

        Args:
            learner_response: The learner's answer or explanation to evaluate.
            expected_concept: The concept the learner was asked about.
            session_id:       Active session ID used to persist the evaluation.

        Returns:
            EvaluationResult with score, feedback, advance flag, and level.

        Requirements: 4.3, 4.4, 4.5, 4.6, 18.2
        """
        # ── 1. Recall concept context ────────────────────────────────────────
        concept_data: list = await safe_recall(
            query_text=f"concept definition for {expected_concept}",
            dataset=None,
        )
        logger.debug(
            "safe_recall returned %d concept result(s) for evaluation of '%s'.",
            len(concept_data),
            expected_concept,
        )

        # ── 2. Build user prompt ─────────────────────────────────────────────
        if concept_data:
            context_section = _serialize_recall_results(concept_data)
            user_content = (
                f"Concept: {expected_concept}\n\n"
                f"Concept knowledge retrieved from memory:\n{context_section}\n\n"
                f"Learner's response to evaluate:\n{learner_response}"
            )
        else:
            # Fallback: LLM-only evaluation without graph context (req 18.2)
            logger.debug(
                "No concept context found for '%s'; falling back to LLM-only evaluation.",
                expected_concept,
            )
            user_content = (
                f"Concept: {expected_concept}\n\n"
                f"Learner's response to evaluate:\n{learner_response}"
            )

        # ── 3. Call Mistral LLM ──────────────────────────────────────────────
        level: str = "partial"
        score: float = 0.5
        feedback: str = "Could not evaluate response."

        try:
            from mistralai import Mistral  # local import — allows testing without SDK

            client = Mistral(api_key=settings.effective_mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw: str = response.choices[0].message.content

            # ── 4. Parse structured JSON response ────────────────────────────
            try:
                parsed = json.loads(raw)
                level = parsed.get("level", "partial")
                if level not in ("poor", "partial", "good"):
                    level = "partial"
                score = float(parsed.get("score", 0.5))
                feedback = parsed.get("feedback", "Could not evaluate response.")
            except (json.JSONDecodeError, AttributeError, ValueError) as parse_exc:
                logger.warning(
                    "Failed to parse LLM JSON response for evaluation of '%s'; "
                    "using defaults. Error: %s",
                    expected_concept,
                    parse_exc,
                )

        except Exception as llm_exc:
            logger.error(
                "LLM call failed for evaluation of '%s'; using default response. Error: %s",
                expected_concept,
                llm_exc,
            )
            # Fall back to defaults set above.

        # ── 5. Persist evaluation to memory ──────────────────────────────────
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        try:
            await safe_remember(
                data={
                    "learner_response": learner_response,
                    "concept": expected_concept,
                    "evaluation": score,
                    "understanding_level": level,
                    "timestamp": timestamp,
                },
                session_id=session_id,
            )
        except Exception as mem_exc:
            logger.error(
                "safe_remember failed for evaluation of '%s'. Error: %s",
                expected_concept,
                mem_exc,
            )

        # ── 6. Return result ─────────────────────────────────────────────────
        return EvaluationResult(
            score=score,
            feedback=feedback,
            advance=(level == "good"),
            level=level,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        concept_id: str,
        concept_data: list,
        session_history: list,
    ) -> str:
        """Construct the user-role message for the LLM.

        Args:
            concept_id:      The concept being taught.
            concept_data:    Raw recall results for the concept definition.
            session_history: Raw recall results from recent teaching interactions.

        Returns:
            A formatted string combining all available context.
        """
        concept_section = _serialize_recall_results(concept_data)
        history_section = _serialize_recall_results(session_history)

        return (
            f"Concept to teach: {concept_id}\n\n"
            f"Concept knowledge retrieved from memory:\n{concept_section}\n\n"
            f"Recent session history (last 5 interactions):\n{history_section}"
        )

    def _build_source_attribution(self, concept_data: list) -> str:
        """Extract and format source metadata from the first recall result.

        Looks for ``source_title``, ``source_author``, and ``source_year`` fields
        on the first item in *concept_data* (supports both dict and object forms).

        Args:
            concept_data: List of recall results; only the first entry is inspected.

        Returns:
            A formatted attribution string like ``"Title, Author (Year)"`` or ``""``
            if no source metadata is present.
        """
        if not concept_data:
            return ""

        first = concept_data[0]

        # Support both dict and object (dataclass / arbitrary object) forms.
        def _get(key: str) -> Any:
            if isinstance(first, dict):
                return first.get(key, "")
            return getattr(first, key, "")

        title: str = str(_get("source_title") or "").strip()
        author: str = str(_get("source_author") or "").strip()
        year_raw = _get("source_year")
        year: str = str(year_raw).strip() if year_raw else ""

        if not title:
            return ""

        parts = [title]
        if author:
            parts.append(author)

        attribution = ", ".join(parts)
        if year and year != "0":
            attribution = f"{attribution} ({year})"

        return attribution


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _serialize_recall_results(results: list) -> str:
    """Convert a list of recall results to a readable string for the LLM prompt.

    Handles dicts, dataclass instances, and arbitrary objects gracefully.

    Args:
        results: List of items returned by safe_recall.

    Returns:
        Newline-separated string representation of each result, or a placeholder
        when the list is empty.
    """
    if not results:
        return "(none)"

    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        if isinstance(item, dict):
            try:
                lines.append(f"{i}. {json.dumps(item, ensure_ascii=False)}")
            except (TypeError, ValueError):
                lines.append(f"{i}. {item!r}")
        else:
            lines.append(f"{i}. {item!r}")

    return "\n".join(lines)
