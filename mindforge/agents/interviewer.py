"""Interviewer Agent for MindForge.

Assesses learner knowledge through adaptive questioning.  Weak concepts are
retrieved from Cognee, questions are generated via the Mistral LLM, and
difficulty is adapted based on answer correctness (correct → "hard",
incorrect → "easy").

Requirements: 5.1, 5.2, 5.4, 5.5
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from mindforge.config import settings
from mindforge.protocol import AnswerEvaluation, InterviewSession
from mindforge.resilience import safe_recall

logger = logging.getLogger("mindforge.agents.interviewer")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_QUESTION_GEN_SYSTEM_PROMPT = """\
You are the Interviewer for MindForge. Your role is to test a learner's \
understanding of educational concepts through targeted questions.

Rules:
1. Generate a single, clear question that tests the specified concept.
2. Adapt the question's complexity to the requested difficulty level:
   - easy:   Recall / definition ("What is X?", "Name the steps of X.")
   - medium: Application / explanation ("How does X work?", "Why is X important?")
   - hard:   Analysis / synthesis ("Compare X and Y.", "What would happen if X changed?")
3. Include a concise model answer that captures the key points a correct response must contain.
4. Return ONLY valid JSON (no markdown fences) in this exact schema:
{
  "question_id": "<uuid string>",
  "question_text": "<the question to ask the learner>",
  "correct_answer": "<concise model answer with key points>",
  "difficulty": "<easy|medium|hard>"
}
"""

_ANSWER_EVAL_SYSTEM_PROMPT = """\
You are an answer evaluator for MindForge. Your role is to judge whether a \
learner's answer is correct given a model answer for reference.

Rules:
1. Mark the answer as correct (true) only if it captures the essential meaning \
   of the model answer — exact wording is not required.
2. Provide brief, constructive feedback (1-2 sentences).
3. Return ONLY valid JSON (no markdown fences) in this exact schema:
{
  "correct": <true|false>,
  "feedback": "<1-2 sentence explanation>"
}
"""


class InterviewerAgent:
    """Agent that conducts adaptive knowledge-assessment interviews."""

    def __init__(self) -> None:
        self._model = settings.llm_model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_interview(
        self,
        learner_id: str,
        session_id: str,
        dataset: str,
        num_questions: int = 5,
    ) -> InterviewSession:
        """Start an adaptive interview session targeting the learner's weakest concepts.

        Steps:
          1. Retrieve weak concepts via safe_recall.
          2. Pick the weakest concept (first result, or a fallback).
          3. Generate the first question at "medium" difficulty.
          4. Return an InterviewSession with all details.

        Args:
            learner_id:     Identifier of the learner being interviewed.
            session_id:     Active session ID for this interview.
            dataset:        Cognee dataset scope to query for weak concepts.
            num_questions:  Total number of questions planned for the session.

        Returns:
            InterviewSession populated with the first question.

        Requirements: 5.1, 5.2, 5.4, 5.5
        """
        # ── 1. Retrieve weak / low-mastery concepts ──────────────────────────
        weak_concepts: list = await safe_recall(
            query_text=f"concepts with low mastery for learner {learner_id}",
            dataset=dataset,
            limit=num_questions,
        )
        logger.debug(
            "safe_recall returned %d weak concept(s) for learner '%s'.",
            len(weak_concepts),
            learner_id,
        )

        # ── 2. Identify the weakest concept ──────────────────────────────────
        concept_id = self._extract_concept_id(weak_concepts, fallback="general_review")

        # ── 3. Generate first question at medium difficulty ───────────────────
        question = await self._generate_question(concept_id, difficulty="medium")

        return InterviewSession(
            session_id=session_id,
            total_questions=num_questions,
            current_question=1,
            question_id=question["question_id"],
            question_text=question["question_text"],
            concept_id=concept_id,
            difficulty=question["difficulty"],
            correct_answer=question["correct_answer"],
        )

    async def _generate_question(
        self,
        concept_id: str,
        difficulty: str,
    ) -> dict[str, str]:
        """Generate a question for *concept_id* at the requested *difficulty*.

        Uses the Mistral LLM.  Returns a safe fallback dict on any error so
        the interview can always proceed.

        Args:
            concept_id: The concept to test.
            difficulty: One of "easy", "medium", or "hard".

        Returns:
            Dict with keys: question_id, question_text, correct_answer, difficulty.
        """
        fallback = {
            "question_id": str(uuid.uuid4()),
            "question_text": f"Can you explain the concept of {concept_id}?",
            "correct_answer": f"A correct explanation of {concept_id}.",
            "difficulty": difficulty,
        }

        user_content = (
            f"Generate a {difficulty} difficulty question for the concept: {concept_id}."
        )

        try:
            from mistralai import Mistral  # local import — allows testing without SDK

            client = Mistral(api_key=settings.effective_mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _QUESTION_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw: str = response.choices[0].message.content

            try:
                parsed = json.loads(raw)
                # Ensure required keys are present; fill in defaults if missing.
                return {
                    "question_id": str(parsed.get("question_id") or uuid.uuid4()),
                    "question_text": parsed.get("question_text")
                    or fallback["question_text"],
                    "correct_answer": parsed.get("correct_answer")
                    or fallback["correct_answer"],
                    "difficulty": parsed.get("difficulty") or difficulty,
                }
            except (json.JSONDecodeError, AttributeError) as parse_exc:
                logger.warning(
                    "Failed to parse LLM JSON for question on '%s'; "
                    "using fallback. Error: %s",
                    concept_id,
                    parse_exc,
                )
                return fallback

        except Exception as llm_exc:
            logger.error(
                "LLM call failed for question generation on '%s'; "
                "using fallback. Error: %s",
                concept_id,
                llm_exc,
            )
            return fallback

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_concept_id(recall_results: list, fallback: str) -> str:
        """Extract the concept identifier from the first recall result.

        Supports dicts (key "id", "concept_id", or "name") and objects
        (attributes with the same names).  Returns *fallback* if nothing
        useful is found.

        Args:
            recall_results: List returned by safe_recall.
            fallback:       Value to return when no concept can be extracted.

        Returns:
            A concept ID string.
        """
        if not recall_results:
            return fallback

        first = recall_results[0]

        def _get(key: str) -> Any:
            if isinstance(first, dict):
                return first.get(key)
            return getattr(first, key, None)

        for key in ("id", "concept_id", "name"):
            value = _get(key)
            if value and str(value).strip():
                return str(value).strip()

        return fallback
