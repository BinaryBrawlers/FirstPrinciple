"""Interviewer Agent for MindForge.

Assesses learner knowledge through adaptive questioning.  Weak concepts are
retrieved from Cognee, questions are generated via the Mistral LLM, and
difficulty is adapted based on answer correctness (correct → "hard",
incorrect → "easy").

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 14.2
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mindforge.config import settings
from mindforge.models import InterviewResults
from mindforge.protocol import AnswerEvaluation, InterviewSession
from mindforge.resilience import safe_recall, safe_remember

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

    async def evaluate_answer(
        self,
        question_id: str,
        learner_answer: str,
        correct_answer: str,
        concept_id: str,
        session_id: str,
    ) -> AnswerEvaluation:
        """Evaluate a learner's answer against the model answer using the LLM.

        Persists the evaluation result to Cognee memory for later aggregation
        by finish_interview, then returns a structured AnswerEvaluation.

        Steps:
          1. Build an LLM prompt comparing learner_answer to correct_answer.
          2. Parse the LLM JSON response for correctness and feedback.
          3. Adapt next-question difficulty (correct → "hard", incorrect → "easy").
          4. Persist evaluation via safe_remember.
          5. Return AnswerEvaluation.

        Args:
            question_id:    Unique ID of the question being answered.
            learner_answer: The learner's free-text answer.
            correct_answer: The model answer for comparison.
            concept_id:     The concept being tested.
            session_id:     Active session ID for memory persistence.

        Returns:
            AnswerEvaluation with correctness verdict, feedback, and next difficulty.

        Requirements: 5.3, 14.2
        """
        # ── 1. Build LLM user prompt ─────────────────────────────────────────
        user_content = (
            f"Concept: {concept_id}\n\n"
            f"Model answer: {correct_answer}\n\n"
            f"Learner's answer: {learner_answer}"
        )

        # ── 2. Call Mistral LLM ──────────────────────────────────────────────
        correct: bool = False
        feedback: str = "Could not evaluate answer."

        try:
            from mistralai import Mistral  # local import — allows testing without SDK

            client = Mistral(api_key=settings.effective_mistral_api_key)
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": _ANSWER_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw: str = response.choices[0].message.content

            try:
                parsed = json.loads(raw)
                correct = bool(parsed.get("correct", False))
                feedback = parsed.get("feedback", feedback)
            except (json.JSONDecodeError, AttributeError) as parse_exc:
                logger.warning(
                    "Failed to parse LLM JSON for answer evaluation of question '%s'; "
                    "using defaults. Error: %s",
                    question_id,
                    parse_exc,
                )

        except Exception as llm_exc:
            logger.error(
                "LLM call failed for answer evaluation of question '%s'; "
                "using default response. Error: %s",
                question_id,
                llm_exc,
            )

        # ── 3. Adapt difficulty for next question ────────────────────────────
        next_difficulty = "hard" if correct else "easy"

        # ── 4. Persist evaluation to Cognee memory ───────────────────────────
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        try:
            await safe_remember(
                data={
                    "question_id": question_id,
                    "concept_id": concept_id,
                    "correct": correct,
                    "learner_answer": learner_answer,
                    "timestamp": timestamp,
                },
                session_id=session_id,
            )
        except Exception as mem_exc:
            logger.error(
                "safe_remember failed for answer evaluation of question '%s'. Error: %s",
                question_id,
                mem_exc,
            )

        # ── 5. Return result ─────────────────────────────────────────────────
        return AnswerEvaluation(
            correct=correct,
            feedback=feedback,
            next_difficulty=next_difficulty,
        )

    async def finish_interview(
        self,
        session_id: str,
        dataset: str,
    ) -> InterviewResults:
        """Aggregate all answer evaluations from the session and compute final score.

        Recalls all answer records stored during the session, counts correct and
        total answers, then computes score = (correct_count / total) * 100.
        Weak concepts (those answered incorrectly) are collected for the profile.

        Args:
            session_id: The interview session to aggregate.
            dataset:    Cognee dataset scope (used as recall fallback context).

        Returns:
            InterviewResults with score, totals, and a list of weak concept IDs.

        Requirements: 5.6, 14.2
        """
        # ── 1. Recall all answer records from the session ────────────────────
        answer_records: list = await safe_recall(
            query_text="interview answer evaluations",
            session_id=session_id,
        )
        logger.debug(
            "safe_recall returned %d answer record(s) for session '%s'.",
            len(answer_records),
            session_id,
        )

        # ── 2. Tally correct answers and collect weak concepts ───────────────
        total = 0
        correct_count = 0
        weak_concepts: list[str] = []

        for record in answer_records:
            # Support both dict and object forms.
            if isinstance(record, dict):
                is_correct = record.get("correct")
                concept_id = record.get("concept_id", "")
            else:
                is_correct = getattr(record, "correct", None)
                concept_id = str(getattr(record, "concept_id", "") or "")

            # Only count records that look like answer evaluations (have a
            # "correct" field); skip unrelated memory entries.
            if is_correct is None:
                continue

            total += 1
            if is_correct:
                correct_count += 1
            else:
                if concept_id and concept_id not in weak_concepts:
                    weak_concepts.append(concept_id)

        # ── 3. Compute score ─────────────────────────────────────────────────
        score: float = (correct_count / total * 100.0) if total > 0 else 0.0

        logger.info(
            "Interview finished for session '%s': score=%.1f%% (%d/%d correct), "
            "weak_concepts=%s",
            session_id,
            score,
            correct_count,
            total,
            weak_concepts,
        )

        return InterviewResults(
            score=score,
            total_questions=total,
            correct_count=correct_count,
            weak_concepts=weak_concepts,
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
