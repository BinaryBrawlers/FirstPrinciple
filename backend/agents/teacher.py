"""
Teacher Agent — Socratic Interactive Mode and Digest Mode.

IMPORTANT: backend.config must be imported first (before any cognee import)
to apply the LiteLLM patch and set COGNEE_SKIP_CONNECTION_TEST.

Task 7.1: Answer classifier
  classify_answer(answer, episode) -> Literal["matched-failure", "matched-success", "partial", "novel"]
"""
from __future__ import annotations

import logging
import os
from typing import Literal

# --- config MUST come before any cognee import ---
import config  # noqa: F401

import litellm

from models.schemas import HistoricalEpisode, Outcome

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------

ClassificationLabel = Literal["matched-failure", "matched-success", "partial", "novel"]

_VALID_LABELS: frozenset[str] = frozenset(
    {"matched-failure", "matched-success", "partial", "novel"}
)

_LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/mistral-small-latest")

# ---------------------------------------------------------------------------
# classify_answer
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """\
You are an answer classifier for a Socratic learning system.

Given a historical problem-solving episode and a learner's answer, classify the
answer into exactly one of four categories:

  matched-failure  — The learner's answer matches the historical *failure* outcome
                     described in the episode (i.e. the approach that was actually
                     tried and failed historically).

  matched-success  — The learner's answer matches the historical *success* outcome
                     described in the episode (i.e. the approach that eventually
                     worked historically).

  partial          — The learner is on the right track but the answer is incomplete,
                     vague, or only partially correct relative to the episode's solution.

  novel            — The learner proposes an approach not mentioned in the episode at
                     all (neither the historical failure nor the historical success).

Respond with ONLY the single label — one of:
  matched-failure
  matched-success
  partial
  novel

Do not include any explanation, punctuation, or other text.\
"""


def _build_classifier_user_prompt(answer: str, episode: HistoricalEpisode) -> str:
    return (
        f"Episode concept: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical attempted solution: {episode.attempted_solution}\n"
        f"Outcome: {episode.outcome.value}\n"
        f"Why: {episode.why}\n\n"
        f"Learner's answer: {answer}\n\n"
        "Classify the learner's answer using exactly one label."
    )


def _heuristic_classify(answer: str, episode: HistoricalEpisode) -> ClassificationLabel:
    """
    Simple keyword/overlap heuristic used as a fallback when the LLM is
    unavailable or returns an unexpected response.

    Logic:
    - If the answer overlaps substantially with the attempted_solution text,
      return the label that matches the episode outcome.
    - If the answer shares some words with the problem or solution, return "partial".
    - Otherwise return "novel".
    """
    answer_lower = answer.lower()
    solution_lower = episode.attempted_solution.lower()
    problem_lower = episode.problem_posed.lower()

    # Tokenise to word sets for overlap scoring
    answer_words = set(answer_lower.split())
    solution_words = set(solution_lower.split())
    problem_words = set(problem_lower.split())

    # Remove very common stop-words to avoid spurious overlap
    _STOP = {
        "the", "a", "an", "is", "was", "are", "were", "to", "of", "and",
        "in", "it", "that", "this", "for", "on", "with", "by", "at", "as",
        "be", "or", "not", "from", "how", "what", "why", "which", "we",
        "they", "he", "she", "its", "their", "our", "but", "so", "if",
    }
    answer_words -= _STOP
    solution_words -= _STOP
    problem_words -= _STOP

    if not answer_words:
        return "partial"

    overlap_with_solution = len(answer_words & solution_words) / len(answer_words)
    overlap_with_problem = len(answer_words & problem_words) / len(answer_words)

    if overlap_with_solution >= 0.35:
        # Strong overlap with the historical solution — classify by outcome
        if episode.outcome == Outcome.FAILURE:
            return "matched-failure"
        elif episode.outcome == Outcome.SUCCESS:
            return "matched-success"
        else:
            return "partial"

    if overlap_with_solution >= 0.15 or overlap_with_problem >= 0.15:
        return "partial"

    return "novel"


async def classify_answer(
    answer: str,
    episode: HistoricalEpisode,
) -> ClassificationLabel:
    """
    Classify a learner's answer against a HistoricalEpisode.

    Returns exactly one of:
        "matched-failure" | "matched-success" | "partial" | "novel"

    Uses the Mistral LLM for classification; falls back to a heuristic
    classifier if the LLM call fails or returns an unexpected label.

    Requirements: 5.2
    """
    if not answer or not answer.strip():
        logger.debug("classify_answer: empty answer — defaulting to 'partial'")
        return "partial"

    user_prompt = _build_classifier_user_prompt(answer.strip(), episode)

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # deterministic classification
            max_tokens=16,    # label is at most ~20 chars
        )
        raw: str = (response.choices[0].message.content or "").strip().lower()

        # Strip trailing punctuation the model might add despite instructions
        raw = raw.rstrip(".,;:")

        if raw in _VALID_LABELS:
            logger.debug("classify_answer: LLM returned label %r", raw)
            return raw  # type: ignore[return-value]

        # Try to find a valid label as a substring (e.g. "matched-success." or
        # extra whitespace the LLM snuck in)
        for label in _VALID_LABELS:
            if label in raw:
                logger.debug(
                    "classify_answer: extracted label %r from LLM output %r", label, raw
                )
                return label  # type: ignore[return-value]

        logger.warning(
            "classify_answer: unexpected LLM output %r; falling back to heuristic", raw
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "classify_answer: LLM call failed (%s); falling back to heuristic", exc
        )

    # Heuristic fallback
    label = _heuristic_classify(answer.strip(), episode)
    logger.debug("classify_answer: heuristic returned label %r", label)
    return label
