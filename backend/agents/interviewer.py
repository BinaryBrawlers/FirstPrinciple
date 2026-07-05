"""
Interviewer Agent — Adversarial Testing Mode.

IMPORTANT: backend.config must be imported first (before any cognee import)
to apply the LiteLLM patch and set COGNEE_SKIP_CONNECTION_TEST.

Task 8.1: Session start — Track B recall with feedback_influence and question selection
  on_session_start(state)
      → calls cognee.recall() on Track B with feedback_influence weighting
      → calls select_questions(weak_points, track_a_failure_episodes)

  select_questions(weak_points, track_a_failure_episodes) -> list[dict]
      → returns single-concept questions drawn preferentially from failure episodes

Task 8.2: Inline grading, confidence prompt, and confidently-wrong penalty
  grade_answer(question, answer) -> str
      → calls LLM to grade answer as "correct", "partial", or "wrong"
  request_confidence_score() -> str
      → returns the confidence-prompt text (1–5 scale)
  compute_penalty(grade, confidence_score) -> float
      → applies HARSH_MULTIPLIER when confidence ∈ {4,5} and grade is wrong/partial
  on_answer(question, answer) -> AsyncGenerator[str, None]
      → streams grade feedback + confidence prompt
  on_confidence_received(grade, confidence_score) -> AsyncGenerator[str, None]
      → streams penalty feedback, calling out overconfidence when applicable

Task 8.3: on_session_end — misconception diff (implemented)
Task 8.4: SSE streaming, @cognee.agent_memory decorator (implemented)

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.8, 7.9
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, AsyncGenerator

# --- config MUST come before any cognee import ---
import config  # noqa: F401

import cognee
import litellm

from models.schemas import HistoricalEpisode, Outcome

if TYPE_CHECKING:
    from models.schemas import TutorState

logger = logging.getLogger(__name__)

_LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/mistral-small-latest")

# ---------------------------------------------------------------------------
# Question generation prompt
# ---------------------------------------------------------------------------

_QUESTION_SYSTEM_PROMPT = """\
You are an adversarial interviewer testing a learner's understanding of a concept.

Given a historical episode (especially one that ended in failure) and any known
weak points for this learner, generate a single interview question that:

1. Targets EXACTLY ONE concept — never combine two concepts in one question.
2. Is designed to surface the learner's misconceptions rather than test recall.
3. Prefers to probe WHY an approach failed (for failure-outcome episodes).
4. Is open-ended but answerable in a few sentences.

Respond with ONLY the question text — no preamble, no explanation, no numbering.\
"""


def _build_question_prompt(
    episode: HistoricalEpisode,
    weak_concept: str | None,
) -> str:
    """Build the user-side prompt for generating one interview question."""
    base = (
        f"Concept: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical approach: {episode.attempted_solution}\n"
        f"Outcome: {episode.outcome.value}\n"
        f"Why: {episode.why}\n"
    )
    if episode.source:
        base += f"Source: {episode.source}\n"
    if weak_concept:
        base += f"\nKnown learner weak point: {weak_concept}\n"
    base += "\nGenerate one single-concept interview question about this episode."
    return base


# ---------------------------------------------------------------------------
# _extract_weak_concepts  (graceful extraction from cognee.recall() output)
# ---------------------------------------------------------------------------


def _extract_weak_concepts(weak_points: object) -> list[str]:
    """
    Extract concept strings from whatever cognee.recall() returns for Track B.

    Handles None, empty list, TraitStatement dataclasses, or generic objects.
    Never raises.
    """
    if weak_points is None:
        return []

    if not isinstance(weak_points, (list, tuple)):
        weak_points = [weak_points]

    concepts: list[str] = []
    for wp in weak_points:
        # Duck-typed access — works for TraitStatement and cognee wrappers
        resolved = getattr(wp, "resolved", False)
        concept = getattr(wp, "concept", None)
        if not resolved and isinstance(concept, str) and concept.strip():
            concepts.append(concept.strip())
        elif isinstance(wp, str) and wp.strip():
            concepts.append(wp.strip())

    return concepts


# ---------------------------------------------------------------------------
# select_questions
# ---------------------------------------------------------------------------


async def select_questions(
    weak_points: object,
    track_a_failure_episodes: list[HistoricalEpisode],
) -> list[dict]:
    """
    Produce interview questions drawn preferentially from ``outcome: failure``
    episodes in Track A, targeting the learner's weak points from Track B.

    Each question targets exactly one concept (Requirement 7.2).
    Failure-outcome episodes are placed first in the candidate list
    (Requirement 7.3).
    Weak-point concepts from Track B recall are used to prioritise which
    episodes get a question (Requirement 7.1).

    Args:
        weak_points:
            The raw return value from ``cognee.recall()`` on the user's
            Track B graph (any type — gracefully handled).
        track_a_failure_episodes:
            List of ``HistoricalEpisode`` objects from Track A.  Typically the
            caller filters for failure outcomes first, but this function also
            applies its own preference ordering.

    Returns:
        A list of question dicts, each with the shape::

            {
                "episode_id":   str,          # source episode
                "concept":      str,          # exactly one concept
                "question":     str,          # the question text
                "from_failure": bool,         # True when episode.outcome==failure
            }

    Requirements: 7.1, 7.2, 7.3
    """
    weak_concepts = _extract_weak_concepts(weak_points)
    logger.debug(
        "select_questions: %d weak concept(s) from Track B: %r",
        len(weak_concepts),
        weak_concepts,
    )

    if not track_a_failure_episodes:
        logger.debug("select_questions: no episodes supplied — returning empty list")
        return []

    # ------------------------------------------------------------------
    # 1. Sort episodes: failure episodes first (Requirement 7.3), then
    #    by weak-point relevance (Requirement 7.1).
    # ------------------------------------------------------------------
    def _weak_point_score(ep: HistoricalEpisode) -> int:
        """Count how many weak-point concepts overlap with this episode's concept."""
        ep_lower = ep.concept.lower()
        return sum(1 for wc in weak_concepts if wc.lower() in ep_lower)

    def _sort_key(ep: HistoricalEpisode) -> tuple[int, int]:
        # (is_not_failure, -weak_point_score) → failure episodes with high
        # weak-point overlap sort to position 0.
        is_failure = ep.outcome == Outcome.FAILURE
        return (0 if is_failure else 1, -_weak_point_score(ep))

    sorted_episodes = sorted(track_a_failure_episodes, key=_sort_key)

    # ------------------------------------------------------------------
    # 2. Generate one single-concept question per episode.
    # ------------------------------------------------------------------
    questions: list[dict] = []

    for episode in sorted_episodes:
        # Find the best-matching weak concept for this episode (may be None)
        ep_lower = episode.concept.lower()
        matched_weak = next(
            (wc for wc in weak_concepts if wc.lower() in ep_lower),
            None,
        )

        question_text = await _generate_question(episode, matched_weak)

        questions.append(
            {
                "episode_id": episode.id,
                "concept": episode.concept,  # exactly one concept per question
                "question": question_text,
                "from_failure": episode.outcome == Outcome.FAILURE,
            }
        )
        logger.debug(
            "select_questions: generated question for episode %r (failure=%s): %r",
            episode.id,
            episode.outcome == Outcome.FAILURE,
            question_text[:80],
        )

    return questions


async def _generate_question(
    episode: HistoricalEpisode,
    weak_concept: str | None,
) -> str:
    """
    Call the LLM to generate one interview question for ``episode``.

    Falls back to a deterministic template when the LLM is unavailable.
    The question always targets exactly one concept (Requirement 7.2).
    """
    prompt = _build_question_prompt(episode, weak_concept)

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=150,
        )
        question: str = (response.choices[0].message.content or "").strip()
        if question:
            return question
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_generate_question: LLM call failed (%s); using template fallback", exc
        )

    # Deterministic fallback — still targets exactly one concept
    if episode.outcome == Outcome.FAILURE:
        return (
            f"The historical approach to '{episode.concept}' ultimately failed. "
            f"Why do you think '{episode.attempted_solution[:80]}' did not work, "
            f"and what fundamental limitation does that reveal?"
        )
    return (
        f"Explain, in your own words, how '{episode.concept}' was solved and "
        f"why that approach succeeded where earlier attempts had failed."
    )


# ---------------------------------------------------------------------------
# on_session_start
# ---------------------------------------------------------------------------


async def on_session_start(
    state: "TutorState",
    track_a_episodes: list[HistoricalEpisode],
) -> list[dict]:
    """
    Retrieve the user's current weak points from Track B and produce an
    ordered list of interview questions.

    Steps:
    1. Call ``cognee.recall()`` on ``user_{user_id}_traits`` with
       ``feedback_influence=True`` (Requirement 7.1).
    2. Pass the result and the Track A episode list to ``select_questions()``.

    Args:
        state:
            The current ``TutorState`` providing ``user_id`` and session
            context.
        track_a_episodes:
            All (or a pre-filtered subset of) Track A episodes available for
            this interview session.  The caller may pass only failure episodes
            or the full episode list; ``select_questions`` will still prefer
            failure episodes.

    Returns:
        A list of question dicts as produced by ``select_questions()``.

    Requirements: 7.1, 7.2, 7.3
    """
    user_id: str = state["user_id"]
    graph_name = f"user_{user_id}_traits"

    # Step 1 — Track B recall with feedback_influence weighting (Req 7.1)
    weak_points: object = None
    try:
        weak_points = await cognee.recall(
            graph_name=graph_name,
            query_params={"feedback_influence": True},
        )
        logger.debug(
            "on_session_start: Track B recall returned %r for user %r",
            type(weak_points).__name__,
            user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "on_session_start: cognee.recall() failed for %r (%s); "
            "proceeding without Track B weak points",
            graph_name,
            exc,
        )

    # Step 2 — Question selection (Req 7.2, 7.3)
    questions = await select_questions(weak_points, track_a_episodes)
    logger.info(
        "on_session_start: generated %d question(s) for user %r",
        len(questions),
        user_id,
    )
    return questions


# ---------------------------------------------------------------------------
# Task 8.2 — Inline grading, confidence prompt, and confidently-wrong penalty
# ---------------------------------------------------------------------------

# Penalty base values
_BASE_PENALTY_WRONG: float = 1.0
_BASE_PENALTY_PARTIAL: float = 0.5

# Applied when confidence ∈ {4, 5} and the grade is wrong or partial (Req 7.6)
HARSH_MULTIPLIER: float = 2.0

_VALID_GRADES: frozenset[str] = frozenset({"correct", "partial", "wrong"})

_GRADER_SYSTEM_PROMPT = """\
You are a concise, precise academic grader.

Given an interview question and a learner's answer, output EXACTLY one word:
- "correct"   — if the answer is substantially correct
- "partial"   — if the answer captures part of the concept but misses key elements
- "wrong"     — if the answer is incorrect or shows a fundamental misconception

Output only one of those three words. No explanation, no punctuation.\
"""


async def grade_answer(question: dict, answer: str) -> str:
    """
    Call the LLM to grade *answer* against *question*.

    Returns exactly one of: "correct", "partial", "wrong".

    Falls back to a heuristic when the LLM is unavailable or returns an
    unexpected response.

    Requirements: 7.5
    """
    question_text: str = question.get("question", question.get("concept", ""))
    concept: str = question.get("concept", "")

    user_prompt = (
        f"Question: {question_text}\n"
        f"Concept being tested: {concept}\n"
        f"Learner's answer: {answer}\n\n"
        "Grade the answer. Output exactly one word: correct, partial, or wrong."
    )

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        raw: str = (response.choices[0].message.content or "").strip().lower().rstrip(".,;:")

        if raw in _VALID_GRADES:
            logger.debug("grade_answer: LLM returned grade %r", raw)
            return raw

        # Try substring extraction in case the model added extra words
        for grade in _VALID_GRADES:
            if grade in raw:
                logger.debug(
                    "grade_answer: extracted grade %r from LLM output %r", grade, raw
                )
                return grade

        logger.warning(
            "grade_answer: unexpected LLM output %r; using heuristic fallback", raw
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "grade_answer: LLM call failed (%s); using heuristic fallback", exc
        )

    # Heuristic fallback: non-empty answer → "partial", empty → "wrong"
    fallback = "partial" if (answer and answer.strip()) else "wrong"
    logger.debug("grade_answer: heuristic fallback returned %r", fallback)
    return fallback


def request_confidence_score() -> str:
    """
    Return the prompt text asking the learner to self-report their confidence.

    This is a fixed prompt — no LLM call needed.

    Requirements: 7.4
    """
    return (
        "\n\nHow confident were you in that answer? "
        "Please rate your confidence on a scale from **1 to 5**:\n\n"
        "  1 — just guessing\n"
        "  2 — somewhat unsure\n"
        "  3 — moderately confident\n"
        "  4 — quite confident\n"
        "  5 — completely certain\n\n"
        "Reply with a single number (1–5)."
    )


def compute_penalty(grade: str, confidence_score: int) -> float:
    """
    Return the numeric penalty for an answer.

    Rules:
    - "correct"  → 0.0 regardless of confidence
    - "partial"  → base 0.5, multiplied by HARSH_MULTIPLIER if confidence ∈ {4, 5}
    - "wrong"    → base 1.0, multiplied by HARSH_MULTIPLIER if confidence ∈ {4, 5}

    Property 14 guarantee:
        penalty("wrong", 4) > penalty("wrong", 1)
        penalty("wrong", 5) > penalty("wrong", 2)

    Requirements: 7.6
    """
    if grade == "correct":
        return 0.0

    base = _BASE_PENALTY_WRONG if grade == "wrong" else _BASE_PENALTY_PARTIAL
    if confidence_score in {4, 5}:
        return base * HARSH_MULTIPLIER
    return base


async def on_answer(question: dict, answer: str) -> AsyncGenerator[str, None]:
    """
    Grade the answer and stream the grade feedback + confidence prompt to the
    client in the same turn.

    Yields tokens token-by-token for SSE compatibility.

    Requirements: 7.4, 7.5
    """
    grade = await grade_answer(question, answer)
    confidence_prompt_text = request_confidence_score()

    # Build grade-feedback message
    concept: str = question.get("concept", "the concept")
    if grade == "correct":
        grade_feedback = f"✓ **Correct!** Your answer demonstrates a solid understanding of {concept}."
    elif grade == "partial":
        grade_feedback = (
            f"◑ **Partially correct.** Your answer touches on {concept} but misses some "
            "key elements. We'll factor that in once you share your confidence."
        )
    else:  # wrong
        grade_feedback = (
            f"✗ **Incorrect.** Your answer doesn't quite capture {concept}. "
            "We'll factor that into your score once you share your confidence."
        )

    full_response = grade_feedback + confidence_prompt_text

    # Stream word-by-word for SSE
    words = full_response.split(" ")
    for i, word in enumerate(words):
        if i < len(words) - 1:
            yield word + " "
        else:
            yield word


async def on_confidence_received(
    grade: str,
    confidence_score: int,
) -> AsyncGenerator[str, None]:
    """
    Apply the grading penalty and stream feedback explaining the result.

    For confidently-wrong answers (confidence 4 or 5, grade "wrong") explicitly
    calls out overconfidence.

    Requirements: 7.6
    """
    penalty = compute_penalty(grade, confidence_score)
    high_confidence = confidence_score in {4, 5}

    if grade == "correct":
        if high_confidence:
            feedback = (
                "Great — you were right to be confident! "
                f"No penalty applied. Keep that calibration."
            )
        else:
            feedback = (
                f"You got it right despite lower confidence (score {confidence_score}). "
                "No penalty — trust your instincts more next time!"
            )
    elif grade == "partial":
        if high_confidence:
            feedback = (
                f"You rated your confidence at {confidence_score}/5, but your answer was only partial. "
                f"That overconfidence increases your penalty to **{penalty:.1f}** "
                f"(base 0.5 × {HARSH_MULTIPLIER:.0f}). "
                "Being confidently wrong on partial knowledge is a sign of a gap worth addressing."
            )
        else:
            feedback = (
                f"Your answer was partially correct. "
                f"Penalty applied: **{penalty:.1f}**. "
                "A low-confidence partial answer is understandable — keep building on what you know."
            )
    else:  # wrong
        if high_confidence:
            feedback = (
                f"You were confidently wrong — confidence {confidence_score}/5 with an incorrect answer. "
                f"This overconfidence means a harsher penalty: **{penalty:.1f}** "
                f"(base 1.0 × {HARSH_MULTIPLIER:.0f}). "
                "Recognising the limits of your knowledge is an important part of learning."
            )
        else:
            feedback = (
                f"Your answer was incorrect, but you had low confidence ({confidence_score}/5) — "
                f"penalty: **{penalty:.1f}**. "
                "At least you knew you weren't sure. Review this concept before moving on."
            )

    # Stream word-by-word
    words = feedback.split(" ")
    for i, word in enumerate(words):
        if i < len(words) - 1:
            yield word + " "
        else:
            yield word


# ---------------------------------------------------------------------------
# Task 8.4 — Misconception diff (pure function) and session-end streamer
# ---------------------------------------------------------------------------


def compute_misconception_diff(
    trait_snapshot: list[str],
    current_track_b: list[str],
) -> dict:
    """
    Compute a diff of misconception trait IDs between session start and now.

    This is a pure function with no I/O.

    Args:
        trait_snapshot:
            List of trait IDs captured at session start
            (``TutorState["trait_snapshot"]``).
        current_track_b:
            List of trait IDs currently active in Track B, fetched via
            ``cognee.recall()`` at session end.

    Returns:
        A dict with three keys:

        ``cleared``
            IDs present in *trait_snapshot* but absent in *current_track_b*
            — these misconceptions were resolved during the session.

        ``new``
            IDs present in *current_track_b* but absent in *trait_snapshot*
            — these misconceptions were surfaced / added during the session.

        ``persisted``
            IDs present in both — these misconceptions remain unresolved.

    Requirements: 7.7
    """
    snapshot_set = set(trait_snapshot or [])
    current_set = set(current_track_b or [])

    return {
        "cleared": sorted(snapshot_set - current_set),
        "new": sorted(current_set - snapshot_set),
        "persisted": sorted(snapshot_set & current_set),
    }


def _extract_trait_ids(recall_result: object) -> list[str]:
    """
    Safely extract trait ID strings from whatever ``cognee.recall()`` returns.

    Handles None, empty lists, TraitStatement dataclasses, and generic objects.
    Never raises.
    """
    if recall_result is None:
        return []

    if not isinstance(recall_result, (list, tuple)):
        recall_result = [recall_result]

    ids: list[str] = []
    for item in recall_result:
        trait_id = getattr(item, "id", None)
        if isinstance(trait_id, str) and trait_id.strip():
            ids.append(trait_id.strip())
        elif isinstance(item, str) and item.strip():
            ids.append(item.strip())

    return ids


async def on_session_end(state: "TutorState") -> AsyncGenerator[str, None]:
    """
    Compute and stream the misconception diff for this session.

    Steps:
    1. Call ``cognee.recall()`` on the user's Track B graph to get the
       current set of trait IDs.
    2. Compute the diff against ``state["trait_snapshot"]`` captured at
       session start.
    3. Stream a formatted diff summary token-by-token for SSE.

    Requirements: 7.7, 7.9
    """
    user_id: str = state["user_id"]
    graph_name = f"user_{user_id}_traits"

    # Step 1 — recall current Track B traits
    current_traits: list[str] = []
    try:
        result = await cognee.recall(graph_name=graph_name)
        current_traits = _extract_trait_ids(result)
        logger.debug(
            "on_session_end: Track B recall returned %d trait IDs for user %r",
            len(current_traits),
            user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "on_session_end: cognee.recall() failed for %r (%s); "
            "diff will use empty current state",
            graph_name,
            exc,
        )

    # Step 2 — compute diff
    snapshot: list[str] = state.get("trait_snapshot") or []
    diff = compute_misconception_diff(snapshot, current_traits)

    # Step 3 — build summary text
    lines: list[str] = ["## Session Summary — Misconception Diff\n"]

    cleared = diff["cleared"]
    new = diff["new"]
    persisted = diff["persisted"]

    if cleared:
        lines.append(
            f"\n✅ **Cleared this session** ({len(cleared)}):\n"
        )
        for tid in cleared:
            lines.append(f"  - {tid}\n")
    else:
        lines.append("\n✅ **Cleared this session:** none\n")

    if new:
        lines.append(
            f"\n🆕 **Newly surfaced** ({len(new)}):\n"
        )
        for tid in new:
            lines.append(f"  - {tid}\n")
    else:
        lines.append("\n🆕 **Newly surfaced:** none\n")

    if persisted:
        lines.append(
            f"\n⚠️  **Still open** ({len(persisted)}):\n"
        )
        for tid in persisted:
            lines.append(f"  - {tid}\n")
    else:
        lines.append("\n⚠️  **Still open:** none — great work!\n")

    summary = "".join(lines)

    # Stream token-by-token (word-level chunking for SSE — Req 7.9)
    words = summary.split(" ")
    for i, word in enumerate(words):
        if i < len(words) - 1:
            yield word + " "
        else:
            yield word


# ---------------------------------------------------------------------------
# Defensive @cognee.agent_memory decorator  (mirrors teacher.py pattern)
# ---------------------------------------------------------------------------


def _get_agent_memory_decorator() -> object:
    """
    Return ``@cognee.agent_memory(save_traces=True, with_session_memory=True)``
    if supported; fall back to ``save_session_traces`` (the parameter name used
    by older cognee versions); otherwise return a no-op passthrough decorator
    so the module imports cleanly in all environments.

    Requirements: 7.8
    """
    try:
        decorator_factory = cognee.agent_memory
        # Try the canonical param name from the spec first; fall back to the
        # name the installed cognee version actually exposes.
        try:
            return decorator_factory(save_traces=True, with_session_memory=True)
        except TypeError:
            return decorator_factory(save_session_traces=True, with_session_memory=True)
    except AttributeError:
        logger.debug(
            "cognee.agent_memory is not available in this cognee version; "
            "using no-op fallback decorator for interviewer_agent."
        )

        def _noop_decorator(fn):
            return fn

        return _noop_decorator


_agent_memory_decorator = _get_agent_memory_decorator()


# ---------------------------------------------------------------------------
# interviewer_agent  (entry point — Requirements 7.8, 7.9)
# ---------------------------------------------------------------------------


async def _interviewer_agent_impl(
    state: "TutorState",
    user_input: str,
) -> AsyncGenerator[str, None]:
    """
    Internal async-generator implementation of the Interviewer Agent.

    Routes to the appropriate handler based on ``user_input`` and session state:

    - ``"__session_start__"`` → :func:`on_session_start`, streams first question.
    - ``"__session_end__"``   → :func:`on_session_end`, streams misconception diff.
    - Any other input whose prefix matches ``"__confidence__:<grade>:<score>"``
      → :func:`on_confidence_received`, streams penalty feedback.
    - Any other input, when a current question is stored in state (via
      ``state["answer_history"]``), → :func:`on_answer`, streams grade + confidence
      prompt.
    - Fallback: streams a graceful "session not initialised" message.

    The public ``interviewer_agent`` wrapper carries the
    ``@cognee.agent_memory`` decorator, which requires a plain async function
    (not an async generator), so this impl is kept separate.

    Requirements: 7.1, 7.2, 7.3, 7.7, 7.8, 7.9
    """
    # ------------------------------------------------------------------
    # Route: session end
    # ------------------------------------------------------------------
    if user_input == "__session_end__":
        async for token in on_session_end(state):
            yield token
        return

    # ------------------------------------------------------------------
    # Route: confidence score received
    # Format expected: "__confidence__:<grade>:<score>"
    # e.g.  "__confidence__:wrong:4"
    # ------------------------------------------------------------------
    if user_input.startswith("__confidence__:"):
        parts = user_input.split(":", 2)
        if len(parts) == 3:
            _, grade, score_str = parts
            try:
                confidence_score = int(score_str.strip())
            except ValueError:
                confidence_score = 3  # safe default
            async for token in on_confidence_received(grade.strip(), confidence_score):
                yield token
            return
        # Malformed — fall through to session-start path

    # ------------------------------------------------------------------
    # Route: session start (explicit sentinel or first turn with no history)
    # ------------------------------------------------------------------
    if user_input == "__session_start__" or not state.get("answer_history"):
        topic: str = state.get("topic", "")
        track_a_episodes: list[HistoricalEpisode] = []

        try:
            results = await cognee.recall(
                graph_name="content_track",
                query=topic,
            )
            if isinstance(results, (list, tuple)):
                for item in results:
                    if isinstance(item, HistoricalEpisode):
                        track_a_episodes.append(item)
            logger.debug(
                "_interviewer_agent_impl: recalled %d Track A episode(s) for topic %r",
                len(track_a_episodes),
                topic,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_interviewer_agent_impl: Track A recall failed (%s); "
                "proceeding with empty episode list",
                exc,
            )

        questions = await on_session_start(state, track_a_episodes)

        if not questions:
            yield (
                "No interview questions could be generated for this session. "
                "Please ensure the topic has been ingested into Track A."
            )
            return

        # Stream the first question token-by-token
        first_q = questions[0]
        intro = f"**Interview — {first_q['concept']}**\n\n{first_q['question']}"
        for word in intro.split(" "):
            yield word + " "
        return

    # ------------------------------------------------------------------
    # Route: answer to current question
    # Retrieve the last unanswered question from answer_history context.
    # The caller is expected to supply the current question dict via
    # state["answer_history"][-1]["question"] when available, or we build
    # a minimal placeholder dict.
    # ------------------------------------------------------------------
    history = state.get("answer_history") or []
    current_question: dict = {}
    if history:
        last_entry = history[-1]
        # If the last entry already has a question dict stored by the router
        if isinstance(last_entry.get("question"), dict):
            current_question = last_entry["question"]
        else:
            # Build a minimal question dict from available context
            current_question = {
                "concept": last_entry.get("concept", state.get("topic", "the concept")),
                "question": last_entry.get("question_text", "Please elaborate on your answer."),
                "episode_id": last_entry.get("episode_id", ""),
                "from_failure": False,
            }

    async for token in on_answer(current_question, user_input):
        yield token


@_agent_memory_decorator
async def interviewer_agent(
    state: "TutorState",
    user_input: str,
) -> AsyncGenerator[str, None]:
    """
    Public SSE-streaming entry point for the Interviewer Agent.

    Decorated with ``@cognee.agent_memory`` so all interactions are traceable
    by the Trait Synthesis Agent (Requirement 7.8).

    ``cognee.agent_memory`` requires a plain ``async def`` (not an async
    generator), so this thin wrapper delegates to ``_interviewer_agent_impl``
    and returns the generator.  Callers iterate it token-by-token:

        async for token in interviewer_agent(state, user_input):
            yield token

    Special ``user_input`` sentinels (handled by ``_interviewer_agent_impl``):
      - ``"__session_start__"`` — begins the session and streams the first question
      - ``"__session_end__"``   — streams the misconception diff summary
      - ``"__confidence__:<grade>:<score>"`` — streams penalty feedback

    Requirements: 7.8, 7.9
    """
    return _interviewer_agent_impl(state, user_input)
