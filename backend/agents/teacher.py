"""
Teacher Agent — Socratic Interactive Mode and Digest Mode.

IMPORTANT: backend.config must be imported first (before any cognee import)
to apply the LiteLLM patch and set COGNEE_SKIP_CONNECTION_TEST.

Task 7.1: Answer classifier
  classify_answer(answer, episode) -> Literal["matched-failure", "matched-success", "partial", "novel"]

Task 7.3: Socratic branching logic and stuck fallback
  on_user_answer(state, answer, episode) -> AsyncGenerator[str, None]
  stuck_fallback(episode) -> AsyncGenerator[str, None]
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, AsyncGenerator, Literal

# --- config MUST come before any cognee import ---
import config  # noqa: F401

import cognee
import litellm

from models.schemas import HistoricalEpisode, Outcome

if TYPE_CHECKING:
    from models.schemas import TutorState

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
# Helpless answer detection
# ---------------------------------------------------------------------------

_HELPLESS_PATTERNS: frozenset[str] = frozenset({
    "i don't know", "i dont know", "idk", "no idea", "no clue",
    "i don't get it", "i dont get it", "i don't understand",
    "i dont understand", "not sure", "i'm not sure", "im not sure",
    "help", "help me", "i'm confused", "im confused", "i'm lost",
    "im lost", "i have no idea", "what", "what?", "huh", "huh?",
    "i'm stuck", "im stuck", "i need help", "can you explain",
    "explain", "explain me", "explain it", "tell me", "tell me the answer",
    "give me the answer", "i give up", "?", "??", "???",
    "skip", "pass", "next", "i can't", "i cant",
})


def is_helpless_answer(answer: str) -> bool:
    """Detect if the learner's answer is a helpless/confused non-attempt."""
    cleaned = answer.strip().lower().rstrip(".!,")
    if not cleaned or len(cleaned) < 3:
        return True
    if cleaned in _HELPLESS_PATTERNS:
        return True
    # Check for very short answers that are just filler
    words = [w for w in cleaned.split() if len(w) > 2]
    if len(words) <= 2 and any(p in cleaned for p in (
        "don't know", "dont know", "no idea", "not sure", "help",
        "confused", "stuck", "lost", "explain", "don't get", "dont get",
        "don't understand", "dont understand", "give up",
    )):
        return True
    return False


def _build_starter_hint_prompt(episode: HistoricalEpisode) -> str:
    """Build a prompt that gives the learner a gentle starting point."""
    return (
        f"You are a warm, encouraging Socratic tutor. A learner is stuck on the "
        f"very first step — they haven't even attempted an answer yet.\n\n"
        f"Episode: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n\n"
        f"Do THREE things in order:\n"
        f"1. Reassure them (1 sentence — it's okay not to know).\n"
        f"2. Restate the core problem in simpler, everyday language "
        f"(2-3 sentences — use an analogy if helpful).\n"
        f"3. Give them a specific starting direction — not the answer, but a "
        f"concrete first step to think about (1-2 sentences).\n\n"
        f"Keep the total response under 5 sentences. Be warm, not condescending."
    )


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


# ---------------------------------------------------------------------------
# Prompt-building helpers (non-streaming, for internal use)
# ---------------------------------------------------------------------------

def _build_acknowledge_parallel_prompt(
    classification: ClassificationLabel,
    episode: HistoricalEpisode,
    answer: str,
) -> str:
    """
    Build a prompt asking the LLM to acknowledge that the learner has arrived
    at the same approach as the historical researcher (matched-failure or
    matched-success).

    Requirements: 5.3
    """
    outcome_word = "failure" if classification == "matched-failure" else "success"
    return (
        f"You are a Socratic tutor guiding a learner through historical problem-solving.\n\n"
        f"Episode: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical approach: {episode.attempted_solution}\n"
        f"Historical outcome: {outcome_word}\n"
        f"Why: {episode.why}\n"
        f"Source: {episode.source or 'historical record'}\n\n"
        f"The learner answered: {answer}\n\n"
        f"Their answer matches the historical {outcome_word} approach. "
        "Acknowledge warmly that they have arrived at the same idea as the historical "
        "researcher(s) involved. "
        "If it is a failure outcome, briefly explain what went wrong historically "
        "and why this was still a meaningful step forward. "
        "If it is a success outcome, celebrate the insight and briefly explain why "
        "this approach succeeded. "
        "Keep your response concise (2-4 sentences). Do not reveal what comes next."
    )


def _build_targeted_followup_prompt(
    episode: HistoricalEpisode,
    answer: str,
) -> str:
    """
    Build a prompt for a targeted Socratic follow-up question when the answer
    is partial — nudge toward the full solution without revealing it.

    Requirements: 5.4
    """
    return (
        f"You are a Socratic tutor guiding a learner through historical problem-solving.\n\n"
        f"Episode: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical approach: {episode.attempted_solution}\n"
        f"Why it matters: {episode.why}\n\n"
        f"The learner answered: {answer}\n\n"
        "Their answer is on the right track but incomplete. "
        "Ask a single targeted Socratic follow-up question that nudges them toward "
        "the complete historical solution without revealing it directly. "
        "The question should be specific to the gap in their answer. "
        "Do NOT give the answer. Keep your response to 1-2 sentences."
    )


def _build_novel_redirect_prompt(
    episode: HistoricalEpisode,
    answer: str,
) -> str:
    """
    Build a prompt that acknowledges a novel approach, briefly evaluates its
    historical merit, and redirects toward the canonical historical thread.

    Requirements: 5.5
    """
    return (
        f"You are a Socratic tutor guiding a learner through historical problem-solving.\n\n"
        f"Episode: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical approach: {episode.attempted_solution}\n"
        f"Historical outcome: {episode.outcome.value}\n"
        f"Why: {episode.why}\n\n"
        f"The learner proposed a novel approach not seen in the historical record: {answer}\n\n"
        "1. Acknowledge their creativity and briefly evaluate the historical merit of "
        "their novel approach (1-2 sentences — is it viable, does it have precedent elsewhere?). "
        "2. Then redirect them toward the canonical historical thread by asking "
        "a Socratic question about the approach researchers actually took. "
        "Keep the full response to 3-4 sentences."
    )


def _build_stuck_fallback_prompt(episode: HistoricalEpisode) -> str:
    """
    Build a prompt that produces a structured fallback response with all four
    required sections when the learner has been stuck for two nudges.

    Requirements: 5.6
    """
    return (
        f"You are a Socratic tutor. A learner has been stuck on the following episode "
        f"after two attempts. Provide a structured response with EXACTLY these four "
        f"sections, each labelled with the bold header shown:\n\n"
        f"**Problem framing** — Restate the core problem this episode addresses "
        f"in 1-2 clear sentences.\n\n"
        f"**Solution hint** — Give a gentle hint toward the historical solution "
        f"without revealing the full answer (1-2 sentences).\n\n"
        f"**Engineering Insight** — State the deeper engineering principle or "
        f"insight behind the solution (1-2 sentences).\n\n"
        f"**Historical note** — Provide brief historical context: who solved this, "
        f"when, and why it mattered (1-2 sentences).\n\n"
        f"Episode details:\n"
        f"Concept: {episode.concept}\n"
        f"Problem posed: {episode.problem_posed}\n"
        f"Historical approach: {episode.attempted_solution}\n"
        f"Outcome: {episode.outcome.value}\n"
        f"Why it matters: {episode.why}\n"
        f"Source: {episode.source or 'historical record'}\n"
        f"Date: {episode.published_date or 'unknown'}\n\n"
        "Output all four sections in order. Each section MUST appear with its bold header."
    )


# ---------------------------------------------------------------------------
# stuck_fallback
# ---------------------------------------------------------------------------

async def stuck_fallback(episode: HistoricalEpisode) -> AsyncGenerator[str, None]:
    """
    Deliver a structured fallback response covering all four required sections:
    Problem framing, Solution hint, Engineering Insight, Historical note.

    Called when nudge_count >= 2. The caller is responsible for resetting
    nudge_count to 0 after consuming this generator.

    Requirements: 5.6
    """
    prompt = _build_stuck_fallback_prompt(episode)
    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Socratic tutor. When asked to produce a "
                        "structured fallback, always include all four labelled sections "
                        "in the exact order requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=600,
        )
        async for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as exc:  # noqa: BLE001
        logger.warning("stuck_fallback: LLM streaming failed (%s); yielding static fallback", exc)
        # Graceful static fallback so the caller always gets something
        yield (
            f"**Problem framing** — {episode.problem_posed}\n\n"
            f"**Solution hint** — Consider the core idea behind: {episode.attempted_solution[:80]}...\n\n"
            f"**Engineering Insight** — {episode.why}\n\n"
            f"**Historical note** — This is recorded as a {episode.outcome.value} outcome"
            + (f" ({episode.source})" if episode.source else "") + "."
        )


# ---------------------------------------------------------------------------
# select_next_episode  (Track B recall + requires/concurrent_with traversal)
# ---------------------------------------------------------------------------

def _extract_misconception_concepts(traits: object) -> list[str]:
    """
    Extract active misconception concept strings from whatever cognee.recall()
    returns.  Handles None, empty list, TraitStatement dataclasses, or generic
    objects gracefully — never raises.
    """
    if traits is None:
        return []

    if not isinstance(traits, (list, tuple)):
        # Could be a single object — wrap it so the loop below works
        traits = [traits]

    concepts: list[str] = []
    for t in traits:
        # Prefer duck-typed attribute access (works for TraitStatement and
        # any cognee-internal wrapper with the same field names)
        trait_type = getattr(t, "trait_type", None)
        resolved = getattr(t, "resolved", False)
        concept = getattr(t, "concept", None)
        if (
            trait_type == "misconception"
            and not resolved
            and isinstance(concept, str)
            and concept.strip()
        ):
            concepts.append(concept.strip())

    return concepts


def _misconception_score(episode: HistoricalEpisode, misconceptions: list[str]) -> int:
    """
    Count how many active misconception concepts appear (case-insensitive
    substring match) in the episode's concept field.
    """
    if not misconceptions:
        return 0
    ep_concept_lower = episode.concept.lower()
    return sum(1 for m in misconceptions if m.lower() in ep_concept_lower)


async def select_next_episode(
    state: "TutorState",
    all_episodes: list[HistoricalEpisode],
) -> "HistoricalEpisode | None":
    """
    Select the best next episode for the learner, using Track B recall to
    cross-reference active misconceptions and ``requires`` / ``concurrent_with``
    edges for traversal order.

    Priority (highest → lowest):
    1. Unresolved mandatory prerequisites of the current episode
       (current_episode.requires entries not yet resolved)
    2. Natural next-step episodes that list the current episode in their own
       ``requires`` field
    3. ``concurrent_with`` siblings of the current episode
    4. Any other unresolved episode in all_episodes

    Within each priority tier, episodes are sorted by misconception overlap
    score (descending) so that active weak points are addressed first.

    Returns None when all episodes are resolved (session complete).

    Requirements: 5.7, 5.8
    """
    # --- 1. Call cognee.recall() FIRST (Requirement 5.7) ---
    traits = None
    try:
        traits = await cognee.recall(graph_name=f"user_{state['user_id']}_traits")
    except Exception as exc:  # noqa: BLE001
        logger.debug("select_next_episode: cognee.recall() failed (%s); proceeding without traits", exc)

    misconceptions = _extract_misconception_concepts(traits)
    logger.debug(
        "select_next_episode: Track B recall returned %d active misconception(s): %r",
        len(misconceptions),
        misconceptions,
    )

    # --- 2. Build resolved episode ID set from answer_history ---
    resolved_ids: set[str] = {
        entry["episode_id"]
        for entry in state.get("answer_history", [])
        if entry.get("classification") == "matched-success"
    }

    # --- 3. Build episode lookup map ---
    episode_map: dict[str, HistoricalEpisode] = {ep.id: ep for ep in all_episodes}

    # --- 4. Find current episode object ---
    current_ep = episode_map.get(state.get("current_episode", ""))
    if current_ep is None:
        # Fallback: first unresolved episode
        for ep in all_episodes:
            if ep.id not in resolved_ids:
                logger.debug(
                    "select_next_episode: current episode not found; falling back to first unresolved: %s",
                    ep.id,
                )
                return ep
        logger.debug("select_next_episode: all episodes resolved (fallback path)")
        return None

    # --- 5. Mandatory prerequisites (requires of current not yet resolved) ---
    mandatory_prereqs: list[HistoricalEpisode] = [
        episode_map[ep_id]
        for ep_id in current_ep.requires
        if ep_id in episode_map and ep_id not in resolved_ids
    ]

    # --- 6. Natural next steps (episodes that require the current one) ---
    natural_next: list[HistoricalEpisode] = [
        ep
        for ep in all_episodes
        if current_ep.id in ep.requires and ep.id not in resolved_ids
    ]

    # --- 7. concurrent_with siblings (secondary candidates) ---
    concurrent_siblings: list[HistoricalEpisode] = [
        episode_map[ep_id]
        for ep_id in current_ep.concurrent_with
        if ep_id in episode_map and ep_id not in resolved_ids
    ]

    # --- 8. Score and sort each tier by misconception overlap ---
    def sort_by_score(candidates: list[HistoricalEpisode]) -> list[HistoricalEpisode]:
        return sorted(
            candidates,
            key=lambda ep: _misconception_score(ep, misconceptions),
            reverse=True,
        )

    mandatory_prereqs = sort_by_score(mandatory_prereqs)
    natural_next = sort_by_score(natural_next)
    concurrent_siblings = sort_by_score(concurrent_siblings)

    # --- 9. Selection priority ---
    # Priority 1: unresolved mandatory prerequisites
    if mandatory_prereqs:
        chosen = mandatory_prereqs[0]
        logger.debug(
            "select_next_episode: selected mandatory prerequisite %s (score=%d)",
            chosen.id,
            _misconception_score(chosen, misconceptions),
        )
        return chosen

    # Priority 2: natural next-step episodes
    if natural_next:
        chosen = natural_next[0]
        logger.debug(
            "select_next_episode: selected natural next step %s (score=%d)",
            chosen.id,
            _misconception_score(chosen, misconceptions),
        )
        return chosen

    # Priority 3: concurrent_with siblings
    if concurrent_siblings:
        chosen = concurrent_siblings[0]
        logger.debug(
            "select_next_episode: selected concurrent sibling %s (score=%d)",
            chosen.id,
            _misconception_score(chosen, misconceptions),
        )
        return chosen

    # Priority 4: any other unresolved episode (fallback)
    presented_or_resolved = resolved_ids | {current_ep.id}
    for ep in all_episodes:
        if ep.id not in presented_or_resolved:
            logger.debug(
                "select_next_episode: fallback — selected unrelated unresolved episode %s",
                ep.id,
            )
            return ep

    # Priority 5: session complete
    logger.debug("select_next_episode: all episodes resolved — session complete")
    return None


async def get_next_episode(
    state: "TutorState",
    all_episodes: list[HistoricalEpisode],
) -> "HistoricalEpisode | None":
    """
    Convenience wrapper: calls cognee.recall() on Track B first,
    then selects the next episode. This is the function callers should
    use after on_user_answer completes.

    Requirements: 5.7, 5.8
    """
    return await select_next_episode(state, all_episodes)


# ---------------------------------------------------------------------------
# on_digest  (Digest Mode — Requirements 6.1, 6.2)
# ---------------------------------------------------------------------------

_DIGEST_SYSTEM_PROMPT = (
    "You are a learning assistant summarising a video transcript against a "
    "curated knowledge base. Be concise and informative. "
    "Do NOT pose questions. Do NOT suggest what the learner should study next."
)


def _build_digest_user_prompt(transcript: str, episode_concepts: str) -> str:
    return (
        "You are a learning assistant summarising a video transcript against a "
        "curated knowledge base.\n\n"
        f"Track A episodes found (concepts):\n{episode_concepts}\n\n"
        f"Transcript to summarise:\n{transcript[:3000]}"
        "  (truncated for context window)\n\n"
        "Summarise the key concepts from the transcript that align with the "
        "Track A knowledge base.\n"
        "For each aligned concept, note:\n"
        "1. Which episode concept it maps to\n"
        "2. What the transcript adds or confirms\n"
        "3. Any gaps (concepts in Track A not covered by the transcript)\n\n"
        "Do NOT pose questions. Do NOT suggest what the learner should study next.\n"
        "Keep the summary to 3-5 key points."
    )


async def on_digest(
    state: "TutorState",
    transcript: str,
) -> AsyncGenerator[str, None]:
    """
    Summarise a video transcript against Track A (content_track) without
    advancing the episode position.

    - Calls cognee.recall(graph_name="content_track", ...) to retrieve
      related Track A episodes.
    - Uses litellm streaming to summarise the transcript's key points against
      those episodes.
    - Does NOT update state["current_episode"]; episode position remains
      unchanged.
    - Does NOT pose any Socratic questions.

    Requirements: 6.1, 6.2
    """
    # --- 1. Build a query from the transcript (first 200 chars is enough) ---
    recall_query = transcript[:200].strip() or "knowledge concepts"

    # --- 2. Recall related Track A episodes ---
    episode_concepts = ""
    try:
        results = await cognee.recall(
            graph_name="content_track",
            query=recall_query,
        )
        if results:
            # Extract concept strings from whatever cognee.recall() returns
            concepts: list[str] = []
            if isinstance(results, (list, tuple)):
                for item in results:
                    concept = getattr(item, "concept", None)
                    if isinstance(concept, str) and concept.strip():
                        concepts.append(concept.strip())
                    elif isinstance(item, str) and item.strip():
                        concepts.append(item.strip())
            elif isinstance(results, str) and results.strip():
                concepts.append(results.strip())

            if concepts:
                episode_concepts = "\n".join(f"- {c}" for c in concepts)

        if not episode_concepts:
            episode_concepts = "(No matching Track A episodes found — summarising against general knowledge)"

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "on_digest: cognee.recall() failed (%s); proceeding without Track A context", exc
        )
        episode_concepts = "(Track A recall unavailable — summarising transcript directly)"

    # --- 3. Stream the summary via litellm ---
    user_prompt = _build_digest_user_prompt(transcript, episode_concepts)

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _DIGEST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=800,
        )
        async for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as exc:  # noqa: BLE001
        logger.warning("on_digest: LLM streaming failed (%s); yielding minimal summary", exc)
        yield (
            "Digest mode summary unavailable (LLM error). "
            f"Transcript received ({len(transcript)} chars). "
            f"Related Track A concepts: {episode_concepts}"
        )

    # NOTE: state["current_episode"] is deliberately NOT modified here.
    # Requirements 6.1, 6.2: digest mode does not advance episode position.


# ---------------------------------------------------------------------------
# Defensive @cognee.agent_memory decorator
# ---------------------------------------------------------------------------

def _get_agent_memory_decorator() -> object:
    """
    Return cognee.agent_memory(save_traces=True, with_session_memory=True) if
    the current cognee installation supports it; otherwise return a no-op
    passthrough decorator so the module imports cleanly in all environments.
    """
    try:
        decorator_factory = cognee.agent_memory
        # The installed cognee version uses `save_session_traces` (not `save_traces`)
        return decorator_factory(save_session_traces=True, with_session_memory=True)
    except AttributeError:
        logger.debug(
            "cognee.agent_memory is not available in this cognee version; "
            "using no-op fallback decorator for teacher_agent."
        )

        def _noop_decorator(fn):
            return fn

        return _noop_decorator


_agent_memory_decorator = _get_agent_memory_decorator()


# ---------------------------------------------------------------------------
# teacher_agent  (top-level SSE-streaming entry point — Requirements 5.10, 5.11, 6.1, 6.2)
# ---------------------------------------------------------------------------

async def _teacher_agent_impl(
    state: "TutorState",
    user_input: str,
) -> AsyncGenerator[str, None]:
    """
    Internal async-generator implementation of the Teacher Agent.

    Branches on state["mode"]:
      - "digest"  → delegates to on_digest(state, user_input), yields all tokens.
                    Episode position is NOT advanced.
      - "teacher" → retrieves the current episode from Track A via cognee.recall()
                    and delegates to on_user_answer(state, user_input, episode).
                    If no episode is found, yields a graceful error message.

    Not decorated directly — the public ``teacher_agent`` wrapper carries the
    ``@cognee.agent_memory`` decorator, which requires a plain async function
    (not an async generator).

    Requirements: 5.10, 5.11, 6.1, 6.2
    """
    mode: str = state.get("mode", "teacher")

    if mode == "digest":
        # Digest mode: summarise transcript without advancing episode position
        async for token in on_digest(state, user_input):
            yield token
        return

    # --- Teacher (Socratic) mode ---
    # Retrieve the current episode from Track A via cognee.recall()
    current_ep_id: str = state.get("current_episode", "")
    episode: "HistoricalEpisode | None" = None

    try:
        results = await cognee.recall(
            graph_name="content_track",
            query=current_ep_id,
        )
        if results:
            # Accept the first result that looks like a HistoricalEpisode
            if isinstance(results, (list, tuple)):
                for item in results:
                    if isinstance(item, HistoricalEpisode):
                        episode = item
                        break
                    # cognee may return generic objects with episode-like attrs
                    if (
                        hasattr(item, "id")
                        and hasattr(item, "concept")
                        and hasattr(item, "problem_posed")
                        and hasattr(item, "attempted_solution")
                        and hasattr(item, "outcome")
                        and hasattr(item, "why")
                    ):
                        # Attempt to reconstruct a HistoricalEpisode from attrs
                        try:
                            episode = HistoricalEpisode(
                                id=item.id,
                                concept=item.concept,
                                problem_posed=item.problem_posed,
                                attempted_solution=item.attempted_solution,
                                outcome=item.outcome,
                                why=item.why,
                                requires=getattr(item, "requires", []),
                                concurrent_with=getattr(item, "concurrent_with", []),
                                source_confidence=getattr(
                                    item, "source_confidence", None
                                ) or __import__("models.schemas", fromlist=["SourceConfidence"]).SourceConfidence.REASONED,
                                source=getattr(item, "source", None),
                                published_date=getattr(item, "published_date", None),
                            )
                            break
                        except Exception:  # noqa: BLE001
                            continue
            elif isinstance(results, HistoricalEpisode):
                episode = results
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "teacher_agent: cognee.recall() for episode %r failed (%s); "
            "episode will be None",
            current_ep_id,
            exc,
        )

    if episode is None:
        logger.warning(
            "teacher_agent: no episode found for current_episode=%r; "
            "yielding graceful message",
            current_ep_id,
        )
        yield (
            "I wasn't able to retrieve your current episode from the knowledge base. "
            "Please check that the topic has been ingested and try again, or ask your "
            "instructor to reload the episode content."
        )
        return

    # Delegate to the Socratic branching logic
    async for token in on_user_answer(state, user_input, episode):
        yield token


# ---------------------------------------------------------------------------
# teacher_agent  (public entry point — plain async fn so @agent_memory works)
# ---------------------------------------------------------------------------

@_agent_memory_decorator
async def teacher_agent(
    state: "TutorState",
    user_input: str,
) -> AsyncGenerator[str, None]:
    """
    Public SSE-streaming entry point for the Teacher Agent.

    ``cognee.agent_memory`` requires a plain ``async def`` (not an async
    generator), so this thin wrapper delegates to ``_teacher_agent_impl``
    and returns the generator.  Callers iterate it token-by-token:

        async for token in teacher_agent(state, user_input):
            yield token

    Decorated with @cognee.agent_memory(save_session_traces=True,
    with_session_memory=True) so all interactions are traceable by the
    Trait Synthesis Agent.

    Requirements: 5.10, 5.11, 6.1, 6.2
    """
    return _teacher_agent_impl(state, user_input)


# ---------------------------------------------------------------------------
# on_user_answer  (Socratic branching logic)
# ---------------------------------------------------------------------------

async def on_user_answer(
    state: "TutorState",
    answer: str,
    episode: HistoricalEpisode,
) -> AsyncGenerator[str, None]:
    """
    Main Socratic branching function.

    1. Classifies the learner's answer.
    2. Branches on the classification:
       - matched-failure / matched-success: acknowledge the historical parallel.
         matched-success also resets nudge_count to 0.
         matched-failure keeps / may increment nudge_count (treated as a nudge).
       - partial: targeted Socratic follow-up; increment nudge_count.
       - novel: acknowledge + evaluate + redirect; increment nudge_count.
    3. Records the answer in state["answer_history"].
    4. If nudge_count >= 2 after branching: deliver stuck_fallback and reset.
    5. Yields response tokens via SSE (AsyncGenerator[str, None]).

    Requirements: 5.2, 5.3, 5.4, 5.5, 5.6
    """

    # --- 0. Helpless detection (before classification) ---
    if is_helpless_answer(answer):
        logger.debug("on_user_answer: helpless answer detected — giving starter hint")
        # Record as helpless but do NOT increment nudge_count
        state["answer_history"].append(
            {
                "episode_id": episode.id,
                "answer": answer,
                "classification": "helpless",
            }
        )
        # Stream a starter hint
        prompt = _build_starter_hint_prompt(episode)
        try:
            response = await litellm.acompletion(
                model=_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a warm Socratic tutor. The learner hasn't "
                            "attempted an answer yet. Help them get started."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.5,
                max_tokens=250,
            )
            async for chunk in response:
                token: str = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as exc:  # noqa: BLE001
            logger.warning("on_user_answer: starter hint LLM failed (%s); yielding static hint", exc)
            yield (
                f"No worries — this is a tough one! Let me help you get started.\n\n"
                f"{episode.problem_posed}\n\n"
                f"Think about it this way: what's the simplest approach you can "
                f"imagine to solve this problem? Even a guess is a great starting point."
            )
        return  # Don't increment nudge_count — this wasn't a real attempt

    # --- 1. Classify ---
    label: ClassificationLabel = await classify_answer(answer, episode)
    logger.debug("on_user_answer: classification=%r", label)

    # --- 2. Record answer in history ---
    state["answer_history"].append(
        {
            "episode_id": episode.id,
            "answer": answer,
            "classification": label,
        }
    )

    # --- 3. Branch on classification ---
    if label in ("matched-failure", "matched-success"):
        # Acknowledge the historical parallel (Requirement 5.3)
        branch_prompt = _build_acknowledge_parallel_prompt(label, episode, answer)
        if label == "matched-success":
            # Success: learner figured it out — reset nudge counter
            state["nudge_count"] = 0
        else:
            # Failure match: treat as a productive nudge
            state["nudge_count"] = state.get("nudge_count", 0) + 1

    elif label == "partial":
        # Targeted follow-up (Requirement 5.4)
        branch_prompt = _build_targeted_followup_prompt(episode, answer)
        state["nudge_count"] = state.get("nudge_count", 0) + 1

    else:  # novel
        # Acknowledge + evaluate + redirect (Requirement 5.5)
        branch_prompt = _build_novel_redirect_prompt(episode, answer)
        state["nudge_count"] = state.get("nudge_count", 0) + 1

    # --- 4. Check stuck threshold BEFORE streaming branch response ---
    if state["nudge_count"] >= 2:
        # Deliver stuck fallback (Requirement 5.6) — four-section structured response
        logger.debug(
            "on_user_answer: nudge_count=%d >= 2, delivering stuck_fallback",
            state["nudge_count"],
        )
        state["nudge_count"] = 0  # reset after fallback
        async for token in stuck_fallback(episode):
            yield token
        return  # fallback consumed; branch response is superseded

    # --- 5. Stream the branch response ---
    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Socratic tutor. Be concise, warm, and encouraging. "
                        "Never reveal the full solution directly."
                    ),
                },
                {"role": "user", "content": branch_prompt},
            ],
            stream=True,
            temperature=0.5,
            max_tokens=300,
        )
        async for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as exc:  # noqa: BLE001
        logger.warning("on_user_answer: LLM streaming failed (%s); yielding minimal response", exc)
        if label == "matched-success":
            yield f"Well done! You've arrived at the historical solution for '{episode.concept}'."
        elif label == "matched-failure":
            yield (
                f"Interesting — that's exactly the approach that was tried historically "
                f"for '{episode.concept}', and it led to a failure. Think about why."
            )
        elif label == "partial":
            yield f"You're on the right track for '{episode.concept}'. Can you elaborate further?"
        else:
            yield (
                f"That's a novel approach! Historically, '{episode.concept}' was tackled "
                "differently. What do you think the original researchers tried?"
            )
