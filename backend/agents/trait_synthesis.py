"""
Trait Synthesis Agent — Track B sole writer.

IMPORTANT: backend.config must be imported first (before any cognee import)
to apply the LiteLLM patch and set COGNEE_SKIP_CONNECTION_TEST.

Task 10.1: Trace reading, evidence grouping, and multi-evidence rule
  - Calls cognee.recall_agent_memory_traces(session_id) defensively,
    falling back to cognee.recall() when the method is unavailable.
  - Groups traces by concept via group_traces_by_concept().
  - Skips any concept with fewer than 2 evidence signals (multi-evidence rule).

Task 10.3: remember / improve / forget dispatch logic
  - Resolved evidence  → gateway.forget()  (if existing trait exists)
  - Updated evidence   → gateway.improve() via add_feedback() (if trait exists)
  - New evidence (≥2)  → gateway.remember() (no existing trait)

Requirements: 3.2, 3.6, 3.7, 3.8, 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

# --- config MUST come before any cognee import ---
import config  # noqa: F401

import cognee
import litellm

from memory.gateway import AgentRole, MemoryGateway
from models.schemas import TraitStatement

if TYPE_CHECKING:
    from models.schemas import TutorState

logger = logging.getLogger(__name__)

_LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/mistral-small-latest")

# Module-level gateway instance — TRAIT_SYNTHESIS role only
gateway = MemoryGateway(role=AgentRole.TRAIT_SYNTHESIS)


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _get_trace_concept(trace: Any) -> str | None:
    """
    Duck-typed concept extraction from a cognee trace object.

    Checks .concept, .content, .text, and str fallback in that order.
    Returns None when no concept can be determined.
    """
    # Direct concept attribute (preferred)
    concept = getattr(trace, "concept", None)
    if isinstance(concept, str) and concept.strip():
        return concept.strip()

    # Some cognee versions store content/text as the identifying string
    for attr in ("content", "text", "data"):
        value = getattr(trace, attr, None)
        if isinstance(value, str) and value.strip():
            # Use the first sentence / first 60 chars as a concept key
            snippet = value.strip().split(".")[0][:60].strip()
            if snippet:
                return snippet

    # Last resort — str representation, but only if it looks like a meaningful
    # human-readable string (not a Python object repr like "namespace(...)").
    try:
        s = str(trace).strip()
        if (
            s
            and s not in ("<None>", "None", "")
            and len(s) < 120
            # Exclude Python object repr patterns
            and "namespace(" not in s
            and " object at 0x" not in s
            and not s.startswith("<")
        ):
            return s
    except Exception:  # noqa: BLE001
        pass

    return None


def group_traces_by_concept(traces: Any) -> dict[str, list[Any]]:
    """
    Group cognee memory traces by concept string.

    Handles None, empty sequences, and heterogeneous trace objects gracefully.
    Traces whose concept cannot be determined are silently skipped.

    Args:
        traces: Raw output from cognee.recall_agent_memory_traces() or
                cognee.recall() — any type is accepted.

    Returns:
        A dict mapping concept → list of trace objects.
    """
    if traces is None:
        return {}

    if not isinstance(traces, (list, tuple)):
        traces = [traces]

    grouped: dict[str, list[Any]] = {}
    for trace in traces:
        concept = _get_trace_concept(trace)
        if concept is None:
            logger.debug("group_traces_by_concept: skipping trace with no concept: %r", trace)
            continue
        grouped.setdefault(concept, []).append(trace)

    logger.debug(
        "group_traces_by_concept: grouped %d trace(s) into %d concept bucket(s)",
        sum(len(v) for v in grouped.values()),
        len(grouped),
    )
    return grouped


# ---------------------------------------------------------------------------
# Evidence analysis helpers
# ---------------------------------------------------------------------------


def looks_resolved(evidence_list: list[Any]) -> bool:
    """
    Return True when ALL recent signals in evidence_list indicate resolution
    (i.e. the learner consistently answered correctly).

    Resolution signals:
    - Trace has a `resolved` attribute that is truthy, OR
    - Trace has a `classification` attribute equal to "matched-success", OR
    - Trace has a `grade` attribute equal to "correct".

    If no trace carries an explicit resolution signal, returns False
    (conservative — do not forget traits without clear evidence).

    Requirements: 3.7
    """
    if not evidence_list:
        return False

    resolved_count = 0
    for trace in evidence_list:
        # Explicit resolved flag
        if getattr(trace, "resolved", False):
            resolved_count += 1
            continue
        # Teacher-mode: classification label
        classification = getattr(trace, "classification", None)
        if isinstance(classification, str) and classification == "matched-success":
            resolved_count += 1
            continue
        # Interviewer-mode: grading label
        grade = getattr(trace, "grade", None)
        if isinstance(grade, str) and grade == "correct":
            resolved_count += 1
            continue
        # Content/text-based heuristic for string-only traces
        content = getattr(trace, "content", None) or getattr(trace, "text", None)
        if isinstance(content, str):
            content_lower = content.lower()
            if "resolved" in content_lower or "matched-success" in content_lower or "correct" in content_lower:
                resolved_count += 1
                continue

    # All evidence signals must indicate resolution (conservative)
    return resolved_count == len(evidence_list)


# ---------------------------------------------------------------------------
# LLM-backed helpers
# ---------------------------------------------------------------------------

_ABSTRACT_TRAIT_SYSTEM_PROMPT = """\
You are a learner-profiling assistant.

Given a list of evidence signals about a learner's interaction with a concept,
produce a concise one-sentence description of the most salient learner trait
(misconception, preference, pace, example_affinity, or confidence_calibration).

Respond with JSON on a single line with these fields:
  {"trait_type": "<type>", "description": "<one sentence>", "confidence": <0.0-1.0>}

Use only these trait_type values:
  misconception, preference, pace, example_affinity, confidence_calibration

confidence is a float from 0.0 to 1.0 derived from how many consistent signals
there are (more signals = higher confidence; cap at 0.95 for ≤5 signals).
"""


async def abstract_trait(
    concept: str,
    evidence_list: list[Any],
    user_id: str,
) -> TraitStatement:
    """
    Synthesise a TraitStatement from a list of evidence traces for a concept.

    Uses the LLM to infer trait_type, description, and confidence.
    Falls back to a deterministic construction when the LLM is unavailable.

    Requirements: 3.2, 8.5
    """
    # Build evidence summary for the LLM
    evidence_texts: list[str] = []
    for trace in evidence_list:
        for attr in ("content", "text", "data", "description"):
            val = getattr(trace, attr, None)
            if isinstance(val, str) and val.strip():
                evidence_texts.append(val.strip()[:200])
                break
        else:
            # No recognised text attribute — fall back to str(trace) truncated
            evidence_texts.append(str(trace)[:200])

    evidence_summary = "\n".join(f"- {e}" for e in evidence_texts[:10])  # cap at 10

    user_prompt = (
        f"Concept: {concept}\n\n"
        f"Evidence signals ({len(evidence_list)}):\n{evidence_summary}\n\n"
        "Produce the JSON trait description."
    )

    trait_type: str = "misconception"
    description: str = f"Learner shows a pattern related to '{concept}' across multiple interactions."
    confidence: float = min(0.5 + len(evidence_list) * 0.08, 0.95)

    try:
        import json

        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _ABSTRACT_TRAIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        raw: str = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        parsed = json.loads(raw)
        _VALID_TRAIT_TYPES = {
            "misconception", "preference", "pace",
            "example_affinity", "confidence_calibration",
        }
        if parsed.get("trait_type") in _VALID_TRAIT_TYPES:
            trait_type = parsed["trait_type"]
        if parsed.get("description"):
            description = str(parsed["description"])[:500]
        if isinstance(parsed.get("confidence"), (int, float)):
            confidence = max(0.0, min(1.0, float(parsed["confidence"])))

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "abstract_trait: LLM synthesis failed for concept %r (%s); using fallback",
            concept,
            exc,
        )

    # Collect evidence IDs
    evidence_ids: list[str] = []
    for trace in evidence_list:
        tid = getattr(trace, "id", None)
        if isinstance(tid, str) and tid.strip():
            evidence_ids.append(tid.strip())

    return TraitStatement(
        id=str(uuid.uuid4()),
        user_id=user_id,
        concept=concept,
        trait_type=trait_type,  # type: ignore[arg-type]
        description=description,
        confidence=confidence,
        resolved=False,
        evidence_ids=evidence_ids,
    )


_FEEDBACK_SYSTEM_PROMPT = """\
You are a learner-profiling assistant.

Given updated evidence about a learner's interaction with a concept,
produce a concise one-sentence feedback string that describes the most recent
pattern. This feedback will be used to update an existing learner trait.

Respond with ONLY the feedback sentence — no preamble, no JSON.\
"""


async def synthesise_feedback(concept: str, evidence_list: list[Any]) -> str:
    """
    Produce a feedback string for gateway.improve() describing the updated
    evidence pattern for *concept*.

    Falls back to a deterministic template when the LLM is unavailable.

    Requirements: 3.8, 8.4
    """
    evidence_texts: list[str] = []
    for trace in evidence_list:
        for attr in ("content", "text", "data", "description"):
            val = getattr(trace, attr, None)
            if isinstance(val, str) and val.strip():
                evidence_texts.append(val.strip()[:200])
                break

    evidence_summary = "\n".join(f"- {e}" for e in evidence_texts[:5])

    user_prompt = (
        f"Concept: {concept}\n\n"
        f"Recent evidence signals ({len(evidence_list)}):\n{evidence_summary}\n\n"
        "Describe the updated learner pattern in one sentence."
    )

    try:
        response = await litellm.acompletion(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=80,
        )
        feedback: str = (response.choices[0].message.content or "").strip()
        if feedback:
            return feedback
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "synthesise_feedback: LLM call failed for concept %r (%s); using fallback",
            concept,
            exc,
        )

    return (
        f"Updated evidence ({len(evidence_list)} signal(s)) for concept '{concept}' "
        "shows continued interaction patterns."
    )


async def add_feedback(
    graph_name: str,
    trait_id: str,
    feedback: str,
) -> None:
    """
    Wrapper that calls gateway.improve() with structured feedback kwargs.

    Passes feedback as a keyword argument so the gateway → cognee.improve()
    chain can consume it.

    Requirements: 3.8
    """
    await gateway.improve(graph_name, trait_id, feedback=feedback)


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------


async def _recall_traces(session_id: str) -> Any:
    """
    Recall agent memory traces for *session_id*, trying
    ``cognee.recall_agent_memory_traces()`` first and falling back to
    ``cognee.recall()`` when the method is unavailable (Requirements 8.1, 3.6).
    """
    try:
        traces = await cognee.recall_agent_memory_traces(session_id)
        logger.debug(
            "_recall_traces: recall_agent_memory_traces returned %s",
            type(traces).__name__,
        )
        return traces
    except AttributeError:
        logger.debug(
            "_recall_traces: recall_agent_memory_traces not available; "
            "falling back to cognee.recall(query=%r)",
            session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_recall_traces: recall_agent_memory_traces failed (%s); "
            "falling back to cognee.recall()",
            exc,
        )

    # Fallback: cognee.recall() with session_id as query
    try:
        traces = await cognee.recall(query=session_id)
        logger.debug(
            "_recall_traces: fallback cognee.recall() returned %s",
            type(traces).__name__,
        )
        return traces
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_recall_traces: fallback cognee.recall() also failed (%s); returning []",
            exc,
        )
        return []


async def _recall_existing_trait(
    graph_name: str,
    concept: str,
) -> Any | None:
    """
    Query Track B for an existing trait matching *concept*.

    Returns the first matching trait object, or None if not found.
    """
    try:
        result = await cognee.recall(graph_name=graph_name, query=concept)
        if result is None:
            return None
        if isinstance(result, (list, tuple)):
            return result[0] if result else None
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_recall_existing_trait: cognee.recall() failed for concept %r in %r (%s)",
            concept,
            graph_name,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Core trait_synthesis_agent  (Tasks 10.1 + 10.3)
# ---------------------------------------------------------------------------


async def trait_synthesis_agent(state: "TutorState") -> None:
    """
    Trait Synthesis Agent — reads agent-memory traces for the current session,
    groups them by concept, applies the multi-evidence rule (≥2 signals), and
    dispatches remember / improve / forget to Track B accordingly.

    Does NOT return a modified TutorState because it performs only side
    effects (Track B writes).  LangGraph wraps this in trait_synthesis_node
    which returns state unchanged.

    Requirements: 3.2, 3.6, 3.7, 3.8, 8.1, 8.3, 8.4, 8.5
    """
    user_id: str = state["user_id"]
    session_id: str = state["session_id"]
    graph_name: str = f"user_{user_id}_traits"

    logger.info(
        "trait_synthesis_agent: starting for user=%r session=%r",
        user_id,
        session_id,
    )

    # --- Task 10.1: Read traces and group by concept ---
    traces = await _recall_traces(session_id)
    evidence_map: dict[str, list[Any]] = group_traces_by_concept(traces)

    logger.info(
        "trait_synthesis_agent: %d concept bucket(s) after grouping",
        len(evidence_map),
    )

    for concept, evidence_list in evidence_map.items():
        # Multi-evidence rule (Requirements 3.6, 8.3)
        if len(evidence_list) < 2:
            logger.debug(
                "trait_synthesis_agent: skipping concept %r — only %d signal(s) (multi-evidence rule)",
                concept,
                len(evidence_list),
            )
            continue

        # Recall existing trait for this concept from Track B
        existing_trait = await _recall_existing_trait(graph_name, concept)

        # --- Task 10.3: Dispatch logic ---

        if looks_resolved(evidence_list):
            # Evidence shows resolution → forget existing trait if present
            if existing_trait is not None:
                trait_id = getattr(existing_trait, "id", None)
                if trait_id:
                    logger.info(
                        "trait_synthesis_agent: FORGET concept=%r trait_id=%r (resolved)",
                        concept,
                        trait_id,
                    )
                    try:
                        await gateway.forget(graph_name, existing_trait.id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "trait_synthesis_agent: forget() failed for concept=%r (%s)",
                            concept,
                            exc,
                        )
            else:
                logger.debug(
                    "trait_synthesis_agent: concept %r resolved but no existing trait — nothing to forget",
                    concept,
                )

        elif existing_trait is not None:
            # Trait exists but not resolved → improve with updated feedback
            trait_id = getattr(existing_trait, "id", None)
            if trait_id:
                feedback_text = await synthesise_feedback(concept, evidence_list)
                logger.info(
                    "trait_synthesis_agent: IMPROVE concept=%r trait_id=%r",
                    concept,
                    trait_id,
                )
                try:
                    await add_feedback(graph_name, existing_trait.id, feedback=feedback_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "trait_synthesis_agent: improve() failed for concept=%r (%s)",
                        concept,
                        exc,
                    )

        else:
            # No existing trait and not resolved → abstract and remember
            logger.info(
                "trait_synthesis_agent: REMEMBER new trait for concept=%r (%d signals)",
                concept,
                len(evidence_list),
            )
            try:
                trait = await abstract_trait(concept, evidence_list, user_id=user_id)
                await gateway.remember(graph_name, trait)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "trait_synthesis_agent: remember() failed for concept=%r (%s)",
                    concept,
                    exc,
                )

    logger.info(
        "trait_synthesis_agent: completed for user=%r session=%r",
        user_id,
        session_id,
    )
