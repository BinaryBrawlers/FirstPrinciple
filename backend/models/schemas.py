from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class SourceConfidence(str, Enum):
    CITED_SOURCE = "cited_source"
    NAMED_REFERENCE = "named_reference"
    REASONED = "reasoned"


# ---------------------------------------------------------------------------
# HistoricalEpisode
# ---------------------------------------------------------------------------

@dataclass
class HistoricalEpisode:
    """A structured data point capturing a historical problem-solving event.

    Required fields capture the core narrative (id, concept, problem posed,
    attempted solution, outcome, and explanation).  Optional relational fields
    encode the dependency graph (``requires``) and concurrency annotation
    (``concurrent_with``).  ``source_confidence`` defaults to the weakest tier
    so that hand-authored or synthesised episodes never accidentally inherit a
    stronger confidence level.
    """

    id: str
    concept: str
    problem_posed: str
    attempted_solution: str
    outcome: Outcome
    why: str
    requires: list[str] = field(default_factory=list)
    concurrent_with: list[str] = field(default_factory=list)
    source_confidence: SourceConfidence = SourceConfidence.REASONED
    source: Optional[str] = None
    published_date: Optional[date] = None


# ---------------------------------------------------------------------------
# TutorState
# ---------------------------------------------------------------------------

class TutorState(TypedDict):
    """Shared state TypedDict passed between all LangGraph nodes.

    Carries enough context for any node to continue or hand off a session
    without consulting external storage mid-turn.
    """

    user_id: str
    topic: str
    current_episode: str                        # episode ID
    mode: Literal["teacher", "interviewer", "digest"]
    session_id: str
    nudge_count: int                            # consecutive stuck nudges
    answer_history: list[dict]                  # {episode_id, answer, classification}
    trait_snapshot: list[str]                   # Track B trait IDs at session start
    ingest_needed: bool


# ---------------------------------------------------------------------------
# TraitStatement
# ---------------------------------------------------------------------------

@dataclass
class TraitStatement:
    """An abstracted learner-trait statement persisted in Track B.

    Written exclusively by the Trait Synthesis Agent after at least two
    corroborating evidence signals.  ``confidence`` is a float in [0.0, 1.0]
    derived from the number of corroborating evidence items.
    """

    id: str
    user_id: str
    concept: str
    trait_type: Literal[
        "misconception",
        "preference",
        "pace",
        "example_affinity",
        "confidence_calibration",
    ]
    description: str
    confidence: float           # 0.0–1.0 derived from corroborating evidence count
    resolved: bool = False
    evidence_ids: list[str] = field(default_factory=list)   # agent-memory trace IDs
