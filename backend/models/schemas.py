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
    CITED_SOURCE    = "cited_source"
    NAMED_REFERENCE = "named_reference"
    REASONED        = "reasoned"


# ---------------------------------------------------------------------------
# HistoricalEpisode — a single historical problem-solving event (Track A node)
# ---------------------------------------------------------------------------

@dataclass
class HistoricalEpisode:
    id:                 str
    concept:            str
    problem_posed:      str
    attempted_solution: str
    outcome:            Outcome
    why:                str
    requires:           list[str]        = field(default_factory=list)
    concurrent_with:    list[str]        = field(default_factory=list)
    source_confidence:  SourceConfidence = SourceConfidence.REASONED
    source:             Optional[str]    = None
    published_date:     Optional[date]   = None


# ---------------------------------------------------------------------------
# TutorState — shared LangGraph state TypedDict
# ---------------------------------------------------------------------------

class TutorState(TypedDict, total=False):
    user_id:                            str
    topic:                              str
    current_episode:                    str                                    # episode ID
    mode:                               Literal["teacher", "interviewer", "digest"]
    session_id:                         str
    nudge_count:                        int                                    # consecutive stuck nudges
    answer_history:                     list[dict]                             # {episode_id, answer, classification}
    trait_snapshot:                     list[str]                              # Track B trait IDs at session start
    ingest_needed:                      bool
    awaiting_response_to_posed_problem: bool                                   # True only after agent posed a problem


# ---------------------------------------------------------------------------
# TraitStatement — an abstracted learner-trait entry (Track B node)
# ---------------------------------------------------------------------------

@dataclass
class TraitStatement:
    id:          str
    user_id:     str
    concept:     str
    trait_type:  Literal[
        "misconception",
        "preference",
        "pace",
        "example_affinity",
        "confidence_calibration",
    ]
    description: str
    confidence:  float                  # 0.0–1.0 derived from corroborating evidence count
    resolved:    bool      = False
    evidence_ids: list[str] = field(default_factory=list)  # agent-memory trace IDs
