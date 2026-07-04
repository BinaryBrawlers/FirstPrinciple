"""Core data models for MindForge.

All models are plain Python dataclasses — no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Concept:
    """Atomic unit of knowledge extracted from a source document."""

    id: str
    name: str
    definition: str
    difficulty: str  # "beginner" | "intermediate" | "advanced"
    prerequisites: List[str] = field(default_factory=list)  # list of concept IDs

    # Source attribution
    source_title: str = ""
    source_author: str = ""
    source_year: int = 0
    source_url: str = ""

    mastery_percentage: float = 0.0


@dataclass
class Relationship:
    """Directed edge between two concepts in the knowledge graph."""

    from_concept: str  # concept ID
    to_concept: str    # concept ID
    relationship_type: str = "prerequisite"


@dataclass
class LearnerProfile:
    """Persistent profile tracking a learner's progress and feedback history."""

    learner_id: str
    mastered_concepts: List[str] = field(default_factory=list)
    feedback_weights: Dict[str, float] = field(default_factory=dict)  # concept_id → weight
    session_history: List[str] = field(default_factory=list)          # session IDs
    overall_mastery: float = 0.0

    def apply_feedback(self, concept_id: str, correct: bool) -> None:
        """Adjust feedback weight for a concept by ±1.0.

        Initialises the weight to 0.0 if the concept has no prior entry,
        then increments by +1.0 for a correct answer or decrements by -1.0
        for an incorrect one.
        """
        current = self.feedback_weights.get(concept_id, 0.0)
        self.feedback_weights[concept_id] = current + (1.0 if correct else -1.0)


@dataclass
class Session:
    """A single interaction session between a learner and MindForge."""

    session_id: str
    learner_id: str
    mode: str    # "teach" | "interview" | "ingest"
    status: str  # "active" | "incomplete" | "completed"
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    concepts_covered: List[str] = field(default_factory=list)


@dataclass
class ConceptStep:
    """A single step in an ordered learning path."""

    concept_id: str
    title: str
    estimated_hours: float
    prerequisites: List[str] = field(default_factory=list)
    difficulty: str = "beginner"  # "beginner" | "intermediate" | "advanced"
    order: int = 0


@dataclass
class LearningPath:
    """An ordered sequence of concept steps generated for a learner."""

    learner_id: str
    goal: str
    concepts: List[ConceptStep] = field(default_factory=list)
    total_concepts: int = 0
    estimated_hours: float = 0.0
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InterviewResults:
    """Aggregated results from a completed interview session."""

    score: float          # percentage 0–100
    total_questions: int
    correct_count: int
    weak_concepts: List[str] = field(default_factory=list)
