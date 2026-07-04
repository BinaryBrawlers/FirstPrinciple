"""Agent message protocol for MindForge.

Defines the structured request/response types used for inter-agent communication
through the Orchestrator, as well as the result types returned by each agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core request / response envelope
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """Structured message sent to any agent via the Orchestrator.

    Attributes:
        intent:      What the caller wants — "ingest" | "learn" | "teach" |
                     "answer" | "test" | "interview_answer" | "finish_interview" |
                     "status" | "reset"
        learner_id:  Identifier of the learner initiating the request.
        session_id:  Active session identifier (empty string when not yet assigned).
        dataset:     Cognee dataset scope for the request (e.g. "deep_learning").
        payload:     Arbitrary key/value data specific to the intent.
        timestamp:   UTC time the request was created.
    """

    intent: str
    learner_id: str
    session_id: str
    dataset: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResponse:
    """Structured message returned by any agent to the Orchestrator.

    Attributes:
        status:    Outcome — "success" | "error" | "partial"
        data:      Result payload; content varies by agent and intent.
        errors:    List of human-readable error strings (empty on success).
        agent_id:  Identifier of the agent that produced the response
                   (e.g. "knowledge_curator", "teacher", "interviewer").
        timestamp: UTC time the response was produced.
    """

    status: str   # "success" | "error" | "partial"
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Knowledge Curator result types
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Result returned by KnowledgeCuratorAgent.ingest_content().

    Attributes:
        concepts_count:       Number of concepts extracted.
        relationships_count:  Number of prerequisite relationships found.
        concepts:             List of concept IDs (or names) that were extracted.
        source_url:           The URL or path of the ingested source, if any.
        dataset:              Cognee dataset the content was stored into.
    """

    concepts_count: int
    relationships_count: int
    concepts: List[str] = field(default_factory=list)
    source_url: str = ""
    dataset: str = ""


# ---------------------------------------------------------------------------
# Teacher Agent result types
# ---------------------------------------------------------------------------

@dataclass
class TeachingResponse:
    """Result returned by TeacherAgent.teach_concept().

    Attributes:
        concept_id:  The concept being taught.
        explanation: The explanation chunk presented to the learner.
        question:    The Socratic probing question following the explanation.
        source:      Source attribution string (e.g. "Deep Learning Book, Goodfellow et al.").
        turn:        Dialogue turn number within the current concept (1-indexed).
    """

    concept_id: str
    explanation: str
    question: str
    source: str = ""
    turn: int = 1


@dataclass
class EvaluationResult:
    """Result returned by TeacherAgent.evaluate_response().

    Attributes:
        score:    Numeric score for the response (0.0–1.0).
        feedback: Textual feedback shown to the learner.
        advance:  True when the learner has demonstrated mastery and should
                  move to the next concept.
        level:    Understanding level — "poor" | "partial" | "good".
    """

    score: float
    feedback: str
    advance: bool
    level: str = "partial"   # "poor" | "partial" | "good"


# ---------------------------------------------------------------------------
# Interviewer Agent result types
# ---------------------------------------------------------------------------

@dataclass
class InterviewSession:
    """Result returned by InterviewerAgent.start_interview().

    Attributes:
        session_id:        Active session identifier.
        total_questions:   Total number of questions planned for this interview.
        current_question:  1-indexed number of the current question.
        question_id:       Unique identifier for the first question.
        question_text:     The question text to display to the learner.
        concept_id:        The concept being tested by this question.
        difficulty:        Difficulty of the first question — "easy" | "medium" | "hard".
        correct_answer:    The expected correct answer (used internally for evaluation).
    """

    session_id: str
    total_questions: int
    current_question: int
    question_id: str
    question_text: str
    concept_id: str
    difficulty: str = "medium"   # "easy" | "medium" | "hard"
    correct_answer: str = ""


@dataclass
class AnswerEvaluation:
    """Result returned by InterviewerAgent.evaluate_answer().

    Attributes:
        correct:          Whether the learner's answer was correct.
        feedback:         Explanation or correction shown to the learner.
        next_difficulty:  Adapted difficulty for the next question — "easy" | "medium" | "hard".
        next_question_id: ID of the next question, if any (empty string when interview ends).
        next_question_text: Text of the next question (empty string when interview ends).
        next_concept_id:  Concept targeted by the next question.
    """

    correct: bool
    feedback: str
    next_difficulty: str = "medium"   # "easy" | "medium" | "hard"
    next_question_id: str = ""
    next_question_text: str = ""
    next_concept_id: str = ""


# ---------------------------------------------------------------------------
# Orchestrator / session result types
# ---------------------------------------------------------------------------

@dataclass
class SessionStatus:
    """Result returned by OrchestratorAgent.get_session_status().

    Attributes:
        session_id:        The session being queried.
        learner_id:        The learner who owns the session.
        mode:              Current mode — "teach" | "interview" | "ingest" | "idle".
        status:            Lifecycle state — "active" | "incomplete" | "completed".
        concepts_covered:  List of concept IDs touched in this session.
        current_concept:   The concept the learner is currently working on (if any).
        created_at:        UTC time the session was created.
        completed_at:      UTC time the session completed (None if still active).
    """

    session_id: str
    learner_id: str
    mode: str    # "teach" | "interview" | "ingest" | "idle"
    status: str  # "active" | "incomplete" | "completed"
    concepts_covered: List[str] = field(default_factory=list)
    current_concept: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
