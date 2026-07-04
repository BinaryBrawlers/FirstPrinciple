# Requirements Document: MindForge

## Introduction

MindForge is an agentic AI multi-agent system for personalized teaching and learning, built on top of Cognee (the open-source AI memory platform). The system addresses the challenge of scaling quality 1-on-1 education by providing an AI tutor with persistent memory that remembers learner progress, adapts teaching strategies, and builds on knowledge across sessions.

The system ingests research papers, Wikipedia pages, and textbooks into a knowledge graph, plans personalized learning paths based on concept dependencies, teaches using the Socratic method, and tests knowledge through adaptive interviews. Unlike stateless AI tutors, MindForge maintains context across sessions, remembers weaknesses, and improves its teaching approach over time.

**Target Use Case:** Hackathon MVP demonstrating creative use of all 4 Cognee API calls (remember, recall, improve, forget) for personalized learning at scale.

## Glossary

- **MindForge**: The complete multi-agent AI tutoring system
- **Cognee**: The open-source AI memory platform providing persistent storage and retrieval capabilities
- **Knowledge_Graph**: A structured representation of concepts, their relationships, and dependencies stored in Cognee
- **Learner**: The end user receiving instruction from the system
- **Learner_Profile**: Persistent memory of a learner's progress, weaknesses, and breakthroughs
- **Learning_Path**: An ordered sequence of concepts based on prerequisite dependencies
- **Session**: A single interaction period between the learner and the system
- **Session_Memory**: Temporary memory scoped to a single learning session
- **Permanent_Memory**: Long-term storage that persists across all sessions
- **Socratic_Method**: Teaching approach that presents information in small chunks followed by probing questions
- **Teacher_Mode**: Interactive mode where the system guides learning through dialogue
- **Interviewer_Mode**: Assessment mode where the system tests knowledge and provides feedback
- **Knowledge_Curator_Agent**: Agent responsible for ingesting and organizing educational content
- **Curriculum_Architect_Agent**: Agent responsible for building learning paths from concept dependencies
- **Teacher_Agent**: Agent responsible for conducting Socratic teaching sessions
- **Interviewer_Agent**: Agent responsible for testing knowledge and recording performance
- **Orchestrator**: Coordinator agent that routes user intent and manages mode switching
- **Concept**: An atomic unit of knowledge with prerequisites and relationships
- **Dataset**: A scoped collection of memory items in Cognee organized by topic
- **Feedback_Weight**: A numerical value influencing concept ranking based on learner performance

## Requirements

### Requirement 1: Knowledge Ingestion

**User Story:** As a learner, I want the system to ingest educational content from multiple sources, so that I can learn from research papers, Wikipedia pages, and textbooks.

#### Acceptance Criteria

1. WHEN educational content is provided (research paper, Wikipedia page, or textbook), THE Knowledge_Curator_Agent SHALL store the content in Permanent_Memory using cognee.remember()
2. WHEN content is stored, THE Knowledge_Curator_Agent SHALL extract concepts, prerequisites, and relationships into the Knowledge_Graph
3. WHEN multiple sources are ingested for the same topic, THE Knowledge_Curator_Agent SHALL organize them into a scoped Dataset
4. WHEN content ingestion is requested, THE Knowledge_Curator_Agent SHALL support PDF, markdown, plain text, and URL formats
5. FOR ALL ingested content, THE Knowledge_Curator_Agent SHALL extract metadata including source, author, publication date, and topic domain

### Requirement 2: Concept Dependency Extraction

**User Story:** As a learner, I want the system to understand which concepts must be learned before others, so that I follow a logical learning progression.

#### Acceptance Criteria

1. WHEN content is ingested, THE Knowledge_Curator_Agent SHALL identify prerequisite relationships between concepts
2. WHEN concepts are extracted, THE Knowledge_Curator_Agent SHALL create directed edges in the Knowledge_Graph representing dependencies
3. WHEN a concept has multiple prerequisites, THE Knowledge_Curator_Agent SHALL store all prerequisite relationships
4. THE Knowledge_Graph SHALL represent concept dependencies as a directed acyclic graph (DAG)
5. WHEN prerequisite chains are detected, THE Knowledge_Curator_Agent SHALL preserve transitive relationships

### Requirement 3: Learning Path Generation

**User Story:** As a learner, I want the system to create a personalized learning path, so that I learn concepts in the optimal order based on my current knowledge.

#### Acceptance Criteria

1. WHEN a learner requests to learn a topic, THE Curriculum_Architect_Agent SHALL query the Knowledge_Graph using cognee.recall() to retrieve concept dependencies
2. WHEN concept dependencies are retrieved, THE Curriculum_Architect_Agent SHALL generate a topologically sorted Learning_Path
3. WHEN multiple valid orderings exist, THE Curriculum_Architect_Agent SHALL prioritize foundational concepts first
4. WHEN a learner has existing knowledge, THE Curriculum_Architect_Agent SHALL retrieve the Learner_Profile and skip mastered concepts
5. THE Learning_Path SHALL include concept identifiers, titles, estimated duration, and prerequisite references

### Requirement 4: Socratic Teaching Mode

**User Story:** As a learner, I want the system to teach using the Socratic method, so that I actively engage with concepts rather than passively reading.

#### Acceptance Criteria

1. WHEN Teacher_Mode is activated, THE Teacher_Agent SHALL retrieve the next concept from the Learning_Path using cognee.recall()
2. WHEN a concept is retrieved, THE Teacher_Agent SHALL present a small chunk of information followed by a probing question
3. WHEN the learner answers a question, THE Teacher_Agent SHALL evaluate the response and adapt the next explanation based on understanding level
4. WHEN the learner demonstrates mastery, THE Teacher_Agent SHALL advance to the next concept in the Learning_Path
5. WHEN the learner struggles, THE Teacher_Agent SHALL provide additional explanation and ask clarifying questions
6. THE Teacher_Agent SHALL store all dialogue interactions in Session_Memory using cognee.remember() with session scope

### Requirement 5: Adaptive Interview Mode

**User Story:** As a learner, I want the system to test my knowledge through structured interviews, so that I can identify gaps and receive feedback.

#### Acceptance Criteria

1. WHEN Interviewer_Mode is activated, THE Interviewer_Agent SHALL retrieve concepts from the Learner_Profile marked as weak or untested using cognee.recall()
2. WHEN concepts are retrieved, THE Interviewer_Agent SHALL generate questions targeting those concepts
3. WHEN the learner answers a question, THE Interviewer_Agent SHALL evaluate correctness and provide immediate feedback
4. WHEN the learner answers correctly, THE Interviewer_Agent SHALL increase question difficulty for that concept
5. WHEN the learner answers incorrectly, THE Interviewer_Agent SHALL decrease question difficulty and mark the concept for reinforcement
6. WHEN the interview session completes, THE Interviewer_Agent SHALL calculate a performance score and store results in Session_Memory

### Requirement 6: Persistent Learner Memory

**User Story:** As a learner, I want the system to remember my progress across sessions, so that I can build on previous learning without repetition.

#### Acceptance Criteria

1. THE MindForge SHALL maintain a Learner_Profile in Permanent_Memory containing progress, weaknesses, and breakthroughs
2. WHEN a Session completes, THE Orchestrator SHALL use cognee.improve() to bridge Session_Memory into the Learner_Profile
3. WHEN incorrect answers are recorded, THE Orchestrator SHALL apply negative Feedback_Weights to associated concepts in the Knowledge_Graph
4. WHEN correct answers are recorded, THE Orchestrator SHALL apply positive Feedback_Weights to associated concepts
5. FOR ALL sessions, THE MindForge SHALL persist interaction history, performance metrics, and concept mastery status
6. WHEN a new Session starts, THE Orchestrator SHALL retrieve the Learner_Profile using cognee.recall() to resume from previous progress

### Requirement 7: Memory Enrichment and Feedback Integration

**User Story:** As a learner, I want the system to improve its teaching based on my performance, so that it focuses on areas where I struggle.

#### Acceptance Criteria

1. WHEN a Session completes, THE Orchestrator SHALL invoke cognee.improve() to enrich the Learner_Profile with performance data
2. WHEN feedback is applied, THE Orchestrator SHALL update Feedback_Weights for concepts based on answer correctness
3. WHEN weak areas are identified, THE Curriculum_Architect_Agent SHALL prioritize those concepts in future Learning_Paths
4. WHEN a concept is mastered, THE Curriculum_Architect_Agent SHALL reduce its priority in future sessions
5. THE MindForge SHALL use cognee.improve() to bridge Session_Memory into Permanent_Memory after each teaching or interview session

### Requirement 8: Memory Removal and Dataset Management

**User Story:** As a learner, I want to reset my progress or remove outdated content, so that I can start fresh or update my knowledge base.

#### Acceptance Criteria

1. WHEN a learner requests to reset progress, THE Orchestrator SHALL invoke cognee.forget() to remove the Learner_Profile
2. WHEN a learner requests to remove a topic, THE Knowledge_Curator_Agent SHALL invoke cognee.forget() with the Dataset identifier to remove all content for that topic
3. WHEN outdated content is identified, THE Knowledge_Curator_Agent SHALL invoke cognee.forget() to remove specific items
4. WHERE a learner requests a fresh start, THE Orchestrator SHALL provide options to forget all progress, forget a specific topic, or forget specific sessions
5. WHEN memory is removed, THE MindForge SHALL confirm deletion and update the Knowledge_Graph accordingly

### Requirement 9: Mode Orchestration and Intent Routing

**User Story:** As a learner, I want to seamlessly switch between learning and testing modes, so that I can control my learning experience.

#### Acceptance Criteria

1. WHEN a learner submits a request, THE Orchestrator SHALL determine intent and route to the appropriate agent
2. WHEN a learner requests teaching, THE Orchestrator SHALL activate Teacher_Mode and invoke the Teacher_Agent
3. WHEN a learner requests testing, THE Orchestrator SHALL activate Interviewer_Mode and invoke the Interviewer_Agent
4. WHEN a learner requests content ingestion, THE Orchestrator SHALL invoke the Knowledge_Curator_Agent
5. WHEN a learner requests a learning path, THE Orchestrator SHALL invoke the Curriculum_Architect_Agent
6. THE Orchestrator SHALL maintain Session context and coordinate agent interactions throughout the Session

### Requirement 10: Query Routing and Retrieval Strategies

**User Story:** As a system, I want to optimize memory retrieval based on query type, so that responses are fast and relevant.

#### Acceptance Criteria

1. WHEN retrieving concept definitions, THE MindForge SHALL use cognee.recall() with auto-routing enabled
2. WHEN retrieving Learner_Profile data, THE MindForge SHALL query Permanent_Memory with learner-specific scope
3. WHEN retrieving Session context, THE MindForge SHALL query Session_Memory with session-specific scope
4. WHEN retrieving topic content, THE MindForge SHALL query the Knowledge_Graph with Dataset scope filtering
5. THE MindForge SHALL leverage Cognee's auto-routing strategies to optimize retrieval performance across query types

### Requirement 11: Multi-Agent Communication Protocol

**User Story:** As a system architect, I want agents to communicate through well-defined interfaces, so that the system is maintainable and extensible.

#### Acceptance Criteria

1. THE Orchestrator SHALL expose a unified API for agent communication
2. WHEN an agent completes a task, THE agent SHALL return a structured response containing status, data, and any errors
3. WHEN an agent requires data from another agent, THE agent SHALL request data through the Orchestrator
4. THE MindForge SHALL implement message passing between agents using structured JSON payloads
5. THE MindForge SHALL log all inter-agent communications for debugging and performance monitoring

### Requirement 12: User Interface for Teaching and Testing

**User Story:** As a learner, I want a simple interface to interact with the tutor, so that I can focus on learning without technical complexity.

#### Acceptance Criteria

1. THE MindForge SHALL provide a chat-based interface for Teacher_Mode interactions
2. THE MindForge SHALL provide a quiz interface for Interviewer_Mode interactions
3. WHEN in Teacher_Mode, THE interface SHALL display the current concept, explanation, and question
4. WHEN in Interviewer_Mode, THE interface SHALL display questions, accept answers, and show immediate feedback
5. THE interface SHALL display the current Learning_Path with progress indicators
6. THE interface SHALL provide controls to switch between Teacher_Mode and Interviewer_Mode
7. THE interface SHALL display the Learner_Profile including mastered concepts, weak areas, and overall progress

### Requirement 13: Session Management

**User Story:** As a learner, I want my learning sessions to be tracked and resumable, so that I can learn in multiple sittings.

#### Acceptance Criteria

1. WHEN a learner starts the system, THE Orchestrator SHALL create a new Session with a unique session identifier
2. WHEN a Session is active, THE MindForge SHALL store all interactions in Session_Memory
3. WHEN a learner exits without completing, THE Orchestrator SHALL mark the Session as incomplete and preserve Session_Memory
4. WHEN a learner returns, THE Orchestrator SHALL offer to resume the incomplete Session or start a new Session
5. WHEN a Session is resumed, THE MindForge SHALL retrieve Session_Memory and restore the learning context
6. WHEN a Session completes successfully, THE Orchestrator SHALL use cognee.improve() to persist learnings and mark the Session as complete

### Requirement 14: Performance Tracking and Analytics

**User Story:** As a learner, I want to see my learning progress and performance metrics, so that I can understand my strengths and weaknesses.

#### Acceptance Criteria

1. THE MindForge SHALL track concept mastery percentage for each concept in the Learner_Profile
2. THE MindForge SHALL calculate session performance scores based on interview results
3. THE MindForge SHALL maintain a history of all Sessions with timestamps, duration, and concepts covered
4. THE MindForge SHALL identify weak concepts based on Feedback_Weights and incorrect answers
5. THE MindForge SHALL display progress visualizations showing mastery over time
6. WHEN analytics are requested, THE MindForge SHALL retrieve and aggregate data from the Learner_Profile using cognee.recall()

### Requirement 15: Content Source Attribution

**User Story:** As a learner, I want to know the source of information being taught, so that I can verify accuracy and explore further.

#### Acceptance Criteria

1. WHEN the Teacher_Agent presents information, THE Teacher_Agent SHALL include source attribution (paper title, author, publication)
2. WHEN content is recalled from the Knowledge_Graph, THE MindForge SHALL preserve source metadata
3. THE interface SHALL display source references with clickable links to original content where available
4. WHEN multiple sources contain conflicting information, THE Teacher_Agent SHALL present both perspectives and their sources
5. THE Knowledge_Curator_Agent SHALL validate source URLs and document formats during ingestion

### Requirement 16: Custom Extraction and Indexing

**User Story:** As a system, I want to customize how content is extracted and indexed, so that domain-specific concepts are properly identified.

#### Acceptance Criteria

1. WHERE domain-specific content is ingested, THE Knowledge_Curator_Agent SHALL apply custom extraction prompts to identify technical concepts
2. WHEN papers contain mathematical notation, THE Knowledge_Curator_Agent SHALL preserve LaTeX formatting in the Knowledge_Graph
3. WHEN diagrams or figures are referenced, THE Knowledge_Curator_Agent SHALL extract figure captions and descriptions
4. THE Knowledge_Curator_Agent SHALL use Cognee's custom prompt capabilities for domain-specific extraction
5. THE Knowledge_Curator_Agent SHALL leverage global context indexing for cross-topic concept linking

### Requirement 17: API Layer for External Integration

**User Story:** As a developer, I want a REST API to integrate MindForge with other applications, so that the system can be embedded in learning platforms.

#### Acceptance Criteria

1. THE MindForge SHALL expose a FastAPI REST API with endpoints for all core operations
2. THE API SHALL provide endpoints for content ingestion, learning path generation, teaching sessions, and interview sessions
3. THE API SHALL accept authentication tokens and maintain learner-specific sessions
4. THE API SHALL return structured JSON responses with consistent error handling
5. THE API SHALL provide OpenAPI documentation accessible at /docs
6. THE API SHALL implement rate limiting and request validation for security

### Requirement 18: Error Handling and Graceful Degradation

**User Story:** As a learner, I want the system to handle errors gracefully, so that I can continue learning even when issues occur.

#### Acceptance Criteria

1. WHEN Cognee API calls fail, THE MindForge SHALL log the error and retry up to 3 times with exponential backoff
2. WHEN retrieval returns no results, THE Teacher_Agent SHALL provide a fallback explanation and notify the Orchestrator
3. WHEN an agent fails to respond, THE Orchestrator SHALL timeout after 30 seconds and return an error message to the learner
4. WHEN memory storage fails, THE MindForge SHALL cache interactions locally and retry persistence asynchronously
5. THE MindForge SHALL provide informative error messages to learners without exposing internal system details

### Requirement 19: Hackathon Demo Requirements

**User Story:** As a hackathon participant, I want to demonstrate all required Cognee API calls in a compelling demo, so that judges can evaluate the project effectively.

#### Acceptance Criteria

1. THE demo SHALL explicitly demonstrate cognee.remember() by ingesting at least 2 educational sources into Permanent_Memory
2. THE demo SHALL explicitly demonstrate cognee.recall() by querying the Knowledge_Graph and Learner_Profile
3. THE demo SHALL explicitly demonstrate cognee.improve() by bridging Session_Memory into the Learner_Profile after a teaching session
4. THE demo SHALL explicitly demonstrate cognee.forget() by removing a concept or resetting learner progress
5. THE demo SHALL complete within 2-3 minutes and clearly map features to judging criteria (Impact, Creativity, Technical Excellence, Best Use of Cognee, UX, Presentation)
6. THE demo SHALL showcase both Teacher_Mode and Interviewer_Mode with realistic learner interactions

### Requirement 20: Minimum Viable Product Scope

**User Story:** As a hackathon participant, I want to deliver a focused MVP, so that the core value proposition is clear and functional within time constraints.

#### Acceptance Criteria

1. THE MVP SHALL support at least one topic domain (e.g., "Introduction to Deep Learning")
2. THE MVP SHALL ingest at least 2 sample papers or articles during setup
3. THE MVP SHALL generate a Learning_Path with at least 5 concepts
4. THE MVP SHALL support at least 3 turns of Socratic dialogue in Teacher_Mode
5. THE MVP SHALL support at least 5 questions in Interviewer_Mode
6. THE MVP SHALL persist Learner_Profile across at least 2 Sessions
7. THE MVP SHALL use Streamlit or Gradio for the user interface with simple chat and quiz components
8. THE MVP SHALL use OpenAI GPT-4o or Anthropic Claude for LLM-based agent reasoning
