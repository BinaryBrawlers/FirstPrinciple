# Requirements Document

## Introduction

FirstPrinciple is a multi-agent learning system that teaches users by making them reinvent concepts the way history actually did. Rather than presenting polished knowledge, the system poses the same problems real researchers faced — in the same rough order, including dead ends and failures — and adapts adversarially to each user's specific tracked weak points. The system persists cross-session learner trait memory and never regenerates work already cached.

The system is composed of four agents (Ingestion, Teacher, Interviewer, Trait Synthesis), two memory tracks in cognee (content graph and per-user trait graph), a LangGraph orchestration layer with two chains, a FastAPI backend with SSE streaming, and a React+TypeScript single-panel chat UI.

---

## Glossary

- **System**: The FirstPrinciple application as a whole.
- **Ingestion Agent**: The agent responsible for decomposing topics, fetching source material, tagging confidence, and writing HistoricalEpisode nodes into Track A.
- **Teacher Agent**: The Socratic interactive agent that guides a user through historical problem sequences using Track A episodes and Track B misconceptions.
- **Interviewer Agent**: The adversarial testing agent that generates single-concept questions targeting Track B weak points, grades answers, and penalises confident wrong answers.
- **Trait Synthesis Agent**: The sole writer of Track B; reads agent-memory traces and abstracts them into durable learner-trait statements.
- **Track A**: The shared `content_track` cognee knowledge graph containing HistoricalEpisode nodes, ingested from Wikipedia, arXiv, and video transcripts. Shared across all users.
- **Track B**: The per-user `user_{id}_traits` cognee knowledge graph containing learner-trait statements about misconceptions, preferences, pace, example affinity, and confidence calibration.
- **HistoricalEpisode**: A structured data point capturing a historical problem-solving event, including its outcome, dependencies, concurrent events, and source confidence level.
- **source_confidence**: One of three tiers — `cited_source`, `named_reference`, or `reasoned` — assigned to every HistoricalEpisode.
- **LangGraph Chain**: A directed sequence of LangGraph nodes executed with a shared `TutorState` TypedDict and a `RetryPolicy`.
- **TutorState**: The shared TypedDict passed between LangGraph nodes containing session state.
- **SSE**: Server-Sent Events — the streaming protocol used by Teacher and Interviewer to push token-by-token responses to the frontend.
- **feedback_influence**: A weighting applied during Track B recall that prioritises weak points flagged by the Interviewer.
- **cognee API**: The memory-graph library providing `remember()`, `recall()`, `improve()`, `forget()`, `temporal_cognify`, `@cognee.agent_memory`, `consolidate_entity_descriptions_pipeline()`, and `visualize_graph()`.

---

## Requirements

### Requirement 1 — HistoricalEpisode Schema

**User Story:** As a system developer, I want every piece of ingested knowledge to conform to a strict schema, so that all agents can reliably traverse dependency and concurrency edges without runtime schema errors.

#### Acceptance Criteria

1. THE System SHALL represent every ingested knowledge unit as a `HistoricalEpisode` containing the fields: `id` (string), `concept` (string), `problem_posed` (string), `attempted_solution` (string), `outcome` (one of: `success`, `failure`, `partial`), `why` (string), `requires` (list of string episode IDs), `concurrent_with` (list of string episode IDs), `source_confidence` (one of: `cited_source`, `named_reference`, `reasoned`), `source` (optional string), and `published_date` (optional date).
2. THE System SHALL treat the `requires` field as the sole dependency ontology between HistoricalEpisode nodes, such that episode traversal order is determined exclusively by `requires` edges.
3. THE System SHALL treat the `concurrent_with` field as a non-ordering annotation indicating episodes that occurred in parallel with no implied prerequisite relationship.
4. IF a `HistoricalEpisode` has `source_confidence` equal to `reasoned`, THEN the System SHALL generate that episode exactly once per topic, cache the result in Track A, and never regenerate it for any individual learner.

---

### Requirement 2 — Track A: Content Knowledge Graph

**User Story:** As a learner, I want the system to draw on a shared, historically accurate knowledge graph so that every user benefits from the same high-quality ingested content without redundant re-ingestion.

#### Acceptance Criteria

1. THE System SHALL maintain a single shared cognee knowledge graph named `content_track` (Track A) accessible to all users.
2. THE System SHALL store all HistoricalEpisode nodes exclusively in Track A.
3. THE Ingestion Agent SHALL be the sole writer of new HistoricalEpisode nodes into Track A.
4. WHEN the Ingestion Agent adds episodes to Track A, THE Ingestion Agent SHALL call `cognee.add_data_points()` with `temporal_cognify=True` to preserve temporal ordering.
5. WHEN the Ingestion Agent has completed a multi-source ingestion pass for a topic, THE Ingestion Agent SHALL call `consolidate_entity_descriptions_pipeline()` to merge duplicate entity descriptions across sources.

---

### Requirement 3 — Track B: Per-User Learner Trait Graph

**User Story:** As a returning learner, I want the system to remember how I specifically learn — my misconceptions, pace, and confidence patterns — so that future sessions are adapted to my actual weaknesses.

#### Acceptance Criteria

1. THE System SHALL maintain a separate cognee knowledge graph per user, named `user_{id}_traits` (Track B), isolated from all other users' trait graphs.
2. THE Trait Synthesis Agent SHALL be the sole writer of Track B for any user.
3. THE Teacher Agent SHALL NOT write directly to Track B.
4. THE Interviewer Agent SHALL NOT write directly to Track B.
5. THE System SHALL store Track B content exclusively as abstracted learner-trait statements (covering misconceptions, preferences, pace, example affinity, and confidence calibration) and SHALL NOT store raw interaction logs in Track B.
6. WHEN the Trait Synthesis Agent writes a new trait to Track B, THE Trait Synthesis Agent SHALL require at least two corroborating evidence signals from agent-memory traces before persisting the trait via `remember()`.
7. WHEN the Trait Synthesis Agent confirms a previously tracked misconception is resolved, THE Trait Synthesis Agent SHALL remove it from Track B via `forget()`.
8. WHEN the Trait Synthesis Agent updates confidence scores for an existing trait, THE Trait Synthesis Agent SHALL call `improve()` via `add_feedback()` rather than overwriting the trait.

---

### Requirement 4 — Ingestion Agent

**User Story:** As a learner, I want topics to be ingested from multiple authoritative sources in historical narrative order so that the episodes I encounter reflect how ideas actually developed.

#### Acceptance Criteria

1. WHEN a new topic is requested for ingestion, THE Ingestion Agent SHALL decompose the topic into constituent subtopics before fetching any source material.
2. THE Ingestion Agent SHALL perform a Wikipedia skeleton pass to establish a base episode set, followed by an arXiv detail pass (restricting arXiv fetches to abstract and introduction sections only).
3. WHERE a topic has associated video content, THE Ingestion Agent SHALL use `youtube-transcript-api` to extract and ingest transcript content as an additional source.
4. THE Ingestion Agent SHALL assign a `source_confidence` tier (`cited_source`, `named_reference`, or `reasoned`) to every HistoricalEpisode before writing it to Track A.
5. THE Ingestion Agent SHALL sort episodes into narrative order using published dates as a sanity-check signal only, not as the primary ordering criterion.
6. WHEN a self-check `recall()` after ingestion fails to surface the expected episodes, THE Ingestion Agent SHALL retry the ingestion, up to a maximum of two additional iterations (three total attempts).
7. IF all three ingestion attempts fail to produce recall-verifiable episodes, THEN THE Ingestion Agent SHALL fall back to generating `reasoned`-tier episodes for the affected subtopics.
8. WHEN a source fetch fails transiently, THE Ingestion Agent SHALL retry that fetch up to three times using exponential backoff between attempts, without using the `tenacity` library.

---

### Requirement 5 — Teacher Agent: Socratic Interactive Mode

**User Story:** As a learner, I want the Teacher to guide me through historical problem sequences using Socratic dialogue so that I construct understanding by reasoning through the same problems original researchers faced.

#### Acceptance Criteria

1. WHEN a learning session begins for a topic, THE Teacher Agent SHALL present the `problem_posed` field of the first relevant HistoricalEpisode to the user before revealing any solution.
2. WHEN the user submits an answer, THE Teacher Agent SHALL classify it into exactly one category: `matched-failure`, `matched-success`, `partial`, or `novel`.
3. WHEN the classification is `matched-failure` or `matched-success`, THE Teacher Agent SHALL branch its response to acknowledge the historical parallel and continue accordingly.
4. WHEN the classification is `partial`, THE Teacher Agent SHALL provide a targeted follow-up prompt rather than revealing the full solution.
5. WHEN the classification is `novel`, THE Teacher Agent SHALL acknowledge the novel approach, evaluate its historical merit, and redirect toward the canonical historical thread.
6. WHEN a user has been stuck on the same episode for two consecutive nudges without progress, THE Teacher Agent SHALL deliver a structured fallback response covering: the Problem framing, a Solution hint, the relevant Engineering Insight, and a Historical note.
7. BEFORE selecting the next episode to present, THE Teacher Agent SHALL call `recall()` on Track B to retrieve the user's active prerequisite misconceptions and factor them into episode selection.
8. THE Teacher Agent SHALL determine the next episode to present by traversing `requires` and `concurrent_with` edges from the current episode in Track A.
9. WHEN a user successfully resolves an episode, THE Teacher Agent SHALL record the resolution in cognee agent memory via `remember()` as a plaintext note.
10. THE Teacher Agent SHALL be decorated with `@cognee.agent_memory(save_traces=True, with_session_memory=True)` so that all interactions are traceable by the Trait Synthesis Agent.
11. THE Teacher Agent SHALL stream all responses to the client token-by-token via SSE.

---

### Requirement 6 — Teacher Agent: Digest Mode

**User Story:** As a learner, I want to submit a video transcript for the Teacher to process so that existing content I have already consumed can be reflected in my learning path.

#### Acceptance Criteria

1. WHEN the Teacher Agent receives a video transcript as input, THE Teacher Agent SHALL switch to digest mode, summarising the transcript's key episodes against Track A rather than entering Socratic dialogue.
2. WHILE in digest mode, THE Teacher Agent SHALL NOT pose Socratic questions and SHALL NOT advance the learner's episode position in Track A.

---

### Requirement 7 — Interviewer Agent

**User Story:** As a learner, I want to be adversarially tested against my own tracked weak points so that I can identify and correct gaps in my understanding.

#### Acceptance Criteria

1. WHEN an interview session begins, THE Interviewer Agent SHALL call `recall()` on Track B with `feedback_influence` weighting to retrieve the user's current weak points.
2. THE Interviewer Agent SHALL generate questions that each target exactly one concept per question.
3. THE Interviewer Agent SHALL draw questions preferentially from `outcome: failure` episodes in Track A.
4. WHEN a user answers a question, THE Interviewer Agent SHALL ask the user to self-report a confidence score between 1 and 5 inclusive.
5. THE Interviewer Agent SHALL grade each answer inline, within the same response turn.
6. WHEN a user provides a confidently wrong answer (confidence score 4 or 5 with an incorrect response), THE Interviewer Agent SHALL apply a harsher penalty than for a low-confidence wrong answer.
7. WHEN an interview session ends, THE Interviewer Agent SHALL compute a graph diff of misconceptions cleared during the session and surface it to the user.
8. THE Interviewer Agent SHALL be decorated with `@cognee.agent_memory(save_traces=True, with_session_memory=True)` so that all interactions are traceable by the Trait Synthesis Agent.
9. THE Interviewer Agent SHALL stream all responses to the client token-by-token via SSE.

---

### Requirement 8 — Trait Synthesis Agent

**User Story:** As a learner, I want my learning traits to be synthesised from real evidence across sessions so that the system's model of me improves over time without drifting from false signals.

#### Acceptance Criteria

1. THE Trait Synthesis Agent SHALL read exclusively from agent-memory traces produced by the Teacher Agent and Interviewer Agent; it SHALL NOT read from raw session logs.
2. THE Trait Synthesis Agent SHALL run at exactly three trigger points: after a Teacher session ends, after an Interviewer session ends, and when the LangGraph `teacher_node → trait_synthesis_node` chain completes.
3. WHEN the Trait Synthesis Agent abstracts a new learner trait, THE Trait Synthesis Agent SHALL require corroboration from at least two independent trace signals before writing the trait to Track B via `remember()`.
4. WHEN the Trait Synthesis Agent detects that a previously stored misconception has been resolved, THE Trait Synthesis Agent SHALL call `forget()` to remove it from Track B.
5. WHEN the Trait Synthesis Agent updates an existing trait's confidence weight, THE Trait Synthesis Agent SHALL call `improve()` via `add_feedback()`.

---

### Requirement 9 — LangGraph Orchestration

**User Story:** As a system developer, I want LangGraph to orchestrate the agents in well-defined chains with retry semantics so that transient failures do not interrupt the user's learning session.

#### Acceptance Criteria

1. THE System SHALL define exactly two LangGraph chains: Chain 1 (`ingestion_node → teacher_node`) triggered when a requested topic is not present in Track A, and Chain 2 (`teacher_node → trait_synthesis_node` and `interviewer_node → trait_synthesis_node`) triggered at session end.
2. THE System SHALL pass a shared `TutorState` TypedDict as the state object between all nodes in both chains.
3. THE System SHALL apply a `RetryPolicy` with a maximum of three attempts and exponential backoff to every node in both LangGraph chains.
4. THE System SHALL NOT define an Orchestrator agent; coordination SHALL be handled exclusively by LangGraph chain definitions.

---

### Requirement 10 — Seed Topics

**User Story:** As a new user, I want at least two fully hand-authored topic sequences available at launch so that I can start learning before any external ingestion completes.

#### Acceptance Criteria

1. THE System SHALL ship with a hand-authored seed episode sequence for OS memory management covering, in order: base+limit registers, segmentation, external fragmentation (failure outcome), paging, page tables, and MMU/TLB.
2. THE System SHALL ship with a hand-authored seed episode sequence for deep learning covering, in order: perceptron, XOR failure (failure outcome), MLP, backpropagation, CNN and RNN (concurrent), vanishing gradient (failure outcome), LSTM, attention mechanism, and Transformer.
3. THE System SHALL load seed topic episodes into Track A on first startup if those topics are not already present in Track A.

---

### Requirement 11 — FastAPI Backend and SSE Streaming

**User Story:** As a frontend developer, I want the backend to expose well-defined HTTP endpoints with SSE streaming for chat so that the UI can be built without WebSocket complexity.

#### Acceptance Criteria

1. THE System SHALL expose a `POST /chat` endpoint that accepts a user message and mode (`teacher` or `interviewer`) and returns an SSE stream of response tokens.
2. THE System SHALL expose a `POST /ingest` endpoint that accepts a topic name and triggers the Ingestion Agent asynchronously.
3. WHEN a client connects to `POST /chat`, THE System SHALL begin streaming tokens via SSE before the full response is generated.
4. THE System SHALL NOT use WebSockets for any client-facing communication.

---

### Requirement 12 — React+TypeScript Frontend

**User Story:** As a learner, I want a full-width chat interface so that I can interact with the Teacher and Interviewer agents seamlessly.

#### Acceptance Criteria

1. THE System SHALL render a full-width chat panel as the primary interface.
2. THE Chat Panel SHALL display streamed Teacher and Interviewer responses token-by-token as they arrive via SSE.
3. THE Chat Panel SHALL provide a mode toggle allowing the user to switch between Teacher mode and Interviewer mode.
4. THE Chat Panel SHALL annotate each message with an episode-match tag indicating which HistoricalEpisode the message corresponds to, where applicable.

---

### Requirement 13 — Deployment

**User Story:** As a system operator, I want the entire system to run via Docker Compose so that the backend and frontend can be started with a single command on any machine with Docker installed.

#### Acceptance Criteria

1. THE System SHALL provide a `docker-compose.yml` file that defines containerised services for at minimum: the FastAPI backend and the React frontend.
2. THE System SHALL NOT require any host-level dependency installation beyond Docker and docker-compose to start all services.
3. WHEN `docker-compose up` is executed, THE System SHALL start all services and make the frontend accessible on a documented host port.

---

### Requirement 14 — Agent Memory Isolation

**User Story:** As a system developer, I want strict boundaries enforced between agents' write permissions on the two tracks so that Track B never accumulates noisy or unverified data.

#### Acceptance Criteria

1. THE System SHALL enforce at the application layer that only the Ingestion Agent calls `cognee.add_data_points()` targeting Track A.
2. THE System SHALL enforce at the application layer that only the Trait Synthesis Agent calls `remember()`, `forget()`, or `improve()` targeting any Track B graph.
3. WHEN any agent other than the Trait Synthesis Agent attempts to write to Track B, THE System SHALL raise an application-level error and abort the write.
4. WHEN any agent other than the Ingestion Agent attempts to call `add_data_points()` on Track A, THE System SHALL raise an application-level error and abort the write.
