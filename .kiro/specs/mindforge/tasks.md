# Implementation Plan: MindForge

## Overview

Incremental implementation of MindForge, a multi-agent AI tutoring system built on Cognee's memory platform. Tasks are ordered so each step builds on the previous: project scaffolding → data models → agents → FastAPI → React UI → demo setup. MVP targets the "Introduction to Deep Learning" domain with 5 concepts, 3-turn Socratic dialogue, and 5-question interviews.

## Tasks

- [x] 1. Project scaffolding and environment setup
  - Create the top-level package directory `mindforge/` with `__init__.py`
  - Create `requirements.txt` pinning: `cognee`, `fastapi`, `uvicorn[standard]`, `openai`, `anthropic`, `tenacity`, `aiosqlite`, `pydantic>=2`, `httpx`, `python-dotenv`
  - Create `mindforge/config.py` reading `COGNEE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LLM_PROVIDER` from environment using `pydantic-settings`
  - Create `.env.example` documenting all required environment variables
  - Scaffold `frontend/` with Vite + React + TypeScript: `npm create vite@latest frontend -- --template react-ts`
  - Install frontend dependencies: `shadcn/ui`, `tailwindcss`, `framer-motion`, `zustand`, `@tanstack/react-query`, `reactflow`, `recharts`, `lucide-react`, `react-router-dom`
  - _Requirements: 17.1, 20.1, 20.8_


- [x] 2. Core data models
  - Implement `Concept`, `Relationship`, `LearnerProfile`, `Session`, `ConceptStep`, and `LearningPath` dataclasses in `mindforge/models.py`
  - Use `@dataclass` with typed fields matching the design schemas
  - Add `LearnerProfile.apply_feedback(concept_id, correct: bool)` that adjusts `feedback_weights` by ±1.0 per call
  - Add `InterviewResults` dataclass with `score`, `total_questions`, `correct_count`, `weak_concepts`
  - _Requirements: 1.2, 2.1, 3.5, 5.6, 6.1, 14.1_


- [x] 3. Agent message protocol
  - Implement `AgentRequest` and `AgentResponse` dataclasses in `mindforge/protocol.py`
  - Fields: `intent`, `learner_id`, `session_id`, `dataset`, `payload`, `timestamp` / `status`, `data`, `errors`, `agent_id`, `timestamp`
  - Add `IngestionResult`, `TeachingResponse`, `EvaluationResult`, `InterviewSession`, `AnswerEvaluation`, `SessionStatus` result types
  - _Requirements: 11.2, 11.4_


- [x] 4. Error handling utilities and local fallback cache
  - [x] 4.1 Implement `mindforge/resilience.py` with `safe_remember`, `safe_recall`, `safe_improve`, `safe_forget` wrappers using `tenacity`
    - Use `stop_after_attempt(3)` and `wait_exponential(multiplier=1, min=1, max=10)` on each wrapper
    - `safe_recall` returns `[]` on failure; `safe_remember` writes to local fallback cache on failure
    - _Requirements: 18.1, 18.2, 18.3_

  - [x] 4.2 Implement `LocalFallbackCache` in `mindforge/cache.py` using `aiosqlite`
    - Schema: `pending_writes(id INTEGER PRIMARY KEY, data TEXT, kwargs TEXT, created_at TEXT)`
    - `async store(data, cognee_kwargs)` — queue a failed write
    - `async flush()` — retry all queued writes against Cognee, delete on success
    - Call `flush()` at every session start
    - _Requirements: 18.4_


- [x] 5. Knowledge Curator Agent
  - [x] 5.1 Implement `KnowledgeCuratorAgent` in `mindforge/agents/knowledge_curator.py`
    - `async ingest_content(content, dataset, source_metadata) -> IngestionResult`
    - `async _extract_concepts(content: str) -> List[Concept]` — LLM call with curator system prompt
    - `async _extract_relationships(concepts) -> List[Relationship]` — LLM call for prerequisite pairs
    - Call `safe_remember(data={content, concepts, relationships, metadata}, dataset=dataset, self_improvement=True)`
    - Support plain text, markdown, and URL input (`httpx.get` for URLs)
    - Include source attribution (`title`, `author`, `year`, `url`) in every `Concept`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 15.1, 16.1_

  - [x] 5.2 Implement dataset management on `KnowledgeCuratorAgent`
    - `async remove_topic(dataset: str)` → `safe_forget(dataset=dataset)`
    - `async remove_item(data_item: str)` → `safe_forget(data_item=data_item)`
    - _Requirements: 8.2, 8.3_


- [x] 6. Curriculum Architect Agent
  - [x] 6.1 Implement `CurriculumArchitectAgent` in `mindforge/agents/curriculum_architect.py`
    - `async generate_learning_path(goal, learner_id, dataset) -> LearningPath`
    - `safe_recall` for concept graph; `safe_recall` for learner profile; filter mastered concepts
    - `_topological_sort(concepts, relationships) -> List[Concept]` — Kahn's algorithm
    - `_apply_feedback_weights(concepts, profile) -> List[Concept]` — re-order ascending by weight (weak first)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.3, 7.4_


- [x] 7. Teacher Agent
  - [x] 7.1 Implement `TeacherAgent` in `mindforge/agents/teacher.py`
    - `async teach_concept(concept_id, learner_id, session_id, dataset) -> TeachingResponse`
    - Retrieve concept via `safe_recall(query_text=f"concept definition for {concept_id}", dataset=dataset)`
    - Retrieve session history via `safe_recall(query_text="recent teaching interactions", session_id=session_id, limit=5)`
    - Generate Socratic explanation + probing question via LLM with teacher system prompt
    - Include source attribution in `TeachingResponse.source`
    - _Requirements: 4.1, 4.2, 15.1, 15.2_

  - [x] 7.2 Implement `TeacherAgent.evaluate_response`
    - `async evaluate_response(learner_response, expected_concept, session_id) -> EvaluationResult`
    - LLM scores response as `"poor" | "partial" | "good"`
    - `safe_remember(data={learner_response, concept, evaluation, understanding_level, timestamp}, session_id=session_id)`
    - Return `EvaluationResult(score, feedback, advance=level=="good")`
    - Fallback: if `recall()` returns `[]`, use LLM-only explanation without graph context
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 18.2_


- [x] 8. Interviewer Agent
  - [x] 8.1 Implement `InterviewerAgent` in `mindforge/agents/interviewer.py`
    - `async start_interview(learner_id, session_id, dataset, num_questions=5) -> InterviewSession`
    - Retrieve weak concepts via `safe_recall(query_text=f"concepts with low mastery for learner {learner_id}", dataset=dataset, limit=num_questions)`
    - Generate first question targeting the weakest concept; adapt difficulty (correct → `"hard"`, incorrect → `"easy"`)
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

  - [x] 8.2 Implement `InterviewerAgent.evaluate_answer` and `finish_interview`
    - `async evaluate_answer(question_id, learner_answer, correct_answer, concept_id, session_id) -> AnswerEvaluation`
    - LLM evaluates correctness; `safe_remember(data={question_id, concept_id, correct, learner_answer, timestamp}, session_id=session_id)`
    - `async finish_interview(session_id, dataset) -> InterviewResults`
    - Recall all answers from session; compute `score = (correct / total) * 100`
    - _Requirements: 5.3, 5.6, 14.2_


- [ ] 9. Orchestrator Agent
  - [ ] 9.1 Implement `OrchestratorAgent` in `mindforge/agents/orchestrator.py`
    - `async start_session(learner_id) -> str` — generate `session_id`, flush cache, retrieve learner profile
    - `async route_request(intent, learner_id, session_id, payload) -> AgentResponse` — dispatch by intent
    - `async end_session(session_id, dataset)` — call `safe_improve(dataset=dataset, session_ids=[session_id])`
    - `async get_session_status(session_id) -> SessionStatus`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 13.1, 13.2, 13.6_

  - [ ] 9.2 Implement reset operations on `OrchestratorAgent`
    - `async reset_learner(learner_id)` — `safe_forget(data_item=f"learner_profile_{learner_id}")`
    - `async reset_all()` — `safe_forget(everything=True)`
    - _Requirements: 8.1, 8.4, 8.5_


- [ ] 10. Checkpoint — all agents complete
  - Manually run a Python script that instantiates all agents and exercises the dispatch chain with mocked Cognee responses; confirm no import or runtime errors; ask the user if questions arise before continuing.


- [ ] 11. FastAPI REST API layer
  - [ ] 11.1 Create `mindforge/api/main.py` with FastAPI app, CORS middleware (allow `http://localhost:5173`), and startup/shutdown hooks
    - Startup: initialize Cognee client from config; flush `LocalFallbackCache`
    - Wire `OrchestratorAgent` as a singleton dependency via `Depends`
    - _Requirements: 17.1, 17.5_

  - [ ] 11.2 Implement content and learning path endpoints in `mindforge/api/routes/content.py`
    - `POST /api/v1/ingest` — `{content, dataset, metadata}` → `KnowledgeCuratorAgent.ingest_content` → `IngestionResult`
    - `POST /api/v1/learning-path` — `{goal, learner_id, dataset}` → `CurriculumArchitectAgent.generate_learning_path` → `LearningPath`
    - _Requirements: 17.2, 3.1, 1.1_

  - [ ] 11.3 Implement teaching session endpoints in `mindforge/api/routes/teaching.py`
    - `POST /api/v1/session/start` — `{learner_id}` → `OrchestratorAgent.start_session` → `{session_id}`
    - `POST /api/v1/teach` — `{concept_id, learner_id, session_id, dataset}` → `TeacherAgent.teach_concept`
    - `POST /api/v1/teach/answer` — `{session_id, answer, concept_id}` → `TeacherAgent.evaluate_response`
    - `POST /api/v1/session/{session_id}/end` — `OrchestratorAgent.end_session` → `{status: "completed"}`
    - _Requirements: 17.2, 4.1, 4.6, 13.3, 13.6_

  - [ ] 11.4 Implement interview and management endpoints in `mindforge/api/routes/interview.py`
    - `POST /api/v1/interview/start` → `InterviewerAgent.start_interview`
    - `POST /api/v1/interview/answer` → `InterviewerAgent.evaluate_answer`
    - `POST /api/v1/interview/finish` → `InterviewerAgent.finish_interview`
    - `GET /api/v1/session/{session_id}/status` → `OrchestratorAgent.get_session_status`
    - `DELETE /api/v1/learner/{learner_id}/reset` → `OrchestratorAgent.reset_learner`
    - _Requirements: 17.2, 5.1, 5.6, 13.1, 8.1_

  - [ ] 11.5 Add Pydantic schemas and error middleware in `mindforge/api/schemas.py`
    - Pydantic v2 request/response models for every endpoint
    - Global exception handler returning `{status, code, message, retry_after}` envelope
    - _Requirements: 17.4, 18.3, 18.5_


- [ ] 12. React frontend — foundation
  - [ ] 12.1 Configure Tailwind CSS and initialize shadcn/ui
    - Run `npx shadcn-ui@latest init` with CSS variables, slate base color, and `src/components/ui/` output path
    - Add components: `Button`, `Card`, `Badge`, `Input`, `Textarea`, `Separator`, `Tooltip`, `Dialog`, `Sheet`, `Progress`, `ScrollArea`, `Tabs`
    - _Requirements: 12.1, 20.7_

  - [ ] 12.2 Implement global state stores in `frontend/src/store/`
    - `sessionStore.ts` (Zustand): `session_id`, `learner_id`, `currentMode` (`"teach" | "interview" | "dashboard" | "ingest"`), `currentDataset`
    - `profileStore.ts` (Zustand): `learnerProfile`, `learningPath`, `masteryMap`, `weakConcepts`
    - _Requirements: 13.1, 12.6_

  - [ ] 12.3 Implement typed API client in `frontend/src/lib/api.ts`
    - `apiFetch<T>(path, init)` wrapper with error envelope parsing
    - Export typed `api` object with helpers for every FastAPI endpoint (see design)
    - Read `VITE_API_URL` from env with `http://localhost:8000` fallback
    - _Requirements: 17.2_

  - [ ] 12.4 Implement `AppShell` layout with animated sidebar in `frontend/src/components/layout/`
    - `Sidebar.tsx`: collapsible icon nav (Book/Teach, Zap/Interview, BarChart/Dashboard, Upload/Ingest), active indicator slides with Framer Motion `layoutId`
    - `TopBar.tsx`: learner name, overall mastery badge, topic selector, dark mode toggle
    - `ModeToggle.tsx`: prominent pill toggle for Teach ↔ Interview with spring animation
    - `AppShell.tsx`: wraps sidebar + main content area with page transition (`AnimatePresence` + `motion.div`)
    - _Requirements: 12.6, 12.7_


- [ ] 13. React frontend — Teacher Mode
  - [ ] 13.1 Implement chat components in `frontend/src/components/chat/`
    - `ChatBubble.tsx`: AI bubble with typewriter streaming effect (`useEffect` character append, ~25ms delay); learner bubble appears instantly; both use rounded card style with avatar
    - `TypingIndicator.tsx`: three dots with Framer Motion stagger bounce animation
    - `SourceBadge.tsx`: small pill showing `"Paper Title (Year)"` — click opens `Popover` with full citation details
    - `ConceptProgressBar.tsx`: sticky header strip showing `"Concept 3 / 9 — Backpropagation"` with thin progress line
    - `AdvanceButton.tsx`: slides up from bottom with spring animation when `advance=true`; "Next Concept →" CTA

  - [ ] 13.2 Implement `TeacherPage.tsx` in `frontend/src/pages/`
    - On mount: call `api.startSession` if no `session_id` in store; call `api.getLearningPath` to populate path
    - Chat message list in `ScrollArea` (auto-scroll to bottom on new message)
    - `ChatInput`: `Textarea` with `Ctrl+Enter` submit, character limit indicator, send button
    - On submit: call `api.submitAnswer`, receive `EvaluationResult`, append feedback bubble, show `AdvanceButton` if `advance=true`
    - On "Next Concept": call `api.teachConcept` for next concept in path, stream response into new `ChatBubble`
    - Show `TypingIndicator` while awaiting API response
    - _Requirements: 4.1, 4.2, 4.3, 12.1, 12.3, 15.3_


- [ ] 14. React frontend — Interviewer Mode
  - [ ] 14.1 Implement quiz components in `frontend/src/components/quiz/`
    - `QuestionCard.tsx`: full-bleed `Card` with question text, `Badge` for difficulty (`Easy`=green, `Medium`=amber, `Hard`=red), concept tag chip
    - `AnswerInput.tsx`: `Textarea` with character count, `Ctrl+Enter` submit shortcut, disabled state while awaiting feedback
    - `FeedbackOverlay.tsx`: full-viewport overlay sliding up from bottom (Framer Motion `y` transition); green ✓ or red ✗ icon, correct answer text, brief explanation; auto-dismisses after 3s or on tap
    - `ProgressRing.tsx`: animated SVG ring showing `answered / total` — strokeDashoffset animates with each answer
    - `ScoreSummaryCard.tsx`: end-of-interview card — animated count-up to final score, per-concept breakdown table, weak areas highlighted in red with "Practice" shortcut buttons

  - [ ] 14.2 Implement `InterviewPage.tsx` in `frontend/src/pages/`
    - On mount: call `api.startInterview`; render first `QuestionCard`
    - On answer submit: call `api.submitInterviewAnswer`; show `FeedbackOverlay`; after dismiss, render next question with `AnimatePresence` slide transition
    - On all questions answered: call `api.finishInterview`; show `ScoreSummaryCard` with animated entrance
    - Update `profileStore` weak concepts after interview completes
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 12.2, 12.4_


- [ ] 15. React frontend — Progress Dashboard
  - [ ] 15.1 Implement `ConceptGraph.tsx` using React Flow in `frontend/src/components/graph/`
    - Fetch learning path and concept relationships; render as directed graph with auto-layout (dagre)
    - Node colors: mastered=green, in-progress=blue, weak=red, not-started=grey
    - Custom `ConceptNode.tsx`: shows concept name, mastery %, difficulty badge
    - On node click: open `Sheet` side drawer with concept definition, session history for that concept, and "Practice" / "Quiz Me" action buttons
    - Edge labels show prerequisite direction; hovering an edge highlights the dependency chain

  - [ ] 15.2 Implement `DashboardPage.tsx` in `frontend/src/pages/`
    - Top row: overall mastery `Progress` bar, session count `Card`, streak `Card`, weak concept count `Badge`
    - `MasteryTrendChart`: Recharts `LineChart` of mastery % per session (fetched from learner profile session history)
    - `ConceptGraph` below the charts, full width
    - `WeakConceptList`: sorted by feedback weight; each row has concept name, weight bar, last attempted date, and "Practice now" button
    - `SessionHistoryTable`: columns Date, Mode, Duration, Score, Concepts Covered — sortable, paginated
    - _Requirements: 12.5, 12.7, 14.1, 14.3, 14.4, 14.5_


- [ ] 16. React frontend — Content Ingestion page
  - Implement `IngestPage.tsx` in `frontend/src/pages/`
  - URL input field: paste a paper URL or Wikipedia link; on submit calls `api.ingestContent` with `dataset` selector
  - Text area tab: paste raw text directly
  - File upload tab: drag-and-drop `.md`, `.txt`, `.pdf` — `FormData` POST to `/api/v1/ingest`
  - On success: show `IngestionResult` card with concept count, relationship count, extracted concept chips
  - Loading state: animated progress bar with status messages ("Fetching content…", "Extracting concepts…", "Building knowledge graph…")
  - _Requirements: 1.1, 1.4, 12.1_


- [ ] 17. Checkpoint — API and UI wired together
  - Start FastAPI (`uvicorn mindforge.api.main:app --reload`) and React dev server (`npm run dev` in `frontend/`)
  - Manually walk through: ingest a URL → view ingestion result → generate learning path → start teacher chat → submit an answer → check dashboard graph updates
  - Ask the user if anything needs adjustment before moving to demo setup.


- [ ] 18. Demo setup — pre-loaded Deep Learning content
  - [ ] 18.1 Create `demo/data/intro_deep_learning.md` with definitions and prerequisite annotations for 5 concepts: `neural_networks`, `activation_functions`, `backpropagation`, `gradient_descent`, `transformers`
    - Each concept section includes: definition, prerequisites, difficulty, and source citation
    - _Requirements: 20.1, 20.3_

  - [ ] 18.2 Create `demo/setup.py` that ingests 2 sources into the `"deep_learning"` Cognee dataset
    - Source 1: Fetch Wikipedia summary for "Backpropagation" via `httpx.get`
    - Source 2: Read and ingest `demo/data/intro_deep_learning.md`
    - Call `KnowledgeCuratorAgent.ingest_content` for each; print `IngestionResult` summary
    - _Requirements: 19.1, 20.1, 20.2_

  - [ ] 18.3 Create `demo/run_demo.py` — scripted walkthrough exercising all 4 Cognee API calls
    - Step 1 (`remember`): ingest 2 sources via `setup.py`
    - Step 2 (`recall`): generate 5-concept learning path and print it
    - Step 3 (`remember` + `recall`): simulate 3-turn Socratic dialogue with hardcoded learner responses
    - Step 4 (`improve`): end session to bridge into permanent memory; print profile diff
    - Step 5 (`forget`): demonstrate concept reset with `memory_only=True`; print confirmation
    - Print wall-clock timing for each step1
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6_


- [ ] 19. Final checkpoint — full system ready
  - Run `demo/setup.py` and confirm 2 sources ingest without error
  - Run `demo/run_demo.py` and confirm all 4 Cognee API calls execute and produce output
  - Open the React app and walk through the full demo flow in the browser
  - Ask the user if questions arise before marking complete.


## Notes

- No test suite is included — focus is on working demo and polished UI.
- All Cognee API calls (`remember`, `recall`, `improve`, `forget`) are covered across Tasks 5–9 and exercised end-to-end in Task 18.
- Checkpoints at Tasks 10, 17, and 19 provide incremental validation gates.
- Local fallback cache (Task 4.2) ensures demo resilience if Cognee has transient latency.
- MVP scope: one domain (`deep_learning` dataset), 2 ingested sources, 5-concept path, 3-turn Socratic dialogue, 5-question interview, profile persistence across sessions.


## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "3"] },
    { "id": 2, "tasks": ["4.1", "4.2"] },
    { "id": 3, "tasks": ["5.1", "5.2", "6.1"] },
    { "id": 4, "tasks": ["7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "8.2"] },
    { "id": 6, "tasks": ["9.1", "9.2"] },
    { "id": 7, "tasks": ["11.1"] },
    { "id": 8, "tasks": ["11.2", "11.3"] },
    { "id": 9, "tasks": ["11.4", "11.5"] },
    { "id": 10, "tasks": ["12.1", "12.2", "12.3"] },
    { "id": 11, "tasks": ["12.4"] },
    { "id": 12, "tasks": ["13.1", "14.1", "15.1"] },
    { "id": 13, "tasks": ["13.2", "14.2", "15.2"] },
    { "id": 14, "tasks": ["16"] },
    { "id": 15, "tasks": ["18.1"] },
    { "id": 16, "tasks": ["18.2"] },
    { "id": 17, "tasks": ["18.3"] }
  ]
}
```
