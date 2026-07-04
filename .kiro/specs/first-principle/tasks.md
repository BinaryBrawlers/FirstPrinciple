# Implementation Plan: FirstPrinciple

## Overview

Build the FirstPrinciple multi-agent learning system bottom-up: environment and data models first, then the memory isolation layer, then each agent, then LangGraph orchestration, then the FastAPI backend, and finally the React+TypeScript frontend, Docker deployment, and README. Each layer is tested incrementally before the next is introduced.

## Tasks

- [ ] 1. Project scaffold and environment configuration
  - [ ] 1.1 Create directory structure and dependency files
    - Create `backend/` and `frontend/` root directories
    - Write `backend/requirements.txt` pinning: `cognee`, `langgraph`, `fastapi`, `uvicorn`, `sse-starlette`, `wikipedia-api`, `arxiv`, `youtube-transcript-api`, `python-dotenv`
    - Write `backend/config.py` that reads `MISTRAL_API_KEY` and sets `COGNEE_SKIP_CONNECTION_TEST=true` via `os.environ` before any cognee import; call `cognee.config.set_llm_provider("mistral")` and `cognee.config.set_embedding_provider("fastembed")`
    - Write `.env.example` documenting required env vars
    - Initialise `frontend/` with `package.json` (Vite + React + TypeScript), pinning `eventsource-parser`
    - _Requirements: 13.1, 13.2_

- [ ] 2. Data models and schema
  - [ ] 2.1 Implement `HistoricalEpisode`, `TutorState`, and `TraitStatement` in `backend/models/schemas.py`
    - Write `Outcome`, `SourceConfidence` enums and `HistoricalEpisode` dataclass with all required fields
    - Write `TutorState` TypedDict with all nine fields
    - Write `TraitStatement` dataclass
    - _Requirements: 1.1, 3.1_

  - [ ]* 2.2 Write property test for HistoricalEpisode schema completeness
    - **Property 1: HistoricalEpisode schema completeness**
    - Generate random `HistoricalEpisode` instances via Hypothesis; assert all required fields are present and each value matches its declared type or enum
    - **Validates: Requirements 1.1**

  - [ ]* 2.3 Write unit tests for TutorState defaults and field types
    - Verify each field accepts its specified type; verify `nudge_count` initialises to 0
    - _Requirements: 1.1_


- [ ] 3. Memory isolation layer (`MemoryGateway`)
  - [ ] 3.1 Implement `AgentRole` enum and `MemoryGateway` class in `backend/memory/gateway.py`
    - Define `_TRACK_A_WRITERS` and `_TRACK_B_WRITERS` sets
    - Implement `add_data_points`, `remember`, `forget`, `improve` with role checks; raise `MemoryAccessError` on violations
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [ ]* 3.2 Write property test for Track A write isolation
    - **Property 4: Track A write isolation**
    - For each role in `{TEACHER, INTERVIEWER, TRAIT_SYNTHESIS}`, construct a `MemoryGateway` and call `add_data_points()`; assert `MemoryAccessError` is raised every time
    - **Validates: Requirements 14.1, 14.4**

  - [ ]* 3.3 Write property test for Track B write isolation
    - **Property 5: Track B write isolation**
    - For each role in `{TEACHER, INTERVIEWER}`, call `remember/forget/improve` on a `user_*` graph name; assert `MemoryAccessError` is always raised
    - **Validates: Requirements 14.2, 14.3, 3.3, 3.4**

  - [ ]* 3.4 Write property test for Track B graph naming invariant
    - **Property 6: Track B graph naming invariant**
    - Generate arbitrary user ID strings; assert the graph name produced equals `"user_" + user_id` exactly
    - **Validates: Requirements 3.1**


- [ ] 4. Hand-authored seed episodes
  - [ ] 4.1 Write OS Memory Management seed episodes in `backend/memory/seed.py`
    - Author six `HistoricalEpisode` dicts in order: base+limit registers → segmentation → external fragmentation (failure) → paging → page tables → MMU/TLB
    - Populate `requires`, `concurrent_with`, `source_confidence`, and `published_date` for each
    - _Requirements: 10.1_

  - [ ] 4.2 Write Deep Learning seed episodes in `backend/memory/seed.py`
    - Author nine `HistoricalEpisode` dicts in order: perceptron → XOR failure → MLP → backpropagation → CNN (concurrent: RNN) → vanishing gradient (failure) → LSTM → attention mechanism → Transformer
    - Mark CNN and RNN as `concurrent_with` each other; mark XOR failure and vanishing gradient with `outcome: failure`
    - _Requirements: 10.2_

  - [ ] 4.3 Implement `seed_tracks_if_absent()` lifespan hook in `backend/memory/seed.py`
    - Call `cognee.recall(graph_name="content_track", query=...)` to check existence before writing
    - Call `gateway.add_data_points(episodes, temporal_cognify=True)` then `cognee.consolidate_entity_descriptions_pipeline()` for each topic if absent
    - _Requirements: 10.3, 2.4, 2.5_

  - [ ]* 4.4 Write property test for seed idempotency
    - **Property 16: Seed topics are complete and non-duplicated**
    - Call `seed_tracks_if_absent()` twice; assert Track A contains exactly the expected episode IDs with no duplicates
    - **Validates: Requirements 10.1, 10.2, 10.3**

  - [ ]* 4.5 Write unit test for seed episode narrative sort
    - Construct a small mixed-date episode list; assert `narrative_sort()` produces a topologically valid order consistent with `requires` edges
    - _Requirements: 4.5_


- [ ] 5. Ingestion Agent
  - [ ] 5.1 Implement topic decomposition and Wikipedia skeleton pass in `backend/agents/ingestion.py`
    - Write `decompose_topic(topic)` returning `list[str]` subtopics
    - Write `fetch_wikipedia(subtopic)` returning `list[HistoricalEpisode]` with `source_confidence=named_reference`
    - _Requirements: 4.1, 4.2_

  - [ ] 5.2 Implement arXiv detail pass and YouTube transcript fetch
    - Write `fetch_arxiv(subtopic)` restricting to abstract and introduction; tag `source_confidence=cited_source`
    - Write `fetch_youtube(video_ids)` using `youtube-transcript-api`; tag `source_confidence=named_reference`
    - _Requirements: 4.2, 4.3, 4.4_

  - [ ] 5.3 Implement `tag_source_confidence()`, `narrative_sort()`, and retry-with-backoff
    - Write `tag_source_confidence(episodes)` to apply the three-tier tagging rules
    - Write `narrative_sort(episodes)` respecting `requires` edges; use `published_date` as a tiebreaker only
    - Write `fetch_with_retry(fetch_fn, max_attempts=3)` using `asyncio.sleep(2**attempt)` — no tenacity
    - _Requirements: 4.4, 4.5, 4.8_

  - [ ]* 5.4 Write property test for exponential backoff timing
    - **Property 9: Exponential backoff timing**
    - Mock the clock; for attempt k ∈ {0,1,2}, assert the sleep duration equals 2^k seconds and that `tenacity` is not imported anywhere in the module
    - **Validates: Requirements 4.8**

  - [ ] 5.5 Implement `add_data_points` call, self-check recall, and reasoned fallback
    - After `narrative_sort`, call `gateway.add_data_points(episodes, temporal_cognify=True)`
    - Call `cognee.consolidate_entity_descriptions_pipeline()` after ingestion
    - Write `self_check_recall(topic)` returning bool; retry up to 3 total on failure
    - Write `reasoned_fallback(subtopics)` that generates `reasoned`-tier episodes via Mistral; guard with a pre-check to avoid duplication
    - _Requirements: 2.4, 2.5, 4.6, 4.7_

  - [ ]* 5.6 Write property test for reasoned episode idempotency
    - **Property 3: Reasoned episodes are idempotent per topic**
    - Run `IngestionAgent.run(topic)` twice on a stubbed Track A; assert the set of `reasoned`-tier episode IDs after run 2 equals the set after run 1
    - **Validates: Requirements 1.4, 10.3**

  - [ ]* 5.7 Write property test for retry exhaustion leading to reasoned fallback
    - **Property 8: Ingestion retry exhaustion leads to reasoned fallback**
    - Stub `self_check_recall` to always return False; assert at least one `reasoned`-tier episode exists in Track A after the agent completes
    - **Validates: Requirements 4.6, 4.7**

- [ ] 6. Checkpoint — ingestion layer
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 7. Teacher Agent
  - [ ] 7.1 Implement answer classifier in `backend/agents/teacher.py`
    - Write `classify_answer(answer, episode)` returning exactly one of `{matched-failure, matched-success, partial, novel}`
    - _Requirements: 5.2_

  - [ ]* 7.2 Write property test for answer classification exhaustiveness
    - **Property 10: Answer classification is exhaustive and mutually exclusive**
    - Generate arbitrary (answer_str, episode) pairs; assert classifier returns exactly one value from the four-label set, never null or a combination
    - **Validates: Requirements 5.2**

  - [ ] 7.3 Implement Socratic branching logic and stuck fallback
    - Implement `on_user_answer(state, answer)` branching on classification: acknowledge parallel, targeted follow-up, novel redirect
    - Implement `stuck_fallback(episode)` producing all four required sections (Problem framing, Solution hint, Engineering Insight, Historical note) when `nudge_count >= 2`
    - Reset `nudge_count` to 0 after fallback delivery
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.4 Write property test for two-nudge stuck fallback completeness
    - **Property 11: Two-nudge stuck fallback contains all four parts**
    - Generate episodes with `nudge_count=2`; assert fallback response string contains all four sections
    - **Validates: Requirements 5.6**

  - [ ] 7.5 Implement `select_next_episode` with Track B recall
    - Call `cognee.recall(graph_name=f"user_{user_id}_traits")` before computing next episode
    - Traverse `requires` edges for mandatory candidates; filter resolved episodes from `answer_history`
    - Cross-reference Track B misconceptions; prefer `concurrent_with` siblings aligned with active weak points
    - _Requirements: 5.7, 5.8_

  - [ ]* 7.6 Write property test for `recall()` preceding episode selection
    - **Property 12: recall() precedes episode selection**
    - Instrument `cognee.recall`; assert it is called before `select_next_episode` returns for any valid state
    - **Validates: Requirements 5.7**

  - [ ]* 7.7 Write property test for dependency traversal respecting only `requires` edges
    - **Property 2: Dependency traversal respects only `requires` edges**
    - Generate random episode DAGs; assert `select_next_episode` order is consistent with a topological sort over `requires` only; adding/removing `concurrent_with` edges SHALL NOT change traversal order
    - **Validates: Requirements 1.2, 1.3**

  - [ ] 7.8 Implement Digest Mode and `@cognee.agent_memory` decorator, SSE streaming
    - Write `on_digest(state, transcript)` summarising against Track A without advancing episode position
    - Apply `@cognee.agent_memory(save_traces=True, with_session_memory=True)` to `teacher_agent`
    - Implement `async for token in teacher_agent(...)` SSE streaming via `AsyncGenerator`
    - _Requirements: 5.9, 5.10, 5.11, 6.1, 6.2_

  - [ ]* 7.9 Write property test for digest mode not advancing episode position
    - **Property 13: Digest mode does not advance episode position**
    - For any transcript input, assert `state["current_episode"]` is identical before and after the digest call
    - **Validates: Requirements 6.1, 6.2**


- [ ] 8. Interviewer Agent
  - [ ] 8.1 Implement session start: Track B recall with `feedback_influence` and question selection in `backend/agents/interviewer.py`
    - Call `cognee.recall(graph_name=f"user_{user_id}_traits", query_params={"feedback_influence": True})`
    - Write `select_questions(weak_points, track_a_failure_episodes)` producing single-concept questions drawn preferentially from `outcome: failure` episodes
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 8.2 Implement inline grading, confidence prompt, and confidently-wrong penalty
    - Write `grade_answer(question, answer)` and `request_confidence_score()` within the same turn
    - Write `compute_penalty(grade, confidence_score)` applying `HARSH_MULTIPLIER` when `confidence ∈ {4,5}` and grade is wrong
    - _Requirements: 7.4, 7.5, 7.6_

  - [ ]* 8.3 Write property test for confidently-wrong penalty being strictly harsher
    - **Property 14: Confidently-wrong penalty is strictly harsher**
    - For any wrong answer, assert `penalty(wrong, confidence=4) > penalty(wrong, confidence=1)` and `penalty(wrong, confidence=5) > penalty(wrong, confidence=2)`
    - **Validates: Requirements 7.6**

  - [ ] 8.4 Implement session-end misconception diff, `@cognee.agent_memory` decorator, and SSE streaming
    - Write `compute_misconception_diff(trait_snapshot, current_track_b)` and stream the diff summary at session end
    - Apply `@cognee.agent_memory(save_traces=True, with_session_memory=True)` to `interviewer_agent`
    - Implement SSE token streaming via `AsyncGenerator`
    - _Requirements: 7.7, 7.8, 7.9_

- [ ] 9. Checkpoint — agent layer
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 10. Trait Synthesis Agent
  - [ ] 10.1 Implement trace reading, evidence grouping, and multi-evidence rule in `backend/agents/trait_synthesis.py`
    - Call `cognee.recall_agent_memory_traces(state["session_id"])` and group by concept via `group_traces_by_concept`
    - Skip any concept with fewer than two evidence signals (multi-evidence rule)
    - _Requirements: 8.1, 8.3, 3.6_

  - [ ]* 10.2 Write property test for multi-evidence requirement before `remember()`
    - **Property 7: Multi-evidence requirement before remember()**
    - Generate trace maps with varying evidence counts; assert `gateway.remember()` is never called when evidence count < 2
    - **Validates: Requirements 3.6, 8.3**

  - [ ] 10.3 Implement `remember`, `improve`, and `forget` dispatch logic
    - For resolved evidence: call `gateway.forget(graph_name, existing_trait.id)` if trait exists
    - For updated evidence: call `gateway.improve(graph_name, existing_trait.id, feedback=...)` via `add_feedback()`
    - For new evidence (≥2 signals, no existing trait): call `gateway.remember(graph_name, trait)`
    - _Requirements: 3.2, 3.7, 3.8, 8.4, 8.5_

  - [ ] 10.4 Wire Trait Synthesis Agent to the three trigger points
    - Trigger after Teacher session end (Chain 2: `teacher_node → trait_synthesis_node`)
    - Trigger after Interviewer session end (Chain 2: `interviewer_node → trait_synthesis_node`)
    - Trigger on mode switch (application layer emits `mode_switch` event invoking Chain 2)
    - _Requirements: 8.2_

  - [ ]* 10.5 Write property test for trait synthesis triggers only at defined points
    - **Property 15: Trait Synthesis triggers only at the three defined points**
    - Instrument the agent; simulate full session lifecycles; assert invocation count equals the number of trigger events fired and zero otherwise
    - **Validates: Requirements 8.2**


- [ ] 11. LangGraph orchestration
  - [ ] 11.1 Implement `ingestion_node`, `teacher_node`, `interviewer_node`, and `trait_synthesis_node` wrappers in `backend/chains/langgraph_chains.py`
    - Each node accepts and returns `TutorState`; delegate to the corresponding agent function
    - _Requirements: 9.2_

  - [ ] 11.2 Define Chain 1 (`ingestion_node → teacher_node`) and Chain 2 (`teacher_node/interviewer_node → trait_synthesis_node`) with `RetryPolicy`
    - Apply `RetryPolicy(max_attempts=3, backoff_factor=2.0)` to every node in both chains
    - Set entry points and edges exactly as specified; do NOT define an Orchestrator agent
    - _Requirements: 9.1, 9.3, 9.4_

  - [ ]* 11.3 Write unit tests for LangGraph chain dispatch logic
    - Assert Chain 1 is compiled when topic is absent from Track A
    - Assert Chain 2 is compiled on session-end signal
    - Assert mid-session turns bypass LangGraph and invoke agents directly
    - _Requirements: 9.1_


- [ ] 12. FastAPI backend routes
  - [ ] 12.1 Implement `POST /chat` SSE endpoint in `backend/routers/chat.py`
    - Write `load_or_create_state(user_id, session_id, mode, topic)` and the `token_generator` async generator
    - Return `EventSourceResponse`; begin streaming before full response is generated
    - Do NOT use WebSockets
    - _Requirements: 11.1, 11.3, 11.4_

  - [ ] 12.2 Implement `POST /ingest` endpoint and session lifecycle endpoints in `backend/routers/ingest.py`
    - Use `BackgroundTasks.add_task(ingestion_agent.run, topic)` and return `{"status": "queued", "topic": topic}`
    - Add session start/end lifecycle endpoints consumed by the frontend
    - _Requirements: 11.2_

  - [ ] 12.3 Wire routers, lifespan hook, and `main.py`
    - Register all routers in `backend/main.py`
    - Call `seed_tracks_if_absent()` in the FastAPI lifespan startup hook
    - _Requirements: 10.3, 11.1, 11.2_

- [ ] 13. Checkpoint — backend complete
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 14. React+TypeScript frontend — core structure
  - [ ] 14.1 Create TypeScript types and API client in `frontend/src/types/api.ts` and `frontend/src/api/client.ts`
    - Define `ChatMessage` interface matching the backend SSE contract
    - Write `fetch` wrappers for `POST /chat` and `POST /ingest`
    - _Requirements: 12.1_

  - [ ] 14.2 Implement `useSSEChat` hook in `frontend/src/hooks/`
    - `useSSEChat`: stream tokens from `POST /chat` by reading the response body with `TextDecoder`; parse `data:` lines
    - _Requirements: 12.2_

  - [ ] 14.3 Implement full-width chat layout in `frontend/src/App.tsx`
    - Render `ChatPanel` as the sole full-width component
    - _Requirements: 12.1_


- [ ] 15. React+TypeScript frontend — ChatPanel
  - [ ] 15.1 Implement `ChatPanel`, `MessageList`, and `ModeToggle` components
    - `ChatPanel` consumes `useSSEChat`; renders tokens as they arrive
    - `ModeToggle` switches between `teacher` and `interviewer` mode; on switch emits mode-switch event to backend session lifecycle endpoint
    - `MessageList` renders each message with an episode-match tag where `episode_id` is present in the SSE payload
    - _Requirements: 12.2, 12.3, 12.4_

  - [ ]* 15.2 Write unit tests for `ModeToggle` state flip
    - Assert `TutorState.mode` (reflected in component state) toggles correctly between `teacher` and `interviewer`
    - _Requirements: 12.3_


- [ ] 17. Checkpoint — frontend complete
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 18. Docker and docker-compose deployment
  - [ ] 18.1 Write `backend/Dockerfile` and `backend/requirements.txt` production freeze
    - Use `python:3.11-slim`; copy and install requirements; expose port 8000; set `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`
    - _Requirements: 13.1_

  - [ ] 18.2 Write `frontend/Dockerfile` with multi-stage build and `nginx.conf`
    - Stage 1: `node:20-alpine`, `npm ci`, `npm run build`
    - Stage 2: `nginx:alpine`; copy `/app/dist`; write `nginx.conf` proxying `/chat` and `/ingest` to `http://backend:8000`
    - _Requirements: 13.1_

  - [ ] 18.3 Write `docker-compose.yml` wiring backend and frontend services
    - Define `backend` (port 8000) with `MISTRAL_API_KEY` and `COGNEE_SKIP_CONNECTION_TEST=true` env vars and `cognee_data` volume
    - Define `frontend` (port 3000:80) with `depends_on: backend`
    - _Requirements: 13.1, 13.2, 13.3_


- [ ] 19. README
  - [ ] 19.1 Write `README.md` with cognee API mapping, agent count justification, episode-based-history design rationale, and known limitations
    - Document each cognee API used (`remember`, `recall`, `improve`, `forget`, `temporal_cognify`, `@cognee.agent_memory`, `consolidate_entity_descriptions_pipeline`)
    - Explain why four agents: separation of ingestion, Socratic teaching, adversarial testing, and trait abstraction
    - Explain why episode-based history: reproducibility, historical fidelity, dependency-aware traversal
    - List known limitations: single-user demo scale, no auth, Mistral rate limits, reasoned-tier episode quality
    - _Requirements: 13.1_

- [ ] 20. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 6, 9, 13, 17, and 20 ensure incremental validation
- Property tests (Hypothesis for Python, Vitest for TypeScript) validate universal correctness invariants
- Unit tests validate specific examples and edge cases
- The `MemoryGateway` isolation layer (task 3) must be complete before any agent is implemented
- Seed episodes (task 4) must be complete before the backend lifespan hook is wired (task 12.3)
- All correctness properties from the design document are covered by property-based test sub-tasks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "4.1", "4.2"] },
    { "id": 4, "tasks": ["4.3", "5.1"] },
    { "id": 5, "tasks": ["4.4", "4.5", "5.2"] },
    { "id": 6, "tasks": ["5.3"] },
    { "id": 7, "tasks": ["5.4", "5.5", "7.1"] },
    { "id": 8, "tasks": ["5.6", "5.7", "7.2", "7.3", "8.1"] },
    { "id": 9, "tasks": ["7.4", "7.5", "8.2", "10.1"] },
    { "id": 10, "tasks": ["7.6", "7.7", "7.8", "8.3", "8.4", "10.2"] },
    { "id": 11, "tasks": ["7.9", "10.3", "11.1"] },
    { "id": 12, "tasks": ["10.4", "10.5", "11.2"] },
    { "id": 13, "tasks": ["11.3", "12.1"] },
    { "id": 14, "tasks": ["12.2", "12.3", "14.1"] },
    { "id": 15, "tasks": ["14.2", "14.3"] },
    { "id": 16, "tasks": ["15.1"] },
    { "id": 17, "tasks": ["15.2", "18.1", "18.2"] },
    { "id": 18, "tasks": ["18.3"] },
    { "id": 19, "tasks": ["19.1"] }
  ]
}
```
