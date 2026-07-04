# Design Document — FirstPrinciple

## Overview

FirstPrinciple is a multi-agent learning system that reconstructs the historical development of technical concepts and guides learners through the same problem sequences original researchers faced. The system comprises four specialised agents, two isolated cognee memory tracks, a LangGraph orchestration layer, a FastAPI SSE backend, and a React+TypeScript single-panel chat frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React+TypeScript Frontend                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        Chat Panel                              │  │
│  │  SSE token stream                                              │  │
│  │  Mode toggle                                                   │  │
│  │  Episode-match tags                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                          │
                POST /chat (SSE)    POST /ingest
                          │
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    LangGraph Orchestration                       │ │
│  │  Chain 1: ingestion_node → teacher_node  (new topic)            │ │
│  │  Chain 2: teacher_node  → trait_synthesis_node  (session end)   │ │
│  │           interviewer_node → trait_synthesis_node (session end) │ │
│  │  Shared TutorState TypedDict  |  RetryPolicy(max=3, exp backoff)│ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────┐ ┌────────────────┐ ┌──────────────────────┐   │
│  │ Ingestion Agent   │ │ Teacher Agent  │ │ Interviewer Agent    │   │
│  │ (Track A writer)  │ │ @agent_memory  │ │ @agent_memory        │   │
│  └──────────────────┘ └────────────────┘ └──────────────────────┘   │
│                                 │                  │                  │
│                        ┌─────────────────────────────────┐           │
│                        │    Trait Synthesis Agent         │           │
│                        │    (Track B sole writer)         │           │
│                        └─────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
                          │                │
              ┌───────────┴──┐    ┌────────┴──────────┐
              │   Track A    │    │      Track B       │
              │content_track │    │ user_{id}_traits   │
              │(shared)      │    │(per-user)          │
              └──────────────┘    └───────────────────┘
                          │
                    cognee (Mistral LLM + fastembed)
```

---

## Components and Interfaces

### Agent Interfaces

| Agent | Input | Output | cognee Writes |
|-------|-------|--------|---------------|
| `IngestionAgent` | `topic: str`, optional `video_ids: list[str]` | `list[HistoricalEpisode]` | Track A via `MemoryGateway(INGESTION)` |
| `TeacherAgent` | `TutorState`, `user_input: str` | `AsyncGenerator[str]` (SSE tokens) | agent-memory traces only |
| `InterviewerAgent` | `TutorState`, `user_input: str` | `AsyncGenerator[str]` (SSE tokens) | agent-memory traces only |
| `TraitSynthesisAgent` | `TutorState` | `None` | Track B via `MemoryGateway(TRAIT_SYNTHESIS)` |

### HTTP API Contract

| Method | Path | Request Body | Response |
|--------|------|-------------|----------|
| `POST` | `/chat` | `{user_id, session_id, message, mode, topic}` | SSE stream of tokens |
| `POST` | `/ingest` | `{topic: str, video_ids?: list[str]}` | `{status: "queued", topic}` |

### MemoryGateway Interface

```python
class MemoryGateway:
    def __init__(self, role: AgentRole): ...
    async def add_data_points(self, data_points, *, temporal_cognify: bool = True): ...
    async def remember(self, graph_name: str, *args, **kwargs): ...
    async def forget(self, graph_name: str, *args, **kwargs): ...
    async def improve(self, graph_name: str, *args, **kwargs): ...
```

All cognee write calls pass through this interface. Roles are checked at call time. `MemoryAccessError` is raised for violations.

### LangGraph Node Signatures

```python
# Chain 1
async def ingestion_node(state: TutorState) -> TutorState: ...
async def teacher_node(state: TutorState) -> TutorState: ...

# Chain 2 additions
async def interviewer_node(state: TutorState) -> TutorState: ...
async def trait_synthesis_node(state: TutorState) -> TutorState: ...
```

All nodes accept and return `TutorState`; side effects (cognee writes, SSE streaming) happen within the node body.

---

## Data Models

### HistoricalEpisode

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL  = "partial"

class SourceConfidence(str, Enum):
    CITED_SOURCE     = "cited_source"
    NAMED_REFERENCE  = "named_reference"
    REASONED         = "reasoned"

@dataclass
class HistoricalEpisode:
    id:                 str
    concept:            str
    problem_posed:      str
    attempted_solution: str
    outcome:            Outcome
    why:                str
    requires:           list[str]           = field(default_factory=list)
    concurrent_with:    list[str]           = field(default_factory=list)
    source_confidence:  SourceConfidence    = SourceConfidence.REASONED
    source:             Optional[str]       = None
    published_date:     Optional[date]      = None
```

The `requires` list carries directed dependency edges; `concurrent_with` is an unordered annotation only. Episode traversal (teacher_node, next-episode selection) reads `requires` edges exclusively for ordering and may suggest `concurrent_with` siblings as optional parallel context.

### TutorState

```python
from typing import TypedDict, Literal

class TutorState(TypedDict):
    user_id:          str
    topic:            str
    current_episode:  str                          # episode ID
    mode:             Literal["teacher", "interviewer", "digest"]
    session_id:       str
    nudge_count:      int                          # consecutive stuck nudges
    answer_history:   list[dict]                   # {episode_id, answer, classification}
    trait_snapshot:   list[str]                    # Track B trait IDs at session start
    ingest_needed:    bool
```

### Track B Trait Statement

```python
@dataclass
class TraitStatement:
    id:              str
    user_id:         str
    concept:         str
    trait_type:      Literal["misconception", "preference", "pace", "example_affinity",
                             "confidence_calibration"]
    description:     str
    confidence:      float          # 0.0–1.0 derived from corroborating evidence count
    resolved:        bool = False
    evidence_ids:    list[str] = field(default_factory=list)   # agent-memory trace IDs
```

---

## Component Design

### 1. Memory Isolation Layer

All cognee write operations are gated by a thin access-control module. Every agent call to a cognee write API goes through `MemoryGateway`, which checks the caller's registered role before dispatching.

```python
# memory/gateway.py

from enum import Enum
from typing import Callable
import cognee

class AgentRole(str, Enum):
    INGESTION        = "ingestion"
    TEACHER          = "teacher"
    INTERVIEWER      = "interviewer"
    TRAIT_SYNTHESIS  = "trait_synthesis"

_TRACK_A_WRITERS = {AgentRole.INGESTION}
_TRACK_B_WRITERS = {AgentRole.TRAIT_SYNTHESIS}

class MemoryAccessError(RuntimeError):
    pass

class MemoryGateway:
    def __init__(self, role: AgentRole):
        self._role = role

    # --- Track A writes ---
    async def add_data_points(self, data_points, *, temporal_cognify: bool = True):
        if self._role not in _TRACK_A_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track A"
            )
        await cognee.add_data_points(data_points, temporal_cognify=temporal_cognify)

    # --- Track B writes ---
    async def remember(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.remember(*args, graph_name=graph_name, **kwargs)

    async def forget(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.forget(*args, graph_name=graph_name, **kwargs)

    async def improve(self, graph_name: str, *args, **kwargs):
        self._assert_track_b_writer(graph_name)
        await cognee.improve(*args, graph_name=graph_name, **kwargs)

    def _assert_track_b_writer(self, graph_name: str):
        if not graph_name.startswith("user_"):
            return   # not a Track B graph; no restriction
        if self._role not in _TRACK_B_WRITERS:
            raise MemoryAccessError(
                f"Role {self._role} is not permitted to write to Track B ({graph_name})"
            )
```

All four agents instantiate `MemoryGateway(role=AgentRole.<THEIR_ROLE>)` at construction time and use it exclusively for cognee writes.

---

### 2. Ingestion Agent

```
topic
  └─ decompose_topic()        → list[subtopic]
  └─ for each subtopic:
       fetch_wikipedia()       → list[HistoricalEpisode] (source_confidence=named_reference)
       fetch_arxiv()           → list[HistoricalEpisode] (abstract+intro only, cited_source)
       fetch_youtube()         → list[HistoricalEpisode] (if video_ids present, named_reference)
  └─ tag_source_confidence()
  └─ narrative_sort()          → ordered list[HistoricalEpisode]
  └─ gateway.add_data_points() (temporal_cognify=True)
  └─ consolidate_entity_descriptions_pipeline()
  └─ self_check_recall()       → bool
       if fail: retry (up to 3 total) with exponential backoff (no tenacity)
       if all fail: reasoned_fallback()
```

**Retry without tenacity:**

```python
import asyncio

async def fetch_with_retry(fetch_fn: Callable, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return await fetch_fn()
        except TransientFetchError:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)   # 1s, 2s, 4s
```

**Reasoned fallback:** When all `max_attempts` recall checks fail, the agent uses the Mistral LLM to synthesise `reasoned`-tier episodes for unverified subtopics. Reasoned episodes are written once and cached; a guard checks `recall()` for existing reasoned episodes before synthesis to prevent duplication.

---

### 3. Teacher Agent

The Teacher Agent operates in two modes selected by the `TutorState.mode` field.

#### Socratic Mode

```
on_user_answer(state, answer):
  classification = classify_answer(answer, current_episode)  # matched-failure | matched-success | partial | novel
  if classification in (matched-failure, matched-success):
      response = acknowledge_historical_parallel(classification, episode)
      if classification == matched-success:
          note = await gateway.remember(...)   # resolution note in agent memory
          state.nudge_count = 0
          next_ep = select_next_episode(state)
      else:
          next_ep = current_episode
  elif classification == partial:
      response = targeted_followup(episode)
      state.nudge_count += 1
  else:  # novel
      response = acknowledge_novel_and_redirect(episode)
      state.nudge_count += 1

  if state.nudge_count >= 2:
      response = stuck_fallback(episode)   # Problem / Solution hint / Engineering Insight / Historical note
      state.nudge_count = 0

  traits = await cognee.recall(graph_name=f"user_{state.user_id}_traits")
  next_ep = select_next_episode(state, traits)
  yield sse_stream(response)
```

**Episode selection** (`select_next_episode`):
1. Pull `requires` edges from the current episode in Track A — these are the mandatory next candidates.
2. Filter out episodes the user has already resolved (from `answer_history`).
3. Cross-reference Track B misconceptions to prefer episodes that address active weak points.
4. Among `concurrent_with` siblings, suggest the one most aligned with current misconceptions.

#### Digest Mode

```
on_digest(state, transcript):
  summary = summarise_against_track_a(transcript)
  yield sse_stream(summary)
  # episode position NOT updated; no Socratic questions issued
```

**Decorator:**
```python
@cognee.agent_memory(save_traces=True, with_session_memory=True)
async def teacher_agent(state: TutorState, user_input: str) -> AsyncGenerator[str, None]:
    ...
```

---

### 4. Interviewer Agent

```
on_session_start(state):
  weak_points = await cognee.recall(
      graph_name=f"user_{state.user_id}_traits",
      query_params={"feedback_influence": True}
  )
  questions = select_questions(weak_points, track_a_failure_episodes)

on_answer(question, answer):
  grade    = grade_answer(question, answer)
  prompt   = request_confidence_score()  # 1-5
  yield sse_stream(inline_grade_and_confidence_prompt)

on_confidence_received(grade, confidence_score):
  penalty = compute_penalty(grade, confidence_score)
  # confidently wrong: confidence in {4,5} AND grade==wrong → penalty *= HARSH_MULTIPLIER
  yield sse_stream(feedback_with_penalty)

on_session_end(state):
  diff = compute_misconception_diff(state.trait_snapshot, current_track_b)
  yield sse_stream(session_diff_summary)
```

**Decorator:**
```python
@cognee.agent_memory(save_traces=True, with_session_memory=True)
async def interviewer_agent(state: TutorState, user_input: str) -> AsyncGenerator[str, None]:
    ...
```

---

### 5. Trait Synthesis Agent

Triggered at exactly three points:
- After a Teacher session ends (Chain 2: `teacher_node → trait_synthesis_node`)
- After an Interviewer session ends (Chain 2: `interviewer_node → trait_synthesis_node`)
- On mode switch (application layer emits a `mode_switch` event that triggers Chain 2)

```python
async def trait_synthesis_agent(state: TutorState):
    traces = await cognee.recall_agent_memory_traces(state.session_id)

    # Group evidence by concept
    evidence_map: dict[str, list[Trace]] = group_traces_by_concept(traces)

    for concept, evidence_list in evidence_map.items():
        if len(evidence_list) < 2:
            continue   # multi-evidence rule: skip single-signal observations

        existing_trait = await cognee.recall(
            graph_name=f"user_{state.user_id}_traits",
            query=concept
        )

        if looks_resolved(evidence_list):
            if existing_trait:
                await gateway.forget(f"user_{state.user_id}_traits", existing_trait.id)
        elif existing_trait:
            await gateway.improve(
                f"user_{state.user_id}_traits",
                existing_trait.id,
                feedback=synthesise_feedback(evidence_list)
            )
        else:
            trait = abstract_trait(concept, evidence_list)
            await gateway.remember(f"user_{state.user_id}_traits", trait)
```

The Trait Synthesis Agent never reads raw session logs — it reads only from cognee agent-memory traces surfaced by `recall_agent_memory_traces()`.

---

### 6. LangGraph Orchestration

```python
from langgraph.graph import StateGraph
from langgraph.pregel import RetryPolicy

retry = RetryPolicy(max_attempts=3, backoff_factor=2.0)

# --- Chain 1: topic not in Track A ---
chain1 = StateGraph(TutorState)
chain1.add_node("ingestion_node", ingestion_agent,  retry=retry)
chain1.add_node("teacher_node",   teacher_agent,    retry=retry)
chain1.add_edge("ingestion_node", "teacher_node")
chain1.set_entry_point("ingestion_node")

# --- Chain 2: session end write-back ---
chain2 = StateGraph(TutorState)
chain2.add_node("teacher_node",           teacher_agent,          retry=retry)
chain2.add_node("interviewer_node",       interviewer_agent,      retry=retry)
chain2.add_node("trait_synthesis_node",   trait_synthesis_agent,  retry=retry)
chain2.add_edge("teacher_node",     "trait_synthesis_node")
chain2.add_edge("interviewer_node", "trait_synthesis_node")
chain2.set_entry_point("teacher_node")  # or "interviewer_node" per request
```

Chain dispatch logic in the API layer:
- `POST /chat` with a topic not found in Track A → compile and invoke Chain 1.
- On session-end signal or mode-switch → compile and invoke Chain 2.
- Mid-session turns (existing topic, same mode) → invoke the appropriate agent directly without LangGraph overhead.

---

### 7. FastAPI Backend

```
backend/
├── main.py              # FastAPI app, lifespan (seed loader)
├── routers/
│   ├── chat.py          # POST /chat  → SSE stream
│   └── ingest.py        # POST /ingest → async task
├── agents/
│   ├── ingestion.py
│   ├── teacher.py
│   ├── interviewer.py
│   └── trait_synthesis.py
├── memory/
│   ├── gateway.py       # MemoryGateway (isolation layer)
│   └── seed.py          # seed_tracks_if_absent()
├── chains/
│   └── langgraph_chains.py
├── models/
│   └── schemas.py       # HistoricalEpisode, TutorState, TraitStatement
└── config.py            # env vars, cognee init
```

#### POST /chat

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    async def token_generator():
        state = load_or_create_state(req.user_id, req.session_id, req.mode, req.topic)
        if req.mode == "teacher":
            async for token in teacher_agent(state, req.message):
                yield {"data": token}
        else:
            async for token in interviewer_agent(state, req.message):
                yield {"data": token}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(token_generator())
```

#### POST /ingest

```python
@router.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingestion_agent.run, req.topic)
    return {"status": "queued", "topic": req.topic}
```

---

### 8. React+TypeScript Frontend

```
frontend/src/
├── App.tsx                   # full-width chat layout
├── components/
│   └── ChatPanel/
│       ├── ChatPanel.tsx     # SSE consumer, mode toggle
│       ├── MessageList.tsx   # renders streamed tokens + episode-match tags
│       └── ModeToggle.tsx    # teacher | interviewer switch
├── hooks/
│   └── useSSEChat.ts         # SSE EventSource hook
├── types/
│   └── api.ts                # ChatMessage interface
└── api/
    └── client.ts             # fetch wrappers
```

#### SSE Chat Hook

```typescript
// hooks/useSSEChat.ts
import { useState, useCallback } from "react";

export function useSSEChat() {
  const [tokens, setTokens] = useState<string[]>([]);

  const sendMessage = useCallback(
    async (message: string, mode: "teacher" | "interviewer") => {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, mode }),
      });
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        // parse SSE data lines
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data:")) {
            setTokens((prev) => [...prev, line.slice(5).trim()]);
          }
        }
      }
    },
    []
  );

  return { tokens, sendMessage };
}
```

---

### 9. Seed Topics

Seed data lives in `backend/memory/seed.py` as plain Python dicts conforming to `HistoricalEpisode`. On startup the lifespan hook calls `seed_tracks_if_absent()`:

```python
async def seed_tracks_if_absent():
    existing = await cognee.recall(graph_name="content_track", query="OS memory management")
    if not existing:
        await gateway.add_data_points(OS_MEMORY_EPISODES, temporal_cognify=True)
        await cognee.consolidate_entity_descriptions_pipeline()
    existing_dl = await cognee.recall(graph_name="content_track", query="deep learning")
    if not existing_dl:
        await gateway.add_data_points(DEEP_LEARNING_EPISODES, temporal_cognify=True)
        await cognee.consolidate_entity_descriptions_pipeline()
```

**OS Memory Management seed order:**
base+limit registers → segmentation → external fragmentation (failure) → paging → page tables → MMU/TLB

**Deep Learning seed order:**
perceptron → XOR failure (failure) → MLP → backpropagation → CNN (concurrent: RNN) → vanishing gradient (failure) → LSTM → attention mechanism → Transformer

---

### 10. Deployment

```
project-root/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   └── requirements.txt
└── frontend/
    ├── Dockerfile
    └── package.json
```

**docker-compose.yml:**

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
      - COGNEE_SKIP_CONNECTION_TEST=true
    volumes:
      - cognee_data:/app/.cognee_data

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  cognee_data:
```

**Backend Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

The `nginx.conf` proxies `/chat` and `/ingest` to `http://backend:8000` so the frontend makes all API calls to its own origin.

---

---

## Error Handling

| Error Class | Trigger | Handling |
|-------------|---------|----------|
| `MemoryAccessError` | Wrong agent calls a restricted write API | Raised immediately; request aborted; HTTP 403 returned |
| `TransientFetchError` | Wikipedia / arXiv / YouTube fetch fails | Exponential backoff retry (max 3 attempts), then reasoned fallback |
| `RecallVerificationError` | Self-check recall fails after ingestion | Retry ingestion (max 3 total); then reasoned fallback |
| `LangGraphRetryExhausted` | Node fails all 3 retry attempts | HTTP 503 returned to client with `retry_after` hint |
| `CogneeConnectionError` | cognee unavailable at startup | App exits; Docker restart policy handles recovery |
| `SSEClientDisconnect` | Client closes SSE connection mid-stream | Generator is cancelled; partial state is flushed to session log |

cognee is initialised with `COGNEE_SKIP_CONNECTION_TEST=true` to allow container startup without an active cognee instance (useful for test environments).

---

## Testing Strategy

### Dual Testing Approach

**Unit / example-based tests** cover specific behaviors with known inputs:
- Answer classifier returns each of the four labels for well-chosen inputs
- `narrative_sort` orders a small hand-crafted episode list correctly
- Seed data loader produces the right episode IDs
- Mode-toggle causes `TutorState.mode` to flip correctly
- `POST /ingest` returns HTTP 200 before ingestion completes (async)

**Property-based tests** cover universal invariants across generated inputs (see Correctness Properties below). These are the primary vehicle for correctness assurance. Run with minimum 100 iterations each.

### What Is Not Tested

This spec targets a single-presenter demo with no test suite mandate (per requirements). The correctness properties are written to be runnable but are provided as formal specifications for developer guidance rather than as a CI gate.

Infrastructure checks (docker-compose service startup, cognee graph naming, decorator application) are verified manually during deployment.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: HistoricalEpisode schema completeness

*For any* object that the Ingestion Agent writes to Track A, it SHALL contain all required fields (`id`, `concept`, `problem_posed`, `attempted_solution`, `outcome`, `why`, `requires`, `concurrent_with`, `source_confidence`) with values conforming to their declared types and enumerations.

**Validates: Requirements 1.1**

---

### Property 2: Dependency traversal respects only `requires` edges

*For any* directed episode graph, the traversal order computed by the Teacher Agent's `select_next_episode` SHALL be consistent with a topological sort over `requires` edges only; `concurrent_with` edges SHALL NOT impose any ordering constraint.

**Validates: Requirements 1.2, 1.3**

---

### Property 3: Reasoned episodes are idempotent per topic

*For any* topic, running the Ingestion Agent twice SHALL produce no additional `reasoned`-tier episodes beyond those created in the first run; the second run's Track A episode set SHALL be identical to the first run's.

**Validates: Requirements 1.4, 10.3**

---

### Property 4: Track A write isolation

*For any* attempt by the Teacher Agent, Interviewer Agent, or Trait Synthesis Agent to call `add_data_points()` targeting Track A, the MemoryGateway SHALL raise a `MemoryAccessError` and the write SHALL NOT reach cognee.

**Validates: Requirements 14.1, 14.4**

---

### Property 5: Track B write isolation

*For any* attempt by the Teacher Agent or Interviewer Agent to call `remember()`, `forget()`, or `improve()` on a graph named `user_*`, the MemoryGateway SHALL raise a `MemoryAccessError` and the write SHALL NOT reach cognee.

**Validates: Requirements 14.2, 14.3, 3.3, 3.4**

---

### Property 6: Track B graph naming invariant

*For any* user ID string, the Track B graph name produced by the system SHALL equal `"user_" + user_id`, never any other format.

**Validates: Requirements 3.1**

---

### Property 7: Multi-evidence requirement before remember()

*For any* set of agent-memory traces grouped by concept, the Trait Synthesis Agent SHALL call `remember()` only when the evidence count for that concept is at least 2; single-signal concepts SHALL be skipped.

**Validates: Requirements 3.6, 8.3**

---

### Property 8: Ingestion retry exhaustion leads to reasoned fallback

*For any* topic where all three self-check recall attempts fail, the Ingestion Agent SHALL produce at least one `reasoned`-tier episode for each unverified subtopic and write it to Track A.

**Validates: Requirements 4.6, 4.7**

---

### Property 9: Exponential backoff timing

*For any* sequence of transient fetch failures, the delay before attempt *k* (0-indexed) SHALL be at least 2^k seconds, and the Ingestion Agent SHALL NOT use the `tenacity` library for this retry logic.

**Validates: Requirements 4.8**

---

### Property 10: Answer classification is exhaustive and mutually exclusive

*For any* user answer string and current episode, the Teacher Agent's classifier SHALL return exactly one value from `{matched-failure, matched-success, partial, novel}` — never null, never a combination.

**Validates: Requirements 5.2**

---

### Property 11: Two-nudge stuck fallback contains all four parts

*For any* episode where the user's `nudge_count` reaches 2, the Teacher Agent's fallback response SHALL contain all four required sections: Problem framing, Solution hint, Engineering Insight, and Historical note.

**Validates: Requirements 5.6**

---

### Property 12: recall() precedes episode selection

*For any* call to `select_next_episode`, the Teacher Agent SHALL have called `recall()` on the user's Track B graph within the same turn, before computing the next episode.

**Validates: Requirements 5.7**

---

### Property 13: Digest mode does not advance episode position

*For any* transcript submitted to the Teacher Agent in digest mode, the `current_episode` field in `TutorState` SHALL be identical before and after the digest call.

**Validates: Requirements 6.1, 6.2**

---

### Property 14: Confidently-wrong penalty is strictly harsher

*For any* wrong answer, the penalty assigned when the user self-reports confidence ∈ {4, 5} SHALL be strictly greater than the penalty assigned for the identical wrong answer with confidence ∈ {1, 2}.

**Validates: Requirements 7.6**

---

### Property 15: Trait Synthesis triggers only at the three defined points

*For any* session lifecycle, the Trait Synthesis Agent SHALL be invoked exactly when one of the three trigger conditions occurs (Teacher session end, Interviewer session end, mode switch), and SHALL NOT be invoked at any other point.

**Validates: Requirements 8.2**

---

### Property 16: Seed topics are complete and non-duplicated

*For any* number of times `seed_tracks_if_absent()` is called, Track A SHALL contain exactly the hand-authored episodes for OS memory management and deep learning, with no duplicate episode IDs.

**Validates: Requirements 10.1, 10.2, 10.3**
