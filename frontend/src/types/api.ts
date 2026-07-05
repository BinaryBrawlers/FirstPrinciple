/**
 * API type definitions for the FirstPrinciple frontend.
 * These types match the backend SSE contract and HTTP API.
 */

/**
 * A single chat message in a session.
 * `episode_id` is populated when the backend tags the message
 * with a matching HistoricalEpisode (Requirement 12.4).
 */
export interface ChatMessage {
  content: string;
  role: "user" | "assistant";
  episode_id?: string;
  mode?: "teacher" | "interviewer";
}

/**
 * Request body for POST /chat.
 * Produces an SSE stream of response tokens.
 */
export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
  mode: "teacher" | "interviewer";
  topic?: string;
}

/**
 * Request body for POST /ingest.
 * Triggers the Ingestion Agent asynchronously.
 */
export interface IngestRequest {
  topic: string;
  video_ids?: string[];
}

/**
 * Response from POST /ingest.
 * The ingestion task is queued and runs in the background.
 */
export interface IngestResponse {
  status: "queued";
  topic: string;
}
