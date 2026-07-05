import type { ChatRequest, IngestRequest, IngestResponse } from "../types/api";

/**
 * Base URL for all API calls.
 *
 * - In local dev (Vite proxy) and Docker (nginx proxy): empty string — requests
 *   go to the same origin and are proxied to the backend.
 * - On Vercel: set VITE_API_BASE_URL to your Render backend URL
 *   (e.g. https://first-principle-api.onrender.com) so API calls reach the
 *   correct host instead of the Vercel edge.
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

/**
 * Request body for POST /session/mode_switch.
 * Signals a mode change to the backend so it can trigger
 * Chain 2 (trait synthesis) before starting the new mode session.
 * `current_mode` is the mode being switched *away from*.
 */
export interface ModeSwitchRequest {
  user_id: string;
  session_id: string;
  current_mode: "teacher" | "interviewer" | "digest";
  new_mode: "teacher" | "interviewer" | "digest";
}

/**
 * POST /chat — Returns the raw Response so the caller can consume
 * the SSE stream incrementally via response.body.
 *
 * The backend streams tokens as `data: <token>\n\n` lines and signals
 * end-of-stream with `event: done\ndata: \n\n`.
 */
export async function postChat(req: ChatRequest): Promise<Response> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    throw new Error(`POST /chat failed: ${response.status} ${response.statusText}`);
  }

  return response;
}

/**
 * POST /ingest — Queues an ingestion job and returns the parsed JSON
 * confirmation once the backend has acknowledged the request.
 */
export async function postIngest(req: IngestRequest): Promise<IngestResponse> {
  const response = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    throw new Error(`POST /ingest failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<IngestResponse>;
}

/**
 * POST /session/mode-switch — Notifies the backend of a mode change.
 *
 * The backend uses this signal to end the current session (triggering
 * Trait Synthesis Chain 2) and start a new session in the requested mode.
 * Errors are non-fatal — a failure here should not block the UI from
 * switching modes locally.
 *
 * Requirement 12.3
 */
export async function postSessionModeSwitch(req: ModeSwitchRequest): Promise<void> {
  const response = await fetch(`${API_BASE}/session/mode_switch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    throw new Error(
      `POST /session/mode-switch failed: ${response.status} ${response.statusText}`
    );
  }
}
