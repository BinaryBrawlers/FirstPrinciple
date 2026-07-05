import { useState, useCallback, useRef } from "react";
import { postChat } from "../api/client";
import type { ChatRequest } from "../types/api";

/**
 * Parameters passed to `sendMessage`, extending the full backend ChatRequest
 * with an optional `topic` field (required only on first turn for a new topic).
 */
export interface SendMessageParams {
  message: string;
  mode: "teacher" | "interviewer";
  user_id: string;
  session_id: string;
  topic?: string;
}

/**
 * Return type of the `useSSEChat` hook.
 */
export interface UseSSEChatResult {
  /** Accumulated tokens received from the current/last stream. */
  tokens: string[];
  /** Whether a stream is currently in progress. */
  isStreaming: boolean;
  /** The last error that occurred, if any. */
  error: Error | null;
  /**
   * Send a message and stream the response token-by-token.
   * Clears previous tokens before starting a new stream.
   */
  sendMessage: (params: SendMessageParams) => Promise<void>;
  /** Reset tokens, error, and streaming state. */
  reset: () => void;
}

/**
 * `useSSEChat` — streams tokens from `POST /chat` using the Fetch API and
 * a `ReadableStream` reader with `TextDecoder`. Parses SSE `data:` lines as
 * individual tokens and stops on `event: done`.
 *
 * Requirements: 12.2 — The Chat Panel SHALL display streamed Teacher and
 * Interviewer responses token-by-token as they arrive via SSE.
 */
export function useSSEChat(): UseSSEChatResult {
  const [tokens, setTokens] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Track the active reader so we can cancel on unmount or re-send.
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);

  const reset = useCallback(() => {
    setTokens([]);
    setIsStreaming(false);
    setError(null);
  }, []);

  const sendMessage = useCallback(
    async ({
      message,
      mode,
      user_id,
      session_id,
      topic,
    }: SendMessageParams): Promise<void> => {
      // Cancel any in-progress stream before starting a new one.
      if (readerRef.current) {
        try {
          await readerRef.current.cancel();
        } catch {
          // Ignore cancellation errors from a previous stream.
        }
        readerRef.current = null;
      }

      setTokens([]);
      setError(null);
      setIsStreaming(true);

      const req: ChatRequest = { user_id, session_id, message, mode, topic };

      try {
        const response = await postChat(req);

        if (!response.body) {
          throw new Error("Response body is null — streaming not supported in this environment");
        }

        const reader = response.body.getReader();
        readerRef.current = reader;
        const decoder = new TextDecoder();

        // Buffer for incomplete SSE lines split across chunks.
        let lineBuffer = "";
        // Track whether the current SSE event is a "done" event.
        let isDoneEvent = false;

        outer: while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          // Decode the chunk and append to the running line buffer.
          const chunk = decoder.decode(value, { stream: true });
          lineBuffer += chunk;

          // Split on newlines; the last element may be an incomplete line.
          const lines = lineBuffer.split("\n");
          lineBuffer = lines.pop() ?? "";

          for (const rawLine of lines) {
            const line = rawLine.trimEnd();

            if (line === "") {
              // Blank line = end of an SSE event block; reset event-level state.
              isDoneEvent = false;
              continue;
            }

            if (line.startsWith("event:")) {
              const eventName = line.slice(6).trim();
              isDoneEvent = eventName === "done";
              // "ingesting" is a backend signal only — never shown to the user as chat text
              if (eventName === "ingesting") {
                // No token added — ingestion state is managed by useIngestion hook
              }
              if (isDoneEvent) {
                break outer;
              }
              continue;
            }

            if (line.startsWith("data:")) {
              if (isDoneEvent) {
                break outer;
              }
              // Use slice(5) WITHOUT .trim() — trimming strips the leading space
              // that separates words when Mistral streams tokens like "data: foo".
              // Only skip genuinely empty data lines.
              const token = line.slice(5);
              // Never emit ingestion sentinel or event data as chat tokens
              if (token && !token.startsWith("__ingesting__")) {
                setTokens((prev) => [...prev, token]);
              }
            }

            // Ignore `id:`, `retry:`, and comment lines (`:`) per SSE spec.
          }
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
      } finally {
        readerRef.current = null;
        setIsStreaming(false);
      }
    },
    []
  );

  return { tokens, isStreaming, error, sendMessage, reset };
}
