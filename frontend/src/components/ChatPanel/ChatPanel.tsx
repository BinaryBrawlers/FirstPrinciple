/**
 * ChatPanel — the primary chat interface.
 *
 * Responsibilities:
 * - Manages session state: stable `user_id` (persisted in localStorage),
 *   and a per-session `session_id` (regenerated on mode switch or page load).
 * - Holds the `messages: ChatMessage[]` conversation history.
 * - Consumes `useSSEChat` to stream assistant tokens token-by-token.
 * - Parses any trailing JSON metadata token emitted by the backend to extract
 *   `episode_id` for the most recent assistant message.
 * - Exposes `ModeToggle` and calls `POST /session/mode-switch` when the user
 *   changes mode.
 *
 * Requirements: 12.1, 12.2, 12.3, 12.4
 */
import { useState, useEffect, useRef, useCallback } from "react";
import type { ChatMessage } from "../../types/api";
import { postSessionModeSwitch } from "../../api/client";
import { useSSEChat } from "../../hooks/useSSEChat";
import { MessageList } from "./MessageList";
import { ModeToggle } from "./ModeToggle";

// ---------------------------------------------------------------------------
// Session helpers
// ---------------------------------------------------------------------------

/** Generates or retrieves a stable user_id from localStorage. */
function getOrCreateUserId(): string {
  const key = "fp_user_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

/** Creates a fresh session_id (UUID). */
function newSessionId(): string {
  return crypto.randomUUID();
}

// ---------------------------------------------------------------------------
// JSON metadata parsing
// ---------------------------------------------------------------------------

/**
 * Checks whether a token is a JSON object containing an `episode_id` field.
 * The backend may emit a trailing metadata token like `{"episode_id": "ep-42"}`.
 * Returns the episode_id string if found, otherwise null.
 */
function parseEpisodeId(token: string): string | null {
  const trimmed = token.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.episode_id === "string") {
      return parsed.episode_id;
    }
  } catch {
    // Not valid JSON — treat as a regular text token.
  }
  return null;
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const panelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  width: "100%",
  height: "100%",
  fontFamily: "sans-serif",
  backgroundColor: "#fff",
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "12px 16px",
  borderBottom: "1px solid #e5e7eb",
  flexShrink: 0,
};

const titleStyle: React.CSSProperties = {
  fontSize: 18,
  fontWeight: 700,
  color: "#111827",
  margin: 0,
};

const messagesAreaStyle: React.CSSProperties = {
  flex: 1,
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
};

const inputRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  padding: "12px 16px",
  borderTop: "1px solid #e5e7eb",
  flexShrink: 0,
};

const textareaStyle: React.CSSProperties = {
  flex: 1,
  resize: "none",
  border: "1px solid #d1d5db",
  borderRadius: 8,
  padding: "8px 12px",
  fontFamily: "sans-serif",
  fontSize: 14,
  lineHeight: 1.5,
  outline: "none",
  color: "#111827",
};

const sendButtonStyle = (disabled: boolean): React.CSSProperties => ({
  padding: "8px 20px",
  borderRadius: 8,
  border: "none",
  backgroundColor: disabled ? "#93c5fd" : "#2563eb",
  color: "#fff",
  cursor: disabled ? "not-allowed" : "pointer",
  fontFamily: "sans-serif",
  fontSize: 14,
  fontWeight: 600,
  flexShrink: 0,
});

const errorStyle: React.CSSProperties = {
  padding: "6px 16px",
  backgroundColor: "#fee2e2",
  color: "#b91c1c",
  fontSize: 13,
  fontFamily: "sans-serif",
  flexShrink: 0,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * ChatPanel — full-width, full-height SSE-consuming chat panel.
 *
 * Requirement 12.1 — full-width primary interface (layout handled by App.tsx).
 * Requirement 12.2 — displays streamed responses token-by-token via useSSEChat.
 * Requirement 12.3 — mode toggle (ModeToggle) with backend notification.
 * Requirement 12.4 — episode-match tags rendered by MessageList.
 */
export function ChatPanel() {
  // --- Session identity ---
  const [userId] = useState<string>(() => getOrCreateUserId());
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());

  // --- Conversation state ---
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<"teacher" | "interviewer">("teacher");
  const [topic, setTopic] = useState<string | undefined>(undefined);
  const [inputText, setInputText] = useState("");

  // --- SSE hook ---
  const { tokens, isStreaming, error, sendMessage, reset } = useSSEChat();

  // Scroll-to-bottom anchor
  const bottomRef = useRef<HTMLDivElement>(null);

  // ---------------------------------------------------------------------------
  // Build the current streaming content from accumulated tokens.
  // Filter out any trailing metadata token (JSON with episode_id).
  // ---------------------------------------------------------------------------
  const { displayTokens, episodeId } = tokens.reduce<{
    displayTokens: string[];
    episodeId: string | null;
  }>(
    (acc, token) => {
      const eid = parseEpisodeId(token);
      if (eid !== null) {
        return { ...acc, episodeId: eid };
      }
      return { ...acc, displayTokens: [...acc.displayTokens, token] };
    },
    { displayTokens: [], episodeId: null }
  );

  const streamingContent = displayTokens.join("");

  // ---------------------------------------------------------------------------
  // When streaming ends, commit the accumulated tokens as an assistant message.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!isStreaming && streamingContent) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: streamingContent,
          episode_id: episodeId ?? undefined,
          mode,
        },
      ]);
      reset();
    }
    // Only run when isStreaming transitions to false (not on every token update).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming]);

  // Auto-scroll to bottom whenever messages or streaming content changes.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // ---------------------------------------------------------------------------
  // Send a user message
  // ---------------------------------------------------------------------------
  const handleSend = useCallback(async () => {
    const trimmed = inputText.trim();
    if (!trimmed || isStreaming) return;

    // Append user message immediately.
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed, mode },
    ]);
    setInputText("");

    // Use the first message's content as the topic if none is set yet.
    const currentTopic = topic ?? trimmed;
    if (!topic) setTopic(currentTopic);

    try {
      await sendMessage({
        message: trimmed,
        mode,
        user_id: userId,
        session_id: sessionId,
        topic: currentTopic,
      });
    } catch {
      // Errors are surfaced via the `error` state from useSSEChat.
    }
  }, [inputText, isStreaming, mode, sendMessage, sessionId, topic, userId]);

  // Allow Ctrl+Enter or Shift+Enter to send.
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  // ---------------------------------------------------------------------------
  // Mode switch handler (Requirement 12.3)
  // ---------------------------------------------------------------------------
  const handleModeSwitch = useCallback(
    async (newMode: "teacher" | "interviewer") => {
      if (newMode === mode) return;

      const newSession = newSessionId();

      // Notify backend — non-fatal; log error but don't block the UI.
      try {
        await postSessionModeSwitch({
          user_id: userId,
          session_id: sessionId,
          current_mode: mode,
          new_mode: newMode,
        });
      } catch (err) {
        console.error("Mode-switch notification failed:", err);
      }

      // Start fresh session in the new mode.
      setMode(newMode);
      setSessionId(newSession);
      setTopic(undefined);
      setMessages([]);
      reset();
    },
    [mode, sessionId, userId, reset]
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={panelStyle}>
      {/* Header: title + mode toggle */}
      <header style={headerStyle}>
        <h1 style={titleStyle}>FirstPrinciple</h1>
        <ModeToggle
          currentMode={mode}
          disabled={isStreaming}
          onModeSwitch={(m) => void handleModeSwitch(m)}
        />
      </header>

      {/* Error banner */}
      {error && (
        <div style={errorStyle} role="alert">
          ⚠️ {error.message}
        </div>
      )}

      {/* Message list — scrollable area */}
      <div style={messagesAreaStyle}>
        <MessageList messages={messages} streamingContent={streamingContent} />
        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <form
        style={inputRowStyle}
        onSubmit={(e) => {
          e.preventDefault();
          void handleSend();
        }}
      >
        <textarea
          style={textareaStyle}
          rows={2}
          placeholder={
            mode === "teacher"
              ? "Ask a question or describe a topic to explore…"
              : "Answer the interviewer's question…"
          }
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
          aria-label="Message input"
        />
        <button
          type="submit"
          style={sendButtonStyle(isStreaming || !inputText.trim())}
          disabled={isStreaming || !inputText.trim()}
          aria-label="Send message"
        >
          {isStreaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
