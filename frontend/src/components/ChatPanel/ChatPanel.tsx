import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUp, Loader2 } from "lucide-react";
import type { ChatMessage } from "../../types/api";
import { postSessionModeSwitch } from "../../api/client";
import { useSSEChat } from "../../hooks/useSSEChat";
import { MessageList } from "./MessageList";
import { ModeToggle } from "./ModeToggle";

function getOrCreateUserId(): string {
  const key = "fp_user_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

function parseEpisodeId(token: string): string | null {
  const trimmed = token.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.episode_id === "string") return parsed.episode_id;
  } catch { /* not JSON */ }
  return null;
}

interface ChatPanelProps {
  activeTopic: string;
  /** True while the active topic's ingestion is still running — gates all input */
  isInputGated: boolean;
  mode: "teacher" | "interviewer";
  onModeSwitch: (newMode: "teacher" | "interviewer") => void;
}

interface ToastItem {
  id: number;
  message: string;
  accent: string;
}

export function ChatPanel({ activeTopic, isInputGated, mode, onModeSwitch }: ChatPanelProps) {
  const [userId]    = useState(() => getOrCreateUserId());
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages]   = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [toasts, setToasts]       = useState<ToastItem[]>([]);
  const [modeTint, setModeTint]   = useState(false);

  const toastCounter = useRef(0);
  const textareaRef  = useRef<HTMLTextAreaElement>(null);
  const prevTopic    = useRef(activeTopic);
  const prevMode     = useRef(mode);

  const { tokens, isStreaming, error, sendMessage, reset } = useSSEChat();

  // ── Filter tokens: strip sentinel/metadata ────────────────────────────
  const { displayTokens, episodeId } = tokens.reduce<{
    displayTokens: string[];
    episodeId: string | null;
  }>(
    (acc, token) => {
      const eid = parseEpisodeId(token);
      if (eid !== null) return { ...acc, episodeId: eid };
      return { ...acc, displayTokens: [...acc.displayTokens, token] };
    },
    { displayTokens: [], episodeId: null }
  );
  const streamingContent = displayTokens.join("");

  // ── Reset session when topic changes ──────────────────────────────────
  useEffect(() => {
    if (prevTopic.current !== activeTopic) {
      prevTopic.current = activeTopic;
      setMessages([]);
      setSessionId(crypto.randomUUID());
      setInputText("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      reset();
    }
  }, [activeTopic, reset]);

  // ── Mode change: tint flash + toast ──────────────────────────────────
  useEffect(() => {
    if (prevMode.current === mode) return;
    prevMode.current = mode;

    // flash tint
    setModeTint(true);
    const tintOff = setTimeout(() => setModeTint(false), 700);

    // toast
    const accent = mode === "interviewer" ? "var(--accent-interviewer)" : "var(--accent-teacher)";
    const msg    = mode === "interviewer"
      ? "Interviewer mode — no more hints"
      : "Teacher mode — let's explore together";
    const id = ++toastCounter.current;
    setToasts((prev) => [...prev, { id, message: msg, accent }]);
    const toastOff = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 2500);

    return () => { clearTimeout(tintOff); clearTimeout(toastOff); };
  }, [mode]);

  // ── Commit streamed message when done ────────────────────────────────
  useEffect(() => {
    if (!isStreaming && streamingContent) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: streamingContent, episode_id: episodeId ?? undefined, mode },
      ]);
      reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming]);

  // ── Send ─────────────────────────────────────────────────────────────
  // isInputGated check is in THREE places: canSend flag, handleSend guard, onKeyDown guard
  const canSend = inputText.trim().length > 0 && !isStreaming && !isInputGated;

  const handleSend = useCallback(async () => {
    // Hard gate — block even if button state somehow desynchronised
    if (!inputText.trim() || isStreaming || isInputGated) return;

    const trimmed = inputText.trim();
    setMessages((prev) => [...prev, { role: "user", content: trimmed, mode }]);
    setInputText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      await sendMessage({ message: trimmed, mode, user_id: userId, session_id: sessionId, topic: activeTopic });
    } catch { /* error surfaced via hook state */ }
  }, [inputText, isStreaming, isInputGated, mode, sendMessage, sessionId, activeTopic, userId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      // Gate Enter-to-send the same way as button
      if (!isInputGated) void handleSend();
    }
  };

  const handleTextInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };

  const handleModeSwitch = useCallback(async (newMode: "teacher" | "interviewer") => {
    if (newMode === mode) return;
    const newSession = crypto.randomUUID();
    try {
      await postSessionModeSwitch({
        user_id: userId, session_id: sessionId,
        current_mode: mode, new_mode: newMode,
      });
    } catch (err) { console.error("Mode-switch notification failed:", err); }
    setSessionId(newSession);
    setMessages([]);
    reset();
    onModeSwitch(newMode);
  }, [mode, sessionId, userId, reset, onModeSwitch]);

  // ── Background tint colour ────────────────────────────────────────────
  const tintColor = modeTint
    ? (mode === "interviewer" ? "var(--accent-interviewer-tint)" : "var(--accent-teacher-tint)")
    : (mode === "interviewer" ? "var(--accent-interviewer-tint)" : "var(--accent-teacher-tint)");

  return (
    <motion.div
      animate={{ backgroundColor: tintColor }}
      transition={{ duration: modeTint ? 0.1 : 0.6 }}
      style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}
    >
      {/* ── Message list ── */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <MessageList messages={messages} streamingContent={streamingContent} />
      </div>

      {/* ── Bottom input area ── */}
      <div
        style={{
          padding: "12px 24px 18px",
          borderTop: "1px solid #21262d",
          backgroundColor: "#010409",
          flexShrink: 0,
          position: "relative",
        }}
      >
        {/* Mode toggle */}
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
          <ModeToggle currentMode={mode} disabled={isStreaming} onModeSwitch={(m) => void handleModeSwitch(m)} />
        </div>

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                marginBottom: 10,
                padding: "8px 12px",
                backgroundColor: "#1c0a0a",
                border: "1px solid #3d1515",
                borderRadius: 8,
                color: "#f85149",
                fontSize: 13,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
              role="alert"
            >
              <span style={{ flex: 1 }}>Something went wrong on our end — try again in a moment.</span>
              <button
                onClick={reset}
                style={{ background: "none", border: "none", color: "#f85149", cursor: "pointer", fontSize: 16, padding: 0 }}
                aria-label="Dismiss"
              >
                ×
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Streaming indicator */}
        <AnimatePresence>
          {isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                marginBottom: 8,
                fontSize: 12,
                color: "#58a6ff",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <motion.span
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                style={{ display: "flex" }}
              >
                <Loader2 size={12} />
              </motion.span>
              Generating response…
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input row */}
        <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder={
              isInputGated
                ? "Preparing your first lesson..."
                : mode === "teacher"
                ? "Ask a question or explore a concept…  Enter to send"
                : "Answer the question…  Enter to send"
            }
            value={inputText}
            onChange={handleTextInput}
            onKeyDown={handleKeyDown}
            // Functionally disabled — not just visual
            disabled={isStreaming || isInputGated}
            aria-label="Message input"
            aria-disabled={isStreaming || isInputGated}
            style={{
              flex: 1,
              resize: "none",
              border: `1px solid ${canSend ? "var(--accent-teacher)" : "#21262d"}`,
              borderRadius: 10,
              padding: "11px 14px",
              fontSize: 14,
              lineHeight: 1.5,
              outline: "none",
              color: "#e6edf3",
              backgroundColor: "#0d1117",
              fontFamily: "'Inter', system-ui, sans-serif",
              transition: "border-color 0.15s, opacity 0.2s",
              overflowY: "hidden",
              minHeight: 44,
              maxHeight: 160,
              // Visual gate: grey out during ingestion
              opacity: isInputGated ? 0.35 : 1,
              cursor: isInputGated ? "not-allowed" : isStreaming ? "default" : "text",
            }}
          />
          <motion.button
            onClick={() => void handleSend()}
            // Also functionally disabled
            disabled={!canSend}
            whileTap={canSend ? { scale: 0.93 } : {}}
            aria-label="Send message"
            style={{
              width: 44,
              height: 44,
              borderRadius: 10,
              border: "none",
              backgroundColor: canSend ? "var(--accent-teacher)" : "#21262d",
              color: canSend ? "#fff" : "#484f58",
              cursor: canSend ? "pointer" : "not-allowed",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "background-color 0.15s",
            }}
          >
            <ArrowUp size={18} strokeWidth={2.5} />
          </motion.button>
        </div>

        <p style={{ fontSize: 11, color: "#30363d", marginTop: 7, textAlign: "center" }}>
          Shift+Enter for new line
        </p>

        {/* Toast stack */}
        <div
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            flexDirection: "column",
            gap: 6,
            alignItems: "center",
            pointerEvents: "none",
          }}
          aria-live="polite"
        >
          <AnimatePresence>
            {toasts.map((t) => (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, y: 8, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -4, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                style={{
                  padding: "7px 18px",
                  borderRadius: 20,
                  backgroundColor: t.accent,
                  color: "#fff",
                  fontSize: 13,
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                }}
              >
                {t.message}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}
