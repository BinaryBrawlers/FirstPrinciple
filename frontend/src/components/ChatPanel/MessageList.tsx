import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookMarked } from "lucide-react";
import type { ChatMessage } from "../../types/api";

interface MessageListProps {
  messages: ChatMessage[];
  streamingContent: string;
}

const bubbleVariants = {
  hidden:  { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.22, ease: "easeOut" } },
  exit:    { opacity: 0, y: -4, transition: { duration: 0.15 } },
};

function TypingDots() {
  return (
    <span style={{ display: "inline-flex", gap: 4, alignItems: "center", padding: "2px 0" }}>
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          animate={{ y: [0, -5, 0], opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2, ease: "easeInOut" }}
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            backgroundColor: "#58a6ff",
            display: "inline-block",
          }}
        />
      ))}
    </span>
  );
}

function MessageBubble({
  message,
  isStreaming = false,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";

  return (
    <motion.div
      variants={bubbleVariants}
      initial="hidden"
      animate="visible"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: 20,
      }}
    >
      <span
        style={{
          fontSize: 10,
          color: "#484f58",
          marginBottom: 5,
          fontWeight: 600,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        {isUser ? "You" : "FirstPrinciple"}
      </span>

      <div
        style={{
          maxWidth: "72%",
          borderRadius: isUser ? "18px 18px 5px 18px" : "18px 18px 18px 5px",
          padding: "11px 16px",
          backgroundColor: isUser ? "var(--accent-teacher)" : "#161b22",
          color: isUser ? "#fff" : "#e6edf3",
          border: isUser
            ? "none"
            : `1px solid ${isStreaming ? "var(--accent-teacher)" : "#21262d"}`,
          boxShadow: isStreaming ? "0 0 0 1px rgba(31,111,235,0.15)" : "none",
          wordBreak: "break-word",
          lineHeight: 1.7,
          fontSize: 14,
          fontFamily: "'Inter', system-ui, sans-serif",
          transition: "border-color 0.2s, box-shadow 0.2s",
        }}
      >
        {isStreaming && !message.content
          ? <TypingDots />
          : <span style={{ whiteSpace: "pre-wrap", fontFamily: "'Inter', system-ui, sans-serif" }}>{message.content}</span>
        }

        {!isUser && message.episode_id && (
          <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid #21262d" }}>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                fontSize: 11,
                fontFamily: "monospace",
                color: "var(--accent-teacher)",
                opacity: 0.8,
              }}
            >
              <BookMarked size={11} />
              {message.episode_id}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function MessageList({ messages, streamingContent }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  if (messages.length === 0 && !streamingContent) return null;

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "32px 24px 12px",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <AnimatePresence initial={false}>
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
      </AnimatePresence>

      {streamingContent && (
        <MessageBubble
          message={{ role: "assistant", content: streamingContent }}
          isStreaming
        />
      )}

      <div ref={bottomRef} />
    </div>
  );
}
