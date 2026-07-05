/**
 * MessageList — renders the chat conversation.
 *
 * Each assistant message may carry an optional `episode_id` sourced from the
 * SSE metadata. When present, an episode-match tag is rendered alongside the
 * message content (Requirement 12.4).
 *
 * A `streamingContent` prop holds the partial text of the currently arriving
 * assistant reply so that tokens are displayed as they stream in (Requirement 12.2).
 */
import type { ChatMessage } from "../../types/api";

interface MessageListProps {
  messages: ChatMessage[];
  /** Partial text of the assistant reply currently being streamed. Empty string when idle. */
  streamingContent: string;
}

/** Inline styles for individual message bubbles. */
const userBubbleStyle: React.CSSProperties = {
  alignSelf: "flex-end",
  backgroundColor: "#2563eb",
  color: "#fff",
  borderRadius: "16px 16px 4px 16px",
  padding: "10px 14px",
  maxWidth: "75%",
  wordBreak: "break-word",
  lineHeight: 1.5,
};

const assistantBubbleStyle: React.CSSProperties = {
  alignSelf: "flex-start",
  backgroundColor: "#f3f4f6",
  color: "#111827",
  borderRadius: "16px 16px 16px 4px",
  padding: "10px 14px",
  maxWidth: "75%",
  wordBreak: "break-word",
  lineHeight: 1.5,
};

const streamingBubbleStyle: React.CSSProperties = {
  ...assistantBubbleStyle,
  borderBottom: "2px solid #2563eb",
};

const episodeTagStyle: React.CSSProperties = {
  display: "inline-block",
  marginTop: 4,
  fontSize: 11,
  fontFamily: "monospace",
  backgroundColor: "#dbeafe",
  color: "#1d4ed8",
  borderRadius: 4,
  padding: "1px 6px",
};

const roleLabelStyle: React.CSSProperties = {
  fontSize: 11,
  color: "#9ca3af",
  marginBottom: 2,
  fontFamily: "sans-serif",
};

/**
 * Renders a single message bubble with an optional episode-match tag.
 * Requirement 12.4 — annotate each message with an episode-match tag
 * indicating which HistoricalEpisode the message corresponds to, where applicable.
 */
function MessageBubble({
  message,
  isStreaming,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";
  const bubbleStyle = isStreaming
    ? streamingBubbleStyle
    : isUser
    ? userBubbleStyle
    : assistantBubbleStyle;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: 12,
      }}
    >
      <span style={roleLabelStyle}>{isUser ? "You" : "FirstPrinciple"}</span>
      <div style={bubbleStyle}>
        <span style={{ whiteSpace: "pre-wrap" }}>{message.content}</span>
        {/* Episode-match tag — shown only when episode_id is present (Req 12.4) */}
        {!isUser && message.episode_id && (
          <div>
            <span style={episodeTagStyle} title="Matched HistoricalEpisode">
              📖 {message.episode_id}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * MessageList renders committed messages plus an in-progress streaming bubble.
 *
 * Requirement 12.2 — display streamed responses token-by-token as they arrive.
 * Requirement 12.4 — annotate each message with an episode-match tag where applicable.
 */
export function MessageList({ messages, streamingContent }: MessageListProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        padding: "16px 16px 0",
        overflowY: "auto",
        flex: 1,
      }}
    >
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}

      {/* In-flight streaming bubble — shown while tokens are arriving */}
      {streamingContent && (
        <MessageBubble
          message={{ role: "assistant", content: streamingContent }}
          isStreaming
        />
      )}
    </div>
  );
}
