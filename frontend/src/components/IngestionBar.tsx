import { useState, useRef } from "react";
import { Loader2, Check, AlertCircle, FlaskConical } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import type { TopicState, UseIngestionResult } from "../hooks/useIngestion";
import { INGESTION_STAGES } from "../hooks/useIngestion";

type IngestionBarProps = Pick<
  UseIngestionResult,
  "topics" | "activeTopic" | "addTopic" | "setActiveTopic"
>;

function StageLabel({ stage }: { stage: number }) {
  const label = INGESTION_STAGES[Math.min(stage, INGESTION_STAGES.length - 1)];
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.span
        key={stage}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        style={{ fontSize: 11, color: "#d29922", whiteSpace: "nowrap" }}
      >
        {label}
      </motion.span>
    </AnimatePresence>
  );
}

function TopicChip({
  state,
  isActive,
  onClick,
}: {
  state: TopicState;
  isActive: boolean;
  onClick: () => void;
}) {
  const ingesting = state.status === "ingesting";
  const ready = state.status === "ready";
  const errored = state.status === "error";

  return (
    <motion.button
      onClick={onClick}
      aria-pressed={isActive}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        padding: "5px 13px",
        borderRadius: 20,
        border: isActive
          ? "1px solid var(--accent-teacher)"
          : ingesting
          ? "1px solid rgba(210,153,34,0.4)"
          : "1px solid #21262d",
        backgroundColor: isActive
          ? "rgba(31,111,235,0.12)"
          : ingesting
          ? "rgba(210,153,34,0.07)"
          : "#0d1117",
        color: isActive ? "#58a6ff" : ingesting ? "#d29922" : "#8b949e",
        fontSize: 13,
        fontWeight: isActive ? 600 : 400,
        cursor: "pointer",
        whiteSpace: "nowrap",
        flexShrink: 0,
        fontFamily: "inherit",
        transition: "border-color 0.15s, background-color 0.15s, color 0.15s",
      }}
    >
      {ingesting && (
        <>
          <motion.span
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            style={{ display: "flex", alignItems: "center" }}
          >
            <Loader2 size={12} />
          </motion.span>
          <StageLabel stage={state.stage} />
          <span style={{ color: "#484f58", margin: "0 2px" }}>·</span>
        </>
      )}
      {ready && (
        <Check size={12} color="#3fb950" strokeWidth={2.5} />
      )}
      {errored && (
        <AlertCircle size={12} color="#f85149" />
      )}
      <span>{state.name}</span>
    </motion.button>
  );
}

export function IngestionBar({
  topics,
  activeTopic,
  addTopic,
  setActiveTopic,
}: IngestionBarProps) {
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    addTopic(trimmed);
    setInputValue("");
    inputRef.current?.blur();
  };

  const topicList = Array.from(topics.values());
  const hasTopics = topicList.length > 0;

  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        padding: "12px 24px",
        backgroundColor: "#010409",
        borderBottom: "1px solid #21262d",
        flexShrink: 0,
        flexWrap: "wrap",
      }}
    >
      {/* Left: wordmark + tagline + input */}
      <div style={{ display: "flex", alignItems: "center", gap: 20, flex: "0 1 auto", minWidth: 0 }}>
        {/* Wordmark + tagline */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <FlaskConical size={18} color="#58a6ff" strokeWidth={1.5} />
            <span
              style={{
                fontFamily: "'Fraunces', Georgia, serif",
                fontSize: 19,
                fontWeight: 700,
                letterSpacing: "-0.025em",
                color: "#e6edf3",
                whiteSpace: "nowrap",
                userSelect: "none",
              }}
            >
              FirstPrinciple
            </span>
          </div>
          <span
            style={{
              fontSize: 12,
              color: "#484f58",
              fontStyle: "italic",
              whiteSpace: "nowrap",
              userSelect: "none",
              letterSpacing: "0.01em",
            }}
          >
            Learn by reinventing
          </span>
        </div>

        {/* Input form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="What do you want to reinvent today?"
            aria-label="Topic input"
            style={{
              width: 280,
              padding: "7px 13px",
              borderRadius: 8,
              border: "1px solid #30363d",
              backgroundColor: "#0d1117",
              color: "#e6edf3",
              fontSize: 13,
              fontFamily: "'Inter', system-ui, sans-serif",
              outline: "none",
              transition: "border-color 0.15s",
            }}
            onFocus={(e) => { e.currentTarget.style.borderColor = "var(--accent-teacher)"; }}
            onBlur={(e)  => { e.currentTarget.style.borderColor = "#30363d"; }}
          />
          <motion.button
            type="submit"
            disabled={!inputValue.trim()}
            whileTap={inputValue.trim() ? { scale: 0.96 } : {}}
            style={{
              padding: "7px 16px",
              borderRadius: 8,
              border: "none",
              backgroundColor: inputValue.trim() ? "var(--accent-teacher)" : "#21262d",
              color: inputValue.trim() ? "#fff" : "#484f58",
              fontSize: 13,
              fontWeight: 500,
              cursor: inputValue.trim() ? "pointer" : "not-allowed",
              transition: "background-color 0.15s",
              whiteSpace: "nowrap",
              fontFamily: "inherit",
            }}
          >
            Start Learning
          </motion.button>
        </form>
      </div>

      {/* Right: topic chips */}
      {hasTopics && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            overflowX: "auto",
            flexShrink: 1,
            paddingBottom: 1,
          }}
          aria-label="Active topics"
        >
          <AnimatePresence initial={false}>
            {topicList.map((state) => (
              <motion.div
                key={state.name}
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                transition={{ duration: 0.2 }}
              >
                <TopicChip
                  state={state}
                  isActive={state.name === activeTopic}
                  onClick={() => setActiveTopic(state.name)}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </header>
  );
}
