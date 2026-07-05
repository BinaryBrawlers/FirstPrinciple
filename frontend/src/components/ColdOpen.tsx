import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight } from "lucide-react";

// ── extend this array freely ──────────────────────────────────────────────
const QUOTES = [
  { text: "What I cannot create, I do not understand.", author: "Feynman" },
  { text: "The only way to learn mathematics is to do mathematics.", author: "Halmos" },
  { text: "If you can't explain it simply, you don't understand it well enough.", author: "Einstein" },
  { text: "First principles thinking: refuse to be constrained by convention.", author: "Musk" },
  { text: "The art of programming is the art of organizing complexity.", author: "Dijkstra" },
  { text: "Simplicity is the ultimate sophistication.", author: "da Vinci" },
] as const;

const SUGGESTIONS = [
  "Mutex Locks",
  "Memory Segmentation",
  "Convolutional Neural Networks",
  "The Transformer Architecture",
] as const;

const HEADLINE = "What do you want to\nreinvent today?";

interface ColdOpenProps {
  onSelectTopic: (topic: string) => void;
}

/** Typewriter that types each character one at a time via framer-motion stagger */
function TypewriterHeadline({ text }: { text: string }) {
  const chars = text.split("");
  return (
    <h2
      aria-label={text.replace("\n", " ")}
      style={{
        fontSize: "clamp(28px, 5vw, 52px)",
        fontWeight: 700,
        fontFamily: "'Fraunces', Georgia, serif",
        letterSpacing: "-0.03em",
        color: "#e6edf3",
        textAlign: "center",
        lineHeight: 1.15,
        whiteSpace: "pre-line",
      }}
    >
      <motion.span
        initial="hidden"
        animate="visible"
        variants={{
          visible: { transition: { staggerChildren: 0.035 } },
        }}
      >
        {chars.map((char, i) => (
          <motion.span
            key={i}
            variants={{
              hidden:  { opacity: 0 },
              visible: { opacity: 1 },
            }}
            transition={{ duration: 0.01 }}
            style={{ display: "inline", whiteSpace: char === "\n" ? "pre" : "normal" }}
          >
            {char}
          </motion.span>
        ))}
      </motion.span>
    </h2>
  );
}

function RotatingQuote() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setIndex((prev) => (prev + 1) % QUOTES.length);
    }, 4500);
    return () => clearInterval(id);
  }, []);

  const q = QUOTES[index];

  return (
    <AnimatePresence mode="wait">
      <motion.p
        key={index}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        aria-live="polite"
        style={{
          fontSize: 14,
          color: "#484f58",
          fontStyle: "italic",
          textAlign: "center",
          maxWidth: 480,
          lineHeight: 1.75,
        }}
      >
        &ldquo;{q.text}&rdquo;
        <span style={{ display: "block", marginTop: 4, fontSize: 12, fontStyle: "normal", color: "#30363d" }}>
          — {q.author}
        </span>
      </motion.p>
    </AnimatePresence>
  );
}

export function ColdOpen({ onSelectTopic }: ColdOpenProps) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 36,
        padding: "48px 24px",
        userSelect: "none",
        overflow: "auto",
      }}
    >
      {/* Animated tagline above headline */}
      <motion.p
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        style={{
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--accent-teacher)",
          opacity: 0.85,
        }}
      >
        Learn the way history did
      </motion.p>

      <TypewriterHeadline text={HEADLINE} />

      {/* Suggestion chips */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6, duration: 0.5 }}
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          justifyContent: "center",
          maxWidth: 660,
        }}
        role="list"
        aria-label="Topic suggestions"
      >
        {SUGGESTIONS.map((s, i) => (
          <motion.button
            key={s}
            role="listitem"
            onClick={() => onSelectTopic(s)}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.7 + i * 0.07, duration: 0.3 }}
            whileHover={{ scale: 1.04, borderColor: "var(--accent-teacher)" }}
            whileTap={{ scale: 0.97 }}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              padding: "10px 20px",
              borderRadius: 24,
              border: "1px solid #30363d",
              backgroundColor: "#0d1117",
              color: "#8b949e",
              fontSize: 14,
              fontFamily: "'Inter', system-ui, sans-serif",
              cursor: "pointer",
              whiteSpace: "nowrap",
              transition: "color 0.15s",
            }}
          >
            {s}
            <ArrowRight size={13} style={{ opacity: 0.5 }} />
          </motion.button>
        ))}
      </motion.div>

      <RotatingQuote />
    </div>
  );
}
