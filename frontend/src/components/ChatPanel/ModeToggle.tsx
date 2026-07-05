import { BookOpen, Target } from "lucide-react";
import { motion } from "framer-motion";

interface ModeToggleProps {
  currentMode: "teacher" | "interviewer";
  disabled?: boolean;
  onModeSwitch: (newMode: "teacher" | "interviewer") => void;
}

const MODES = [
  { id: "teacher",     label: "Teacher",     Icon: BookOpen },
  { id: "interviewer", label: "Interviewer", Icon: Target   },
] as const;

export function ModeToggle({ currentMode, disabled = false, onModeSwitch }: ModeToggleProps) {
  return (
    <div
      style={{
        display: "flex",
        gap: 2,
        backgroundColor: "#0d1117",
        borderRadius: 10,
        padding: 3,
        border: "1px solid #30363d",
        position: "relative",
      }}
      role="group"
      aria-label="Chat mode"
    >
      {MODES.map(({ id, label, Icon }) => {
        const active = currentMode === id;
        const accent = id === "interviewer" ? "var(--accent-interviewer)" : "var(--accent-teacher)";
        return (
          <motion.button
            key={id}
            onClick={() => !active && !disabled && onModeSwitch(id)}
            disabled={disabled}
            aria-pressed={active}
            whileTap={!disabled && !active ? { scale: 0.95 } : {}}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "5px 14px",
              borderRadius: 7,
              border: "none",
              cursor: disabled ? "not-allowed" : active ? "default" : "pointer",
              fontSize: 13,
              fontWeight: active ? 600 : 400,
              backgroundColor: active ? accent : "transparent",
              color: active ? "#fff" : "#6e7681",
              transition: "background-color 0.2s, color 0.2s",
              opacity: disabled ? 0.5 : 1,
              letterSpacing: "0.01em",
              fontFamily: "inherit",
            }}
          >
            <Icon size={14} strokeWidth={active ? 2.5 : 2} />
            {label}
          </motion.button>
        );
      })}
    </div>
  );
}
