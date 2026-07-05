/**
 * ModeToggle — switches the chat between Teacher mode and Interviewer mode.
 *
 * On click it:
 * 1. Calls the parent's `onModeSwitch` callback with the new mode.
 * 2. The parent (`ChatPanel`) is responsible for emitting the mode-switch
 *    event to the backend and generating a new session_id.
 *
 * Requirement 12.3 — The Chat Panel SHALL provide a mode toggle allowing the
 * user to switch between Teacher mode and Interviewer mode.
 */
interface ModeToggleProps {
  currentMode: "teacher" | "interviewer";
  disabled?: boolean;
  onModeSwitch: (newMode: "teacher" | "interviewer") => void;
}

const containerStyle: React.CSSProperties = {
  display: "flex",
  gap: 4,
  backgroundColor: "#f3f4f6",
  borderRadius: 8,
  padding: 4,
};

function buttonStyle(active: boolean): React.CSSProperties {
  return {
    padding: "6px 16px",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    fontFamily: "sans-serif",
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    backgroundColor: active ? "#2563eb" : "transparent",
    color: active ? "#fff" : "#4b5563",
    transition: "background-color 0.15s, color 0.15s",
  };
}

/**
 * Renders two toggle buttons: Teacher and Interviewer.
 * The active mode is highlighted; switching fires `onModeSwitch` with the new mode.
 *
 * Requirement 12.3
 */
export function ModeToggle({ currentMode, disabled = false, onModeSwitch }: ModeToggleProps) {
  const handleTeacher = () => {
    if (currentMode !== "teacher") {
      onModeSwitch("teacher");
    }
  };

  const handleInterviewer = () => {
    if (currentMode !== "interviewer") {
      onModeSwitch("interviewer");
    }
  };

  return (
    <div style={containerStyle} role="group" aria-label="Chat mode">
      <button
        style={buttonStyle(currentMode === "teacher")}
        aria-pressed={currentMode === "teacher"}
        disabled={disabled}
        onClick={handleTeacher}
      >
        🎓 Teacher
      </button>
      <button
        style={buttonStyle(currentMode === "interviewer")}
        aria-pressed={currentMode === "interviewer"}
        disabled={disabled}
        onClick={handleInterviewer}
      >
        🔍 Interviewer
      </button>
    </div>
  );
}
