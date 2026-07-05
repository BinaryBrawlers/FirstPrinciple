/**
 * App.tsx — full-width chat layout.
 *
 * Renders ChatPanel as the sole full-width, full-height component,
 * satisfying Requirement 12.1: "The System SHALL render a full-width
 * chat panel as the primary interface."
 */
import { ChatPanel } from "./components/ChatPanel/ChatPanel";

export function App() {
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        margin: 0,
        padding: 0,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <ChatPanel />
    </div>
  );
}

export default App;
