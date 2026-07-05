import { useState } from "react";
import { useIngestion } from "./hooks/useIngestion";
import { IngestionBar } from "./components/IngestionBar";
import { ColdOpen } from "./components/ColdOpen";
import { ChatPanel } from "./components/ChatPanel/ChatPanel";
import "./index.css";

export function App() {
  const ingestion = useIngestion();
  const [mode, setMode] = useState<"teacher" | "interviewer">("teacher");

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100vw",
        height: "100vh",
        backgroundColor: "#0d1117",
      }}
    >
      <IngestionBar
        topics={ingestion.topics}
        activeTopic={ingestion.activeTopic}
        addTopic={ingestion.addTopic}
        setActiveTopic={ingestion.setActiveTopic}
      />
      {ingestion.activeTopic == null ? (
        <ColdOpen
          onSelectTopic={(t) => {
            ingestion.addTopic(t);
            ingestion.setActiveTopic(t);
          }}
        />
      ) : (
        <ChatPanel
          activeTopic={ingestion.activeTopic}
          isInputGated={ingestion.isInputGated}
          mode={mode}
          onModeSwitch={setMode}
        />
      )}
    </div>
  );
}

export default App;
