import { useState, useCallback, useRef } from "react";
import { postIngest } from "../api/client";

export interface TopicState {
  name: string;
  status: "ingesting" | "ready" | "error";
  stage: number;
}

export const INGESTION_STAGES = [
  "Fetching historical context...",
  "Cross-referencing sources...",
  "Building your learning path...",
  "Almost ready...",
] as const;

export interface UseIngestionResult {
  topics: Map<string, TopicState>;
  activeTopic: string | null;
  stages: typeof INGESTION_STAGES;
  addTopic: (name: string) => void;
  setActiveTopic: (name: string) => void;
  isInputGated: boolean;
}

export function useIngestion(): UseIngestionResult {
  const [topics, setTopics] = useState<Map<string, TopicState>>(new Map());
  const [activeTopic, setActiveTopicState] = useState<string | null>(null);
  // Map from topic name to interval ID so we can clear on completion
  const intervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  const addTopic = useCallback((name: string) => {
    const trimmed = name.trim();
    if (!trimmed) return;

    // If the topic already exists, just activate it
    setTopics((prev) => {
      if (prev.has(trimmed)) return prev;
      const next = new Map(prev);
      next.set(trimmed, { name: trimmed, status: "ingesting", stage: 0 });
      return next;
    });

    // Fire-and-forget the real backend ingest
    postIngest({ topic: trimmed }).catch(() => {
      // Non-fatal — the mock timer drives UI state regardless
    });

    // Clear any existing interval for this topic
    const existingInterval = intervalsRef.current.get(trimmed);
    if (existingInterval !== undefined) {
      clearInterval(existingInterval);
    }

    // Staged mock timer: advance through stages every ~3s, then mark ready
    let currentStage = 0;
    const totalStages = INGESTION_STAGES.length;

    const intervalId = setInterval(() => {
      currentStage += 1;

      if (currentStage >= totalStages) {
        // All stages done — mark ready
        clearInterval(intervalId);
        intervalsRef.current.delete(trimmed);
        setTopics((prev) => {
          const next = new Map(prev);
          const existing = next.get(trimmed);
          if (existing) {
            next.set(trimmed, { ...existing, status: "ready", stage: totalStages - 1 });
          }
          return next;
        });
      } else {
        setTopics((prev) => {
          const next = new Map(prev);
          const existing = next.get(trimmed);
          if (existing && existing.status === "ingesting") {
            next.set(trimmed, { ...existing, stage: currentStage });
          }
          return next;
        });
      }
    }, 3000);

    intervalsRef.current.set(trimmed, intervalId);
    setActiveTopicState(trimmed);
  }, []);

  const setActiveTopic = useCallback((name: string) => {
    setActiveTopicState(name);
  }, []);

  const activeState = activeTopic != null ? topics.get(activeTopic) : undefined;
  const isInputGated = activeState?.status === "ingesting";

  return {
    topics,
    activeTopic,
    stages: INGESTION_STAGES,
    addTopic,
    setActiveTopic,
    isInputGated,
  };
}
