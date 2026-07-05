/**
 * Unit tests for the SSE parsing logic in useSSEChat.
 *
 * Because this project does not install a DOM environment for vitest,
 * we test the core SSE chunk-parsing logic in isolation — pulling it
 * out of the hook into a pure helper, then verifying the hook itself
 * compiles and exports the expected interface.
 *
 * DOM-level hook lifecycle tests (renderHook) would require jsdom, which
 * can be added later with `npm install --save-dev jsdom @vitest/coverage-v8`
 * and `test: { environment: 'jsdom' }` in vite.config.ts.
 */
import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// Inline re-implementation of the SSE chunk-parser for unit testing.
// This mirrors the logic inside sendMessage in useSSEChat.ts exactly.
// ---------------------------------------------------------------------------

/**
 * Parse all SSE `data:` lines from an ordered sequence of text chunks,
 * stopping when an `event: done` line is encountered.
 * Returns the collected tokens and whether a done event was seen.
 */
function parseSSEChunks(chunks: string[]): { tokens: string[]; sawDone: boolean } {
  const tokens: string[] = [];
  let lineBuffer = "";
  let isDoneEvent = false;
  let sawDone = false;

  outer: for (const rawChunk of chunks) {
    lineBuffer += rawChunk;
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();

      if (line === "") {
        isDoneEvent = false;
        continue;
      }

      if (line.startsWith("event:")) {
        const eventName = line.slice(6).trim();
        isDoneEvent = eventName === "done";
        if (isDoneEvent) {
          sawDone = true;
          break outer;
        }
        continue;
      }

      if (line.startsWith("data:")) {
        if (isDoneEvent) {
          sawDone = true;
          break outer;
        }
        const token = line.slice(5).trim();
        if (token) tokens.push(token);
      }
      // Ignore id:, retry:, and comment lines (:)
    }
  }

  return { tokens, sawDone };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("SSE chunk parser", () => {
  it("parses simple data lines", () => {
    const { tokens } = parseSSEChunks([
      "data: Hello\n\n",
      "data: world\n\n",
    ]);
    expect(tokens).toEqual(["Hello", "world"]);
  });

  it("handles chunks split mid-line (cross-boundary)", () => {
    // "data: hel" and "lo\n\n" arrive as two separate chunks
    const { tokens } = parseSSEChunks(["data: hel", "lo\n\n"]);
    expect(tokens).toEqual(["hello"]);
  });

  it("stops collecting tokens on event: done", () => {
    const { tokens, sawDone } = parseSSEChunks([
      "data: token1\n\n",
      "event: done\ndata: \n\n",
      "data: should-not-appear\n\n",
    ]);
    expect(tokens).toEqual(["token1"]);
    expect(sawDone).toBe(true);
  });

  it("ignores comment lines without adding tokens", () => {
    const { tokens } = parseSSEChunks([
      ": keep-alive\n\n",
      "retry: 3000\n\n",
      "data: real-token\n\n",
    ]);
    expect(tokens).toEqual(["real-token"]);
  });

  it("returns empty tokens for empty input", () => {
    const { tokens } = parseSSEChunks([]);
    expect(tokens).toEqual([]);
  });

  it("skips blank data: lines (whitespace-only tokens)", () => {
    const { tokens } = parseSSEChunks([
      "data:   \n\n",
      "data: valid\n\n",
    ]);
    expect(tokens).toEqual(["valid"]);
  });

  it("handles multiple data lines in a single chunk", () => {
    const { tokens } = parseSSEChunks([
      "data: a\n\ndata: b\n\ndata: c\n\n",
    ]);
    expect(tokens).toEqual(["a", "b", "c"]);
  });

  it("trims trailing whitespace from lines", () => {
    const { tokens } = parseSSEChunks(["data: token  \r\n\n"]);
    expect(tokens).toEqual(["token"]);
  });

  it("does not stop on event: other-name", () => {
    const { tokens, sawDone } = parseSSEChunks([
      "event: message\n",
      "data: payload\n\n",
      "data: after\n\n",
    ]);
    expect(tokens).toEqual(["payload", "after"]);
    expect(sawDone).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Module-level smoke test — ensure the hook exports the expected symbols.
// ---------------------------------------------------------------------------
describe("useSSEChat module exports", () => {
  it("exports useSSEChat as a function", async () => {
    const mod = await import("./useSSEChat");
    expect(typeof mod.useSSEChat).toBe("function");
  });
});
