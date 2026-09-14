#!/usr/bin/env node
/**
 * pi-tree-navigator upstream reflection probe.
 *
 * Verifies the two remaining reflection points (plus their transitive
 * dependencies) still exist with the required shape in the installed
 * @earendil-works/pi-coding-agent / @earendil-works/pi-agent-core.
 *
 * Reflection points (after #14, extended by #33):
 *   1. `AgentSession.prototype.prompt` — must be a writable plain data
 *      property (not `#`-private, not a getter-only accessor). The
 *      extension stashes the original and replaces it with a wrapper.
 *   2. `session.agent.state.messages` — `agent` is pi-agent-core's Agent
 *      (exposed as a plain field on AgentSession), `agent.state` must be
 *      readable and `agent.state.messages` writable (plain fields, not
 *      `#`-private). The extension assigns `agent.state.messages = ...`.
 *   3. `session.agent.state.tools` (#33) — the cache-preserving summary
 *      request passes the live tool array to the summarizer, so the
 *      accessor pair must exist and be readable.
 *   4. `session.agent.state.systemPrompt` (#33) — fallback source for the
 *      live system prompt when the public `ctx.getSystemPrompt()` is
 *      unavailable; a plain field on the mutable state object.
 *   5. `session.agent.thinkingBudgets` (#33) — plain field on the Agent;
 *      forwarded on the cache path when present.
 *   6. `SessionManager.prototype.getSessionId` (#33) — the summary joins
 *      the live session's cache namespace and reuses its routing id.
 *
 * Transitive dependencies:
 *   - `AgentSession` constructor assigns `this.sessionManager` (plain
 *     field) — `findOwningSession` compares it to `ctx.sessionManager`.
 *   - `SessionManager.prototype.buildSessionContext()` still exists.
 *   - `AgentSession` + `SessionManager` are still exported from the
 *     package root.
 *
 * Exit code: 0 = all good (no upstream break), 1 = something broke.
 */
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const results = [];
const failures = [];

function check(name, ok, detail = "") {
  results.push({ name, ok: !!ok, detail });
  if (!ok) failures.push({ name, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

// Resolve the package dist paths (ESM-only packages).
function resolveDist(pkgName, entry = "dist/index.js") {
  // PROBE_NODE_MODULES, when set, is authoritative: the workflow installs the
  // LATEST packages there, and the probe must inspect those — NOT the repo's
  // own node_modules (dev-deps) or anything require.resolve finds via cwd.
  // require.resolve from cwd can also short-circuit to the repo's version for
  // packages that DO export package.json (e.g. pi-agent-core), which would
  // make the probe pass trivially against a stale version.
  const bases = process.env.PROBE_NODE_MODULES
    ? [process.env.PROBE_NODE_MODULES]
    : [path.join(process.cwd(), "node_modules"), path.resolve(process.cwd(), "../node_modules")];
  for (const base of bases) {
    try {
      const scoped = pkgName.startsWith("@")
        ? path.join(base, ...pkgName.split("/"), "package.json")
        : path.join(base, pkgName, "package.json");
      const pkgJson = require.resolve(scoped);
      return path.join(path.dirname(pkgJson), entry);
    } catch {}
  }
  return null;
}

let AgentSession, SessionManager, Agent, SettingsManager;

try {
  const codingAgentDist = resolveDist("@earendil-works/pi-coding-agent");
  const coreDist = resolveDist("@earendil-works/pi-agent-core");
  if (!codingAgentDist || !coreDist) {
    check("deps installed", false, `codingAgent: ${codingAgentDist}, core: ${coreDist}`);
    process.exit(1);
  }

  const codingAgent = await import(pathToFileURL(codingAgentDist));
  AgentSession = codingAgent.AgentSession;
  SessionManager = codingAgent.SessionManager;
  SettingsManager = codingAgent.SettingsManager;

  const core = await import(pathToFileURL(coreDist));
  Agent = core.Agent;

  // --- 1. AgentSession.prototype.prompt ---
  check("AgentSession exported", typeof AgentSession === "function", typeof AgentSession);
  const proto = AgentSession.prototype;
  const promptDesc = Object.getOwnPropertyDescriptor(proto, "prompt");
  check(
    "prompt is own data property",
    !!promptDesc && "value" in promptDesc && !promptDesc.get,
    JSON.stringify(promptDesc ? { writable: promptDesc.writable, hasGet: !!promptDesc.get } : null),
  );
  check("prompt is writable", !promptDesc || promptDesc.writable !== false, "");
  check("prompt is a function", typeof proto.prompt === "function", typeof proto.prompt);

  // --- 2. sessionManager plain instance field (constructor assignment) ---
  // The d.ts says `readonly sessionManager: SessionManager`. Verify the dist
  // source assigns `this.sessionManager =` (plain field, not `this.#...`).
  const sessionSrc = readFileSync(
    path.join(path.dirname(codingAgentDist), "core/agent-session.js"),
    "utf8",
  );
  check(
    "sessionManager plain-field assignment",
    /this\.sessionManager\s*=\s*config\.sessionManager/.test(sessionSrc),
    sessionSrc.includes("this.#sessionManager") ? "FOUND #-private" : "plain this.sessionManager = found",
  );

  // --- 3. SessionManager.buildSessionContext ---
  check("SessionManager exported", typeof SessionManager === "function", typeof SessionManager);
  check(
    "buildSessionContext is prototype method",
    typeof SessionManager?.prototype?.buildSessionContext === "function",
    typeof SessionManager?.prototype?.buildSessionContext,
  );
  // The extension calls it with NO args and expects { messages } back
  // (index.ts: refreshAgentMessages). A signature change (required param)
  // or a return-shape change would pass the existence check above but
  // throw/break at runtime — so call it for real.
  let bscResult = null;
  let bscThrew = null;
  try {
    bscResult = SessionManager.prototype.buildSessionContext.call({
      getEntries: () => [],
      leafId: null,
      byId: new Map(),
    });
  } catch (e) {
    bscThrew = e instanceof Error ? e.message : String(e);
  }
  check(
    "buildSessionContext callable with no args → { messages }",
    !bscThrew && bscResult && Array.isArray(bscResult.messages),
    bscThrew ? `threw: ${bscThrew}` : Array.isArray(bscResult?.messages) ? "returns messages array" : "no messages array",
  );

  // --- 4. Agent.state accessor + messages writable ---
  check("Agent exported (pi-agent-core)", typeof Agent === "function", typeof Agent);
  const stateDesc = Object.getOwnPropertyDescriptor(Agent.prototype, "state");
  check(
    "Agent.prototype.state readable",
    !!stateDesc && !!stateDesc.get,
    JSON.stringify(stateDesc ? { hasGet: !!stateDesc.get, hasValue: "value" in stateDesc } : null),
  );

  // agent.state.messages: the extension ASSIGNS it. In pi-agent-core the
  // state setter copies the array (see agent.js createMutableAgentState:
  // `set messages(nextMessages) { messages = nextMessages.slice() }`).
  // Verify the accessor pair exists in source (stronger than a bare
  // `includes("messages")` — catches getter-only messages, which would
  // make the assignment throw in strict mode / silently no-op).
  const agentSrc = readFileSync(
    path.join(path.dirname(coreDist), "agent.js"),
    "utf8",
  );
  check(
    "state.messages writable (source: set messages accessor)",
    /set\s+messages\s*\(/.test(agentSrc) &&
      /get\s+messages\s*\(/.test(agentSrc) &&
      !agentSrc.includes("this.#state"),
    agentSrc.includes("this.#state")
      ? "#-private state"
      : /set\s+messages\s*\(/.test(agentSrc)
        ? "set messages accessor found"
        : "set messages accessor MISSING",
  );

  // --- 5. Agent.state.tools + systemPrompt (cache-preserving request, #33) ---
  // The summary request mirrors the live tool array and system prompt; both
  // are read off the mutable agent state object created by
  // createMutableAgentState. Verify the accessor pair for tools and the plain
  // systemPrompt field, and that the state is not #-private.
  check(
    "state.tools accessor pair (source: get/set tools)",
    /get\s+tools\s*\(/.test(agentSrc) && /set\s+tools\s*\(/.test(agentSrc),
    /get\s+tools\s*\(/.test(agentSrc) && /set\s+tools\s*\(/.test(agentSrc)
      ? "tools accessor pair found"
      : "tools accessor pair MISSING",
  );
  check(
    "state.systemPrompt plain field",
    /systemPrompt:\s*initialState\?\.systemPrompt/.test(agentSrc),
    /systemPrompt:\s*initialState\?\.systemPrompt/.test(agentSrc)
      ? "plain systemPrompt field found"
      : "systemPrompt field MISSING",
  );

  // --- 6. Agent.thinkingBudgets (plain field, #33) ---
  check(
    "agent.thinkingBudgets plain field",
    /this\.thinkingBudgets\s*=\s*runtimeOptions\.thinkingBudgets/.test(
      agentSrc,
    ),
    /this\.thinkingBudgets\s*=\s*runtimeOptions\.thinkingBudgets/.test(agentSrc)
      ? "plain this.thinkingBudgets = found"
      : "thinkingBudgets assignment MISSING",
  );

  // --- 7. SessionManager.getSessionId (#33) ---
  check(
    "getSessionId is prototype method",
    typeof SessionManager?.prototype?.getSessionId === "function",
    typeof SessionManager?.prototype?.getSessionId,
  );

  // --- 8. SettingsManager.getShowCacheMissNotices (TUI cache-notice gate) ---
  // The extension reads this reflectively off the captured AgentSession to
  // gate the TUI cache-miss warning. pi defaults it to false; absence would
  // silently disable the notice (or, on a non-optional read, throw).
  check(
    "SettingsManager exported",
    typeof SettingsManager === "function",
    typeof SettingsManager,
  );
  check(
    "getShowCacheMissNotices is prototype method",
    typeof SettingsManager?.prototype?.getShowCacheMissNotices === "function",
    typeof SettingsManager?.prototype?.getShowCacheMissNotices,
  );

  // --- 9. Assistant persisted before tool execution (#37 solo-batch guard) ---
  // The #37 guard reads the active branch to find the in-flight assistant
  // (the one declaring the rewind tool call) BEFORE any mutation. That read
  // is only sound because pi:
  //   (a) emits the assistant `message_end` — and AgentSession persists it
  //       via `sessionManager.appendMessage` — before the loop calls
  //       `executeToolCalls` → emits `tool_execution_start`;
  //   (b) awaits listeners inside the event dispatch, so persistence
  //       completes before the tool body runs.
  // Verify (a)+(b) behaviorally with a real Agent + stub streamFn: a listener
  // that appends on assistant message_end must have the append visible from
  // the tool body. Then pin the AgentSession persistence site and its
  // subscription from source. A future pi reorder must fail this probe
  // loudly instead of silently reopening #37.
  let batchOrdering = { ok: false, detail: "not run" };
  try {
    const ZERO_USAGE = {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    };
    const batchEvents = [];
    const persisted = new Set();
    let toolSawPersistedAssistant = true;
    let streamCalls = 0;
    const makeAssistant = (content, stopReason) => ({
      role: "assistant",
      content,
      api: "anthropic",
      provider: "probe",
      model: "probe",
      stopReason,
      timestamp: Date.now(),
      usage: ZERO_USAGE,
    });
    const probeTool = (name) => ({
      name,
      label: name,
      description: "probe",
      parameters: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      executionMode: "sequential",
      execute: async (toolCallId) => {
        batchEvents.push(`execute:${toolCallId}`);
        if (!persisted.has(toolCallId)) toolSawPersistedAssistant = false;
        return { content: [{ type: "text", text: "ok" }], details: {} };
      },
    });
    const batchAgent = new Agent({
      streamFn: async () => {
        streamCalls++;
        const message =
          streamCalls === 1
            ? makeAssistant(
                [
                  {
                    type: "toolCall",
                    id: "probe-tc-a",
                    name: "probe_batch_a",
                    arguments: {},
                  },
                  {
                    type: "toolCall",
                    id: "probe-tc-b",
                    name: "probe_batch_b",
                    arguments: {},
                  },
                ],
                "toolUse",
              )
            : makeAssistant([{ type: "text", text: "done" }], "endTurn");
        return {
          async *[Symbol.asyncIterator]() {},
          async result() {
            return message;
          },
        };
      },
      initialState: {
        systemPrompt: "probe",
        model: { id: "probe", provider: "probe", api: "anthropic" },
        thinkingLevel: "off",
        messages: [],
        tools: [probeTool("probe_batch_a"), probeTool("probe_batch_b")],
      },
    });
    batchAgent.subscribe((event) => {
      if (event.type === "message_end" && event.message.role === "assistant") {
        for (const block of event.message.content) {
          if (block.type === "toolCall") persisted.add(block.id);
        }
        batchEvents.push("persist:assistant");
      } else if (event.type === "tool_execution_start") {
        batchEvents.push(`tool_execution_start:${event.toolCallId}`);
      }
    });
    await batchAgent.prompt("go");
    const firstPersist = batchEvents.indexOf("persist:assistant");
    const firstToolStart = batchEvents.findIndex((e) =>
      e.startsWith("tool_execution_start:"),
    );
    const ok =
      firstPersist !== -1 &&
      firstToolStart !== -1 &&
      firstPersist < firstToolStart &&
      toolSawPersistedAssistant;
    batchOrdering = {
      ok,
      detail: `${batchEvents.join(" -> ")}${toolSawPersistedAssistant ? "" : " [tool ran before persist]"}`,
    };
  } catch (e) {
    batchOrdering = {
      ok: false,
      detail: `threw: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
  check(
    "assistant message_end (persisted) precedes tool_execution_start (#37)",
    batchOrdering.ok,
    batchOrdering.detail,
  );
  const sessionPersistsOnMessageEnd =
    /if\s*\(event\.type === "message_end"\)[\s\S]{0,800}?this\.sessionManager\.appendMessage\(event\.message\)/.test(
      sessionSrc,
    );
  const sessionSubscribes = /this\.agent\.subscribe\(this\._handleAgentEvent\)/.test(
    sessionSrc,
  );
  check(
    "AgentSession persists message_end payloads via sessionManager.appendMessage (#37)",
    sessionSubscribes && sessionPersistsOnMessageEnd,
    sessionSubscribes && sessionPersistsOnMessageEnd
      ? "subscription + message_end appendMessage found"
      : `subscription: ${sessionSubscribes ? "found" : "MISSING"}, message_end persistence: ${sessionPersistsOnMessageEnd ? "found" : "MISSING"}`,
  );
} catch (e) {
  check("probe crashed", false, String(e.stack || e.message));
}

console.log(`\n${failures.length} of ${results.length} checks failed`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(failures.length ? 1 : 0);
