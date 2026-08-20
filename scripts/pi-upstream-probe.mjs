#!/usr/bin/env node
/**
 * pi-tree-navigator upstream reflection probe.
 *
 * Verifies the two remaining reflection points (plus their transitive
 * dependencies) still exist with the required shape in the installed
 * @earendil-works/pi-coding-agent / @earendil-works/pi-agent-core.
 *
 * Reflection points (after #14):
 *   1. `AgentSession.prototype.prompt` — must be a writable plain data
 *      property (not `#`-private, not a getter-only accessor). The
 *      extension stashes the original and replaces it with a wrapper.
 *   2. `session.agent.state.messages` — `agent` is pi-agent-core's Agent
 *      (exposed as a plain field on AgentSession), `agent.state` must be
 *      readable and `agent.state.messages` writable (plain fields, not
 *      `#`-private). The extension assigns `agent.state.messages = ...`.
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

let AgentSession, SessionManager, Agent;

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
} catch (e) {
  check("probe crashed", false, String(e.stack || e.message));
}

console.log(`\n${failures.length} of ${results.length} checks failed`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(failures.length ? 1 : 0);
