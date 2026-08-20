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
import { fileURLToPath } from "node:url";
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
  // 1. try require.resolve on package.json (CJS packages)
  try {
    const pkgJson = require.resolve(`${pkgName}/package.json`);
    const dir = path.dirname(pkgJson);
    return path.join(dir, entry);
  } catch {}
  // 2. walk node_modules for the package dir (pnpm/ESM-only)
  for (const base of [
    path.join(process.cwd(), "node_modules"),
    path.resolve(process.cwd(), "../node_modules"),
    ...(process.env.PROBE_NODE_MODULES ? [process.env.PROBE_NODE_MODULES] : []),
  ]) {
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

  // --- 4. Agent.state accessor + messages writable ---
  check("Agent exported (pi-agent-core)", typeof Agent === "function", typeof Agent);
  const stateDesc = Object.getOwnPropertyDescriptor(Agent.prototype, "state");
  check(
    "Agent.prototype.state readable",
    !!stateDesc && !!stateDesc.get,
    JSON.stringify(stateDesc ? { hasGet: !!stateDesc.get, hasValue: "value" in stateDesc } : null),
  );

  // agent.state.messages: the extension ASSIGNS it. In pi-agent-core the
  // state setter copies the array (see agent.js: "Assigning state.tools or
  // state.messages copies the provided top-level array"). Verify the setter
  // accepts messages assignment — the state accessor returns an object with
  // settable messages. Source check on agent.js:
  const agentSrc = readFileSync(
    path.join(path.dirname(coreDist), "agent.js"),
    "utf8",
  );
  check(
    "state.messages writable (source: setter copies)",
    /messages/.test(agentSrc) && !agentSrc.includes("this.#state"),
    agentSrc.includes("this.#state") ? "#-private state" : "plain state field",
  );
} catch (e) {
  check("probe crashed", false, String(e.stack || e.message));
}

function pathToFileURL(p) {
  return new URL(`file://${p}`);
}

console.log(`\n${failures.length} of ${results.length} checks failed`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(failures.length ? 1 : 0);
