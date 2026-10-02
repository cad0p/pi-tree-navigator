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
 * Runtime import surface: the probe also extracts every runtime named
 * import the extension sources take from the host-aliased packages
 * (`@earendil-works/pi-coding-agent`, `@earendil-works/pi-agent-core`,
 * `@earendil-works/pi-tui`, `typebox`) and asserts each symbol exists on
 * the package entry actually resolved under `PROBE_NODE_MODULES`. That is
 * the class of break the repo's pinned dev-deps cannot see (pi 1.0.0
 * dropped pi-agent-core's `estimateContextTokens` exactly this way).
 *
 * Exit code: 0 = all good (no upstream break), 1 = something broke.
 */
import { existsSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

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
    /if \(event\.type === "message_end"\)[\s\S]{0,400}?else if \(event\.message\.role === "user" \|\|[\s\S]{0,300}?this\.sessionManager\.appendMessage\(event\.message\)/.test(
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

// ---------------------------------------------------------------------------
// 10. Extension runtime import surface.
//
// The extension loader aliases `@earendil-works/*` to the host process's
// own modules, so a runtime named import that upstream removed resolves to
// `undefined` on the host even though the repo's pinned dev-deps still
// typecheck green. Extract every runtime named import from the extension
// sources and assert it exists on the package entry the host would load.
// ---------------------------------------------------------------------------

const EXTENSION_SOURCE_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "extensions",
  "navigate-tree",
);
const IMPORT_SURFACE_PACKAGES = [
  "@earendil-works/pi-coding-agent",
  "@earendil-works/pi-agent-core",
  "@earendil-works/pi-tui",
  "typebox",
];
const IMPORT_STATEMENT_RE = /(?:^|[;}\)\]])\s*import\s+([\s\S]*?)\s+from\s+["']([^"']+)["']/gm;

/** `type`-prefixed clause or member: the only value-free import form. */
function clauseIsTypeOnly(text) {
  return /^type\b/.test(text);
}

/**
 * Runtime (value) named imports per package found in one source file.
 * Returns `found` (package -> names) and `expected` (packages whose matched
 * statement is a runtime import, recorded even when the member parser
 * yielded no names so the coverage guard can fail loudly). Type-only
 * specifiers are skipped: `import type { X }` statements are erased and can
 * never be `undefined` at runtime, and `type X` members are skipped inside
 * mixed imports.
 */
function extractRuntimeImports(source) {
  const found = new Map();
  const expected = new Set();
  for (const match of source.matchAll(IMPORT_STATEMENT_RE)) {
    const clause = match[1].trim();
    const specifier = match[2];
    if (!IMPORT_SURFACE_PACKAGES.includes(specifier)) continue;
    // `import type { ... }` and namespace imports carry no named values.
    if (clauseIsTypeOnly(clause) || clause.startsWith("*")) continue;
    // A runtime statement matched: the guard expects names for this package
    // even if the member parser below yields none.
    expected.add(specifier);
    const names = new Set();
    const brace = clause.indexOf("{");
    const head = (brace === -1 ? clause : clause.slice(0, brace))
      .trim()
      .replace(/,$/, "")
      .trim();
    if (head) names.add("default");
    if (brace !== -1) {
      const body = clause.slice(brace + 1, clause.lastIndexOf("}"));
      for (const raw of body.split(",")) {
        const member = raw.trim();
        if (!member || clauseIsTypeOnly(member)) continue;
        const imported = member.split(/\s+as\s+/)[0].trim();
        if (imported) names.add(imported);
      }
    }
    if (names.size === 0) continue;
    const bucket = found.get(specifier) ?? new Set();
    for (const name of names) bucket.add(name);
    found.set(specifier, bucket);
  }
  return { found, expected };
}

const SPECIFIER_BOUNDARY_CHARS = new Set([
  ";", "{", "}", "(", ")", "[", "]", '"', "'", "\n", "\r",
]);

/**
 * Blank `//` and block comments and the contents of string / template
 * literals (delimiters kept), so a commented-out or string-embedded
 * `import ... from "pkg"` can never match. Import/export module specifier
 * literals are the one exception: their path characters stay readable
 * because the extractor needs the package name, while every character that
 * cannot appear in a module specifier and could form a statement or quote
 * boundary is blanked, so an inner `'`/`;` cannot smuggle an import-like
 * run into the capture. The probe runs with builtins only (no TypeScript
 * parser in the workflow), so this is a small lexer over the states that
 * matter here.
 *
 * Regex literals are intentionally not masked: distinguishing `/.../` from
 * division with a heuristic can blank real code and create a silent miss —
 * the worse direction. A regex literal containing import-like text can only
 * cause a false-positive probe failure, never a silent miss.
 */
function maskCommentsAndLiterals(source) {
  let out = "";
  let i = 0;
  let state = "code";
  let keepLiteral = false;
  while (i < source.length) {
    const ch = source[i];
    const next = source[i + 1];
    if (state === "code") {
      if (ch === "/" && next === "/") { state = "line"; i += 2; continue; }
      if (ch === "/" && next === "*") { state = "block"; i += 2; continue; }
      if (ch === "'" || ch === '"' || ch === "`") {
        state = ch === "'" ? "single" : ch === '"' ? "double" : "template";
        // Module specifiers (never templates) stay readable for extraction.
        keepLiteral = ch !== "`" && /from\s*$/.test(out);
        out += ch;
        i += 1;
        continue;
      }
      out += ch; i += 1; continue;
    }
    if (state === "line") {
      if (ch === "\n") { state = "code"; out += ch; }
      i += 1; continue;
    }
    if (state === "block") {
      if (ch === "*" && next === "/") { state = "code"; i += 2; continue; }
      i += 1; continue;
    }
    // single | double | template
    const closing = state === "single" ? "'" : state === "double" ? '"' : "`";
    if (keepLiteral) {
      if (ch === "\\") {
        // An escape cannot appear in a module specifier; blank the pair.
        out += "  ";
        i += 2;
        continue;
      }
      if (ch === closing) {
        out += ch;
        state = "code";
        keepLiteral = false;
        i += 1;
        continue;
      }
      out += SPECIFIER_BOUNDARY_CHARS.has(ch) ? " " : ch;
      i += 1;
      continue;
    }
    if (ch === "\\") {
      // Blank the escape pair; keep a line continuation's newline.
      out += next === "\n" ? "\n" : "  ";
      i += 2;
      continue;
    }
    if (ch === closing) {
      out += ch;
      state = "code";
      i += 1;
      continue;
    }
    out += ch === "\n" ? "\n" : " ";
    i += 1;
  }
  return out;
}

/** Parse a package dir's package.json `exports["."]` / `main` root entry. */
function entryFromPackageJson(pkgDir) {
  const pkgJsonPath = path.join(pkgDir, "package.json");
  if (!existsSync(pkgJsonPath)) return null;
  let pkg;
  try {
    pkg = JSON.parse(readFileSync(pkgJsonPath, "utf8"));
  } catch {
    return null;
  }
  const rootExport =
    typeof pkg.exports === "string" ? pkg.exports : pkg.exports?.["."];
  let entry;
  if (typeof rootExport === "string") entry = rootExport;
  else if (rootExport && typeof rootExport === "object") {
    entry = rootExport.import ?? rootExport.default ?? rootExport.require ?? rootExport.node;
  }
  if (!entry && typeof pkg.main === "string") entry = pkg.main;
  if (!entry) return null;
  return path.join(pkgDir, entry);
}

/**
 * Resolve `pkgName` from one package dir the way a loader would:
 * `require.resolve` first (exports-map aware, `require`/`default`
 * conditions; typebox -> build/index.mjs), then the package.json
 * `exports["."]` / `main` fallback the pi packages need because they
 * declare only `import`/`types` and throw ERR_PACKAGE_PATH_NOT_EXPORTED.
 * That fallback is also why `resolveDist`'s hardcoded `dist/index.js`
 * cannot be reused.
 */
function resolveFromPackageDir(pkgDir, pkgName) {
  try {
    return createRequire(path.join(pkgDir, "resolve.cjs")).resolve(pkgName);
  } catch {}
  return entryFromPackageJson(pkgDir);
}

/** Walk the probe's node_modules candidates (hoisted, then nested). */
function resolveFromBaseWalk(pkgName, bases) {
  for (const base of bases) {
    const pkgDirs = [
      path.join(base, ...pkgName.split("/")),
      // npm can hoist a transitive host package under a direct dependency
      // (pi-tui lands in pi-coding-agent/node_modules in the probe install).
      path.join(
        base,
        "@earendil-works",
        "pi-coding-agent",
        "node_modules",
        ...pkgName.split("/"),
      ),
    ];
    for (const pkgDir of pkgDirs) {
      const entry = resolveFromPackageDir(pkgDir, pkgName);
      if (entry) return entry;
    }
  }
  return null;
}

/**
 * Resolve a host package's root entry the way the extension loader would:
 * authoritative to PROBE_NODE_MODULES when set. Anchor on the host's
 * pi-coding-agent install first, then resolve every other package from its
 * directory so a copy nested under `pi-coding-agent/node_modules` wins over
 * a hoisted one — that is the copy the extension loader aliases to. Only if
 * that fails fall back to the base-dir walk.
 */
function resolveHostPackageEntry(pkgName) {
  const bases = process.env.PROBE_NODE_MODULES
    ? [process.env.PROBE_NODE_MODULES]
    : [path.join(process.cwd(), "node_modules"), path.resolve(process.cwd(), "../node_modules")];

  if (pkgName !== "@earendil-works/pi-coding-agent") {
    const piEntry = resolveFromBaseWalk("@earendil-works/pi-coding-agent", bases);
    if (piEntry) {
      // realpath: pnpm's package dirs are symlinks into the virtual store;
      // anchoring on the symlink path misses the store's dependency
      // siblings (e.g. pi-coding-agent's own typebox), so the repo-root
      // copy would win. The loader sees the realpath copy.
      const fromPiDir = createRequire(
        path.join(path.dirname(realpathSync(piEntry)), "probe-resolve.cjs"),
      );
      try {
        return fromPiDir.resolve(pkgName);
      } catch {
        // pi packages expose only `import`/`types`, so require.resolve throws
        // ERR_PACKAGE_PATH_NOT_EXPORTED. Parse the package.json of the copy
        // Node would have picked from the same search paths.
        for (const searchPath of fromPiDir.resolve.paths(pkgName) ?? []) {
          const entry = entryFromPackageJson(path.join(searchPath, ...pkgName.split("/")));
          if (entry) return entry;
        }
      }
    }
  }
  return resolveFromBaseWalk(pkgName, bases);
}

let importSurface;
try {
  const files = readdirSync(EXTENSION_SOURCE_DIR)
    .filter((file) => file.endsWith(".ts") && !file.endsWith(".test.ts"))
    .sort();
  const used = new Map();
  const expectedRuntime = new Set();
  for (const file of files) {
    const source = maskCommentsAndLiterals(
      readFileSync(path.join(EXTENSION_SOURCE_DIR, file), "utf8"),
    );
    const { found, expected } = extractRuntimeImports(source);
    for (const pkgName of expected) expectedRuntime.add(pkgName);
    for (const [specifier, names] of found) {
      const bucket = used.get(specifier) ?? new Set();
      for (const name of names) bucket.add(name);
      used.set(specifier, bucket);
    }
  }
  const missing = [];
  let verified = 0;
  for (const pkgName of IMPORT_SURFACE_PACKAGES) {
    if (expectedRuntime.has(pkgName) && !used.get(pkgName)?.size) {
      missing.push(
        `${pkgName} (runtime import statement matched but no names extracted — clause parser may have drifted)`,
      );
    }
  }
  for (const pkgName of IMPORT_SURFACE_PACKAGES) {
    const names = used.get(pkgName);
    if (!names || names.size === 0) continue;
    const entry = resolveHostPackageEntry(pkgName);
    if (!entry) {
      missing.push(`${pkgName} (entry not found under the probe node_modules)`);
      continue;
    }
    const namespace = await import(pathToFileURL(entry).href);
    for (const name of [...names].sort()) {
      if (name in namespace) verified++;
      else missing.push(`${pkgName} -> ${name}`);
    }
  }
  importSurface = {
    ok: missing.length === 0,
    detail: missing.length
      ? `missing on host: ${missing.join(", ")}`
      : `${verified} runtime named import(s) across ${files.length} source file(s) verified`,
  };
} catch (e) {
  importSurface = {
    ok: false,
    detail: `threw: ${e instanceof Error ? e.message : String(e)}`,
  };
}
check(
  "extension runtime named imports exist on host packages",
  importSurface.ok,
  importSurface.detail,
);

console.log(`\n${failures.length} of ${results.length} checks failed`);
if (failures.length > 0) {
  for (const f of failures) console.log(`  FAILED: ${f.name} — ${f.detail}`);
}
process.exit(failures.length ? 1 : 0);
