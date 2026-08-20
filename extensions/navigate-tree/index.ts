/**
 * navigate-tree — agent-callable session tree navigation.
 *
 * See README “Implementation notes” for the user-facing narrative
 * (Anthropic tool_use↔tool_result pairing, same-loop context refresh,
 * reflection bootstrap). This file-level JSDoc carries only
 * source-internal facts the README doesn't.
 *
 * `/tree`-visible artifacts (empirical):
 *   • Synthetic assistant message right after each `branch_summary`,
 *     with one tool_call sharing the in-flight `toolCallId`.
 *   • Dangling tool_use on the anchor entry whose original
 *     tool_results were cut off by the rewind. Anthropic accepts this
 *     — the dangling tool_use is buffered behind the branch_summary's
 *     user-text rendering and the API doesn't reject it. No walk-up
 *     logic at anchor time.
 *
 * Reflection bootstrap replicates pi's own slash-command line
 * verbatim (kept symmetric so a pi rename here surfaces as the
 * runtime warning rather than silently drifting):
 *
 *   this.agent.state.messages = this.sessionManager.buildSessionContext().messages;
 *
 * Risks of the reflection approach:
 *   • If pi switches either of the two fields this extension reads —
 *     `AgentSession.prototype.prompt` or `agent.state.messages` —
 *     to ES `#` private fields, this breaks fundamentally.
 *   • If pi renames or restructures any of these fields, this breaks.
 *   • Patches `AgentSession.prototype.prompt` globally on import; not
 *     reversible without a process restart; affects every session in
 *     the pi process, including sessions that never call
 *     `navigate_tree`.
 *
 * Verified against pi 0.81.0 / 0.83.0 / 0.84.2. In-loop context
 * refresh runs through the public `context` extension event (see
 * `buildContextMessages`), not `agent.prepareNextTurn*` reflection.
 */

import {
  estimateContextTokens,
  type StreamFn,
} from "@earendil-works/pi-agent-core";
import {
  AgentSession,
  buildSessionContext,
  collectEntriesForBranchSummary,
  type ExtensionAPI,
  generateBranchSummary,
  type ModelRegistry,
  type SessionEntry,
  type SessionManager,
  sessionEntryToContextMessages,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
  extractTextContent,
  formatContextDelta,
  formatPct1,
  formatWindow,
  isValidName,
  MAX_NAME_LENGTH,
  stripBranchSummaryBoilerplate,
  toOneLine,
} from "./helpers.ts";

const LABEL_PREFIX = "anchor:";

// ---------------------------------------------------------------------------
// Exported boundary constants below (MAX_SESSION_REFS, MAX_HINT_WALK_DEPTH,
// MIN_SUMMARY_FOCUS_LENGTH, MAX_SYNTHETIC_FOCUS_LENGTH).
//
// Stability: these are internal tunables. Exported only so the test suite
// can pin boundary cases by constant rather than literal. Re-tuning is
// NOT a semver-breaking change for this package — production callers
// should rely on the registered `navigate_tree` tool surface, not import
// these constants directly. The `__testHooks` JSDoc carries the same
// caveat for module-internal helpers.
// ---------------------------------------------------------------------------

// Cap on captured AgentSession refs across /new + /resume + /reload cycles.
// Worst case is ~one ref per long-lived session before reaping dead WeakRefs;
// 16 leaves headroom for the deepest session-fanout pattern observed (a few
// /resume cycles on top of a couple of /new cycles) without prematurely
// reaping a still-live session. Bump if the reaper fires while a session
// is still live.
export const MAX_SESSION_REFS = 16;
// Cap on parentId chain walks in `findLabelHint` (UX preview only;
// no need to walk to the root for a 50-char snippet).
export const MAX_HINT_WALK_DEPTH = 50;
// Floor on `summaryFocus` length (after trim) for `rewind`. The user's most
// recent instruction lives on the chain about to be collapsed; if the focus
// is shorter than this, it almost always elides that instruction (a terse
// "finish parser fix" is 17 chars and conveys nothing the next turn can
// act on). 20 is the empirical threshold below which the post-rewind turn
// reliably loses continuity — raising it forces more useful focus text
// without inviting verbosity.
export const MIN_SUMMARY_FOCUS_LENGTH = 20;
// Cap on the `summaryFocus` length stored in the synthetic assistant's
// arguments. The full focus is passed live to `generateBranchSummary`, so
// the summarizer always sees the original; we only need a trimmed copy in
// the synthetic's args because pi's `convertToLlm` re-emits the synthetic's
// toolCall block (including its arguments) on every subsequent turn until
// another rewind. Without a cap, a 100K-char focus inflates every later
// turn's input by ~100K chars indefinitely. 1024 chars is generous — well
// above empirically useful focus length, and the agent already saw the
// full focus string when it issued the rewind.
export const MAX_SYNTHETIC_FOCUS_LENGTH = 1024;
// Hint length cap for the per-row hint shown in `list` output. 50 chars
// fits one terminal column without wrapping in typical 80-column TUIs.
const LIST_HINT_MAX_LENGTH = 50;
// Hint length cap for the hint shown in the `anchor` response. The anchor
// response is a single block of prose (not a column-aligned table) so it
// can afford a longer hint than `list`'s per-row preview.
const ANCHOR_HINT_MAX_LENGTH = 60;
// padStart width for the percentage column in `list` output. The longest
// percent label is "100.0%" = 6 chars; "99.9%" = 5 chars covers the
// realistic worst case and keeps the column tight.
const LIST_PCT_COL_WIDTH = 5;
// padEnd width for the anchor-name column in `list` output. MAX_NAME_LENGTH
// is 40, but the typical kebab-case name is 8–20 chars; 28 keeps the
// hint column visible without truncating common names.
const LIST_LABEL_COL_WIDTH = 28;
const ORIG_PROMPT_KEY = Symbol.for("navigate-tree.orig-prompt");

// Two warnings: list-site (read-only path; warns about the next turn's
// context view) and rewind-site (wrote to disk; leads with that). Both
// suggest /reload first, then restart pi, in that order.
const REFLECTION_BOOTSTRAP_WARNING_LIST =
  "⚠ reflection bootstrap missing — anchors and rewinds still work, but the next assistant turn may snapshot pre-bootstrap context. Run `/reload` (or restart pi) to recover.";
const REFLECTION_BOOTSTRAP_WARNING_REWIND =
  "⚠ reflection bootstrap missing — the rewind landed on disk but the next assistant turn may still see the pre-rewind context. Run `/reload` (or restart pi) to recover.";

// ---------------------------------------------------------------------------
// Typed views over pi internals.
//
// pi-coding-agent doesn't expose `agent`, `state`, `prepareNextTurn`, or
// `sessionManager` on `AgentSession` in its public types, but they are plain
// (non-`#`-private) fields on the class. Each cast point is a fragility
// surface for pi version bumps; grouping them here makes the dependency
// surface explicit.
// ---------------------------------------------------------------------------

interface PiInternals {
  agent: {
    state: {
      messages: unknown[];
    };
  };
  sessionManager: SessionManager;
}

function asInternals(session: AgentSession): PiInternals {
  return session as unknown as PiInternals;
}

// =============================================================================
// Reflection bootstrap & in-loop refresh
// =============================================================================

const sessionInstances: WeakRef<AgentSession>[] = [];
let seenSessions = new WeakSet<AgentSession>();

function captureSession(session: AgentSession): void {
  if (seenSessions.has(session)) return;
  seenSessions.add(session);
  sessionInstances.push(new WeakRef(session));
  // Reap dead WeakRefs occasionally so the array doesn't grow unbounded
  // across /new and /resume cycles.
  if (sessionInstances.length > MAX_SESSION_REFS) {
    for (let i = sessionInstances.length - 1; i >= 0; i--) {
      if (!sessionInstances[i].deref()) sessionInstances.splice(i, 1);
    }
  }
}

function patchAgentSessionPrototype(): void {
  const proto = AgentSession.prototype as unknown as Record<
    PropertyKey,
    unknown
  >;
  // Stash the truly-original prompt the FIRST time we patch. On subsequent
  // /reloads the value is already there — we don't overwrite, we just read it
  // back so the new wrapper still calls the original (not a previous wrapper).
  if (!proto[ORIG_PROMPT_KEY]) {
    proto[ORIG_PROMPT_KEY] = proto.prompt;
  }
  const orig = proto[ORIG_PROMPT_KEY] as (...args: unknown[]) => unknown;

  // Always replace the wrapper, even if a previous load already patched. On
  // /reload the previous wrapper closes over the previous module's
  // `sessionInstances` — if we don't replace, captures land in the dead
  // module and reflection finds nothing.
  const patched = function (this: AgentSession, ...args: unknown[]) {
    captureSession(this);
    return orig.apply(this, args);
  };
  proto.prompt = patched;
}

function findOwningSession(sm: SessionManager): AgentSession | null {
  for (const ref of sessionInstances) {
    const s = ref.deref();
    if (s && asInternals(s).sessionManager === sm) {
      return s;
    }
  }
  return null;
}

/**
 * Resolve the provider's `streamSimple` for summarization routing via the
 * PUBLIC modelRegistry API (no reflection).
 *
 * Custom providers registered via `pi.registerProvider(name, { api:
 * <custom-id>, streamSimple })` are composed into the ModelRuntime provider
 * returned by `ctx.modelRegistry.getProvider(...)` — its `streamSimple`
 * dispatches to the extension handler (provider-composer `streamWith`).
 * Without passing it as `streamFn` to `generateBranchSummary`,
 * completeSummarization falls back to the pi-ai compat registry (builtin
 * apis only) and throws "No API provider registered for api: <custom-id>".
 */
function resolveProviderStreamFn(
  modelRegistry: ModelRegistry,
  providerId: string,
): { streamFn: StreamFn } | undefined {
  try {
    const provider = modelRegistry.getProvider(providerId);
    return provider?.streamSimple
      ? { streamFn: provider.streamSimple as StreamFn }
      : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Drop `null` header-deletion markers (pi 0.84+ `ProviderHeaders` can carry
 * `string | null` values). Mirrors pi's own `withoutDeletedHeaders` in
 * agent-session.js — `generateBranchSummary` expects `Record<string, string>`.
 */
function stripNullHeaders(
  headers: Record<string, string | null> | undefined,
): Record<string, string> | undefined {
  if (!headers) return undefined;
  const entries = Object.entries(headers).filter((entry) => entry[1] !== null);
  return entries.length > 0
    ? (Object.fromEntries(entries) as Record<string, string>)
    : undefined;
}

/**
 * Project the session tree's active branch into the messages the next LLM
 * call should see — the public-API replacement for the deleted
 * `prepareNextTurn` double-wrap (which rebuilt `messages` from
 * `sessionManager.buildSessionContext()` between every turn).
 *
 * Registered as a `context` extension handler: pi fires the `context` event
 * inside `streamAssistantResponse` before EVERY LLM call (wired through
 * `Agent.transformContext` → `runner.emitContext`), so this projection is
 * applied at the wire boundary itself rather than at a turn-boundary
 * snapshot. The loop's own `currentContext.messages` is never mutated — the
 * handler only replaces the clone that `convertToLlm` consumes.
 *
 * We always replace (no leaf-gating): every `appendMessage` advances the
 * session leaf, so a "leaf changed since last turn" heuristic would fire on
 * essentially every call anyway — and always-replace is byte-for-byte the
 * behavior of the old wrapper (per-turn rebuild from the tree).
 *
 * The parameter is typed structurally because `ReadonlySessionManager` is
 * NOT re-exported from the package barrel (only `SessionManager` and
 * `sessionEntryToContextMessages` are public); `ctx.sessionManager` at the
 * event site is `ReadonlySessionManager`, which has `buildContextEntries()`.
 */
export function buildContextMessages(sm: {
  buildContextEntries(): SessionEntry[];
}): ReturnType<typeof sessionEntryToContextMessages> {
  return sm
    .buildContextEntries()
    .flatMap((entry) => sessionEntryToContextMessages(entry));
}

function refreshAgentMessages(sm: SessionManager): boolean {
  // Manually replicate the agent-state refresh that pi's
  // commandCtx.navigateTree does after branchWithSummary. Returns true on
  // success, false if reflection couldn't find the owning AgentSession (in
  // which case the rewind is structurally complete on disk, but the next LLM
  // call will still see stale messages).
  const session = findOwningSession(sm);
  if (!session) return false;
  try {
    const sessionContext = sm.buildSessionContext();
    const agent = asInternals(session).agent;
    if (!agent?.state) return false;
    agent.state.messages = sessionContext.messages;
    return true;
  } catch {
    return false;
  }
}

// =============================================================================
// Helpers (extension-internal; pure helpers in ./helpers.ts)
// =============================================================================

function findLabeledEntry(
  sm: SessionManager,
  fullLabel: string,
): string | null {
  const path = sm.getBranch();
  for (let i = path.length - 1; i >= 0; i--) {
    if (sm.getLabel(path[i].id) === fullLabel) return path[i].id;
  }
  return null;
}

function estimateActiveBranchTokens(sm: SessionManager): number {
  return estimateContextTokens(sm.buildSessionContext().messages).tokens;
}

function estimateAtEntry(
  entries: SessionEntry[],
  entryId: string,
  byId: Map<string, SessionEntry>,
): number {
  return estimateContextTokens(
    buildSessionContext(entries, entryId, byId).messages,
  ).tokens;
}

/**
 * Walk parentId chain back from `fromId` and return a one-line preview of
 * the first entry that has meaningful text content. Branch summaries are
 * prefixed with `summary:` so the source is clear; user/assistant text is
 * shown as-is.
 */
function findLabelHint(
  sm: SessionManager,
  fromId: string,
  maxLen: number,
): string | null {
  let cur: string | null | undefined = fromId;
  let depth = 0;
  while (cur && depth < MAX_HINT_WALK_DEPTH) {
    const e = sm.getEntry(cur);
    if (!e) break;
    let text = "";
    let prefix = "";
    if (e.type === "branch_summary" && e.summary) {
      text = stripBranchSummaryBoilerplate(e.summary);
      prefix = "summary: ";
    } else if (e.type === "message") {
      const role = e.message.role;
      if (role === "user" || role === "assistant") {
        text = extractTextContent(e.message.content);
      }
    } else if (e.type === "custom_message") {
      text = extractTextContent(e.content);
    }
    const oneLine = toOneLine(text, maxLen - prefix.length);
    if (oneLine) return prefix + oneLine;
    cur = e.parentId;
    depth++;
  }
  return null;
}

/**
 * Build a synthetic assistant message containing a single tool_call whose id
 * matches the in-flight tool_call id. Appended after `branchWithSummary` so
 * the real tool_result lands paired with a matching tool_use.
 *
 * `usage` fields are zeroed except `totalTokens`, which is set to the
 * chain size measured BEFORE this synthetic is appended (the post-rewind
 * baseline that pi-agent-core's `estimateContextTokens` reads off the last
 * assistant). `stopReason: "toolUse"` survives Kiro's `normalizeMessages`
 * filter (which strips `error` / `aborted`); without it the synthetic would
 * be filtered out and the tool_result would re-orphan.
 */
function buildSyntheticAssistant(
  toolCallId: string,
  toolName: string,
  args: Record<string, unknown>,
  model: { api?: string; provider?: string; id?: string } | undefined,
  totalTokens: number,
) {
  return {
    role: "assistant" as const,
    content: [
      {
        type: "toolCall" as const,
        id: toolCallId,
        name: toolName,
        arguments: args,
      },
    ],
    api: model?.api ?? "unknown",
    provider: model?.provider ?? "unknown",
    model: model?.id ?? "unknown",
    stopReason: "toolUse" as const,
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  };
}

// =============================================================================
// Extension
// =============================================================================

interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  details: Record<string, unknown>;
  isError?: boolean;
}

function toolError(
  text: string,
  details: Record<string, unknown> = {},
): ToolResult {
  return {
    content: [{ type: "text", text }],
    details,
    isError: true,
  };
}

export default function (
  pi: ExtensionAPI,
  opts?: { summarize?: typeof generateBranchSummary },
) {
  // DI seam: tests inject a stub `summarize` to avoid hitting the real
  // model. Production callers (pi's extension loader) pass no second
  // argument, so this falls back to the real `generateBranchSummary`
  // import.
  const summarize = opts?.summarize ?? generateBranchSummary;
  patchAgentSessionPrototype();

  // Public context event: replace the wire messages with the session-tree
  // projection before every LLM call. This is the public-API replacement for
  // the deleted `agent.prepareNextTurnWithContext` per-turn refresh — it
  // fires inside `streamAssistantResponse` (via `Agent.transformContext` →
  // `runner.emitContext`), before EVERY LLM call including the turn right
  // after a mid-loop rewind, so the rewound chain (branch_summary +
  // synthetic assistant + tool_result) reaches the next API call.
  pi.on("context", (_event, ctx) => ({
    messages: buildContextMessages(ctx.sessionManager),
  }));

  pi.registerTool({
    name: "navigate_tree",
    label: "Navigate Tree",
    // Stateful: every action mutates SessionManager; concurrent calls would
    // race on `leafId` / `labelsById` and produce an undefined tree.
    executionMode: "sequential",
    description: `Long-session context management via the pi session tree. Anchor named milestones, then collapse work between them into a model-generated summary while preserving the full history on a sibling branch. Despite the verb, \`rewind\` does not restore prior state — it forks a sibling branch from the anchor and continues forward from a model-generated summary; the abandoned subtree is preserved on disk but no longer on the active path.

Operations (set the \`action\` parameter):
  • action='anchor', name='<milestone-name>': label the current point so a later rewind can target it. Use at the start of a stage you'll summarize (e.g. 'design-start', 'impl-start'). If the same name already exists on the active branch, the prior label is moved to the new leaf (no duplicates).
  • action='rewind', labelStart='<existing>', labelEnd='<new>': collapse work between labelStart and the current leaf into a branch_summary. The new summary entry is itself labeled with labelEnd, so you can chain rewinds.
  • action='list': show all named anchors on the active branch in chronological order, with cumulative context % at each anchor.

Both \`name\` (anchor) and \`labelEnd\` (rewind) write into the same anchor namespace — either becomes addressable as a future \`labelStart\`. \`list\` shows every anchor under the \`anchor:\` prefix, regardless of which action wrote it. Pi labels written via \`/label anchor:foo\` (manually or by other extensions) are also addressable here. Avoid the \`anchor:\` prefix in manually-set labels.

\`summaryFocus\` is required when \`action='rewind'\` (≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim). Calls without it are rejected. It's passed to pi's \`generateBranchSummary\` as \`customInstructions\`, biasing the summarizer LLM toward the agent's specified focus while it rewrites the collapsed work into pi's structured summary format. To preserve continuity, instruct the summarizer to keep: (1) the user's most recent message verbatim, (2) what's done in the collapsed segment, (3) what's left to do as a next action.`,
    promptSnippet:
      "Use to anchor named milestones and rewind the conversation tree to a prior point with a model-generated summary, for token-efficient long autonomous sessions.",
    // The schema is intentionally a flat `Type.Object` with everything-but-
    // `action` optional, with action-conditional required-ness enforced at
    // runtime in `execute`. Discriminated unions / property-level required-
    // when-action shapes break the Kiro/CodeWhisperer adapter, which forwards
    // `inputSchema.json` verbatim and 400s on non-`type: "object"` roots.
    // The runtime guards in `execute` provide the conditional-required
    // behavior the schema can't express.
    parameters: Type.Object({
      action: Type.Union(
        [Type.Literal("anchor"), Type.Literal("rewind"), Type.Literal("list")],
        {
          description:
            "Which operation to perform. 'anchor' labels the current point, 'rewind' collapses work between an anchor and the current leaf into a branch_summary, 'list' shows every anchor on the active branch.",
        },
      ),
      name: Type.Optional(
        Type.String({
          description: `Required when action='anchor'. Kebab-case label (max ${MAX_NAME_LENGTH} chars) for the milestone. If a label with this name already exists on the active branch, it is moved to the new leaf.`,
        }),
      ),
      labelStart: Type.Optional(
        Type.String({
          description: `Required when action='rewind'. Kebab-case name (max ${MAX_NAME_LENGTH} chars) of an existing anchor on the active branch — work between this anchor and the current leaf is summarized.`,
        }),
      ),
      labelEnd: Type.Optional(
        Type.String({
          description: `Required when action='rewind'. Kebab-case name (max ${MAX_NAME_LENGTH} chars) for the resulting branch_summary entry. If a label with this name already exists on the active branch, it is moved to the new entry (mirrors anchor's move-on-collision). Becomes addressable as a future labelStart.`,
        }),
      ),
      summaryFocus: Type.Optional(
        Type.String({
          description: `Required when action='rewind'. ≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim. Should encode (1) the user's most recent instruction verbatim, (2) what was done in the collapsed segment, (3) what's left to do as a next action.`,
        }),
      ),
    }),
    execute: async (toolCallId, params, signal, _onUpdate, ctx) => {
      const sm = ctx.sessionManager as SessionManager;
      const p = params as {
        action: "anchor" | "rewind" | "list";
        name?: string;
        labelStart?: string;
        labelEnd?: string;
        summaryFocus?: string;
      };

      // --- list ---
      if (p.action === "list") {
        const path = sm.getBranch();
        const allEntries = sm.getEntries();
        const byId = new Map<string, SessionEntry>();
        for (const e of allEntries) byId.set(e.id, e);
        const cw = ctx.model?.contextWindow ?? 0;
        const totalTokens = estimateActiveBranchTokens(sm);
        const reflectionOk = !!findOwningSession(sm);

        const lines: string[] = [];
        for (const e of path) {
          const lbl = sm.getLabel(e.id);
          if (lbl?.startsWith(LABEL_PREFIX)) {
            const name = lbl.slice(LABEL_PREFIX.length);
            const tokensAt = estimateAtEntry(allEntries, e.id, byId);
            const pct = formatPct1(tokensAt, cw).padStart(LIST_PCT_COL_WIDTH);
            const hint = findLabelHint(sm, e.id, LIST_HINT_MAX_LENGTH);
            const hintPart = hint ? `  (after: “${hint}”)` : "";
            lines.push(
              `  ${pct}  ${name.padEnd(LIST_LABEL_COL_WIDTH)}${hintPart}`,
            );
          }
        }

        const reflectionWarning = reflectionOk
          ? ""
          : ` · ${REFLECTION_BOOTSTRAP_WARNING_LIST}`;
        const header = `[list] · ${lines.length} label${lines.length === 1 ? "" : "s"} · ctx ${formatPct1(totalTokens, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}${reflectionWarning}`;
        const body = lines.length
          ? `Active labels (root → leaf):\n${lines.join("\n")}`
          : "No labels on the active branch.";
        return {
          content: [{ type: "text", text: `${header}\n\n${body}` }],
          details: {
            count: lines.length,
            contextTokens: totalTokens,
            contextWindow: cw,
            reflectionOk,
          },
        };
      }

      // --- anchor ---
      if (p.action === "anchor") {
        if (!isValidName(p.name)) {
          return toolError(
            `anchor requires \`name\` in kebab-case, max ${MAX_NAME_LENGTH} chars (e.g. 'impl-start').`,
          );
        }
        const leafId = sm.getLeafId();
        if (!leafId) {
          return toolError("No session entries yet — nothing to anchor.");
        }
        // Write the new label first, then clear the prior. If the second
        // setLabel throws, two labels of the same name briefly coexist on
        // the active branch — `findLabeledEntry` walks leaf→root and
        // returns the leaf-side match, so navigation behavior is correct
        // during the overlap. The pre-PR "no enforcement" semantics already
        // tolerated this. The reverse order (clear-then-set) was move-then-
        // lose under failure: a partial collapse left the active branch
        // with no anchor of the requested name at all.
        const fullLabel = LABEL_PREFIX + p.name;
        const prior = findLabeledEntry(sm, fullLabel);
        pi.setLabel(leafId, fullLabel);
        if (prior && prior !== leafId) {
          pi.setLabel(prior, undefined);
        }
        const cw = ctx.model?.contextWindow ?? 0;
        const tokensHere = estimateActiveBranchTokens(sm);
        const labelHint = findLabelHint(sm, leafId, ANCHOR_HINT_MAX_LENGTH);
        const positionLine = `${formatPct1(tokensHere, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}`;
        const hintLine = labelHint ? ` (after: “${labelHint}”)` : "";
        return {
          content: [
            {
              type: "text",
              text:
                `[anchor '${p.name}'] set at ${positionLine}${hintLine}\n\n` +
                `When you finish this stage, call: navigate_tree(action='rewind', labelStart='${p.name}', labelEnd='<milestone-name>', summaryFocus='<≥${MIN_SUMMARY_FOCUS_LENGTH}-char focus: latest user instruction + done + remaining>').`,
            },
          ],
          details: {
            label: p.name,
            entryId: leafId,
            contextTokens: tokensHere,
            labelHint,
            movedFromPriorEntry: prior && prior !== leafId ? prior : null,
          },
        };
      }

      // --- rewind ---
      if (!isValidName(p.labelStart)) {
        return toolError(
          `rewind requires \`labelStart\` in kebab-case, max ${MAX_NAME_LENGTH} chars.`,
        );
      }
      if (!isValidName(p.labelEnd)) {
        return toolError(
          `rewind requires \`labelEnd\` in kebab-case, max ${MAX_NAME_LENGTH} chars.`,
        );
      }
      if (
        !p.summaryFocus ||
        p.summaryFocus.trim().length < MIN_SUMMARY_FOCUS_LENGTH
      ) {
        const focusLen = p.summaryFocus?.trim().length ?? 0;
        return toolError(
          `\`summaryFocus\` must be ≥${MIN_SUMMARY_FOCUS_LENGTH} chars after trim (got ${focusLen}). The user's most recent instruction (which triggered this rewind) lives on the chain that's about to be collapsed — if summaryFocus doesn't preserve it, the post-rewind turn won't know what's left to do.\n\n` +
            `Include in summaryFocus:\n` +
            `  1. the user's most recent instruction verbatim,\n` +
            `  2. which parts have already been done in the work being collapsed,\n` +
            `  3. which parts remain unactioned.`,
        );
      }

      const target = findLabeledEntry(sm, LABEL_PREFIX + p.labelStart);
      if (!target) {
        return toolError(
          `No label '${p.labelStart}' on the active branch. Use action='list' to see available labels.`,
        );
      }

      const oldLeaf = sm.getLeafId();
      if (!oldLeaf || oldLeaf === target) {
        return toolError(
          `Already at '${p.labelStart}' — nothing to summarize.`,
        );
      }

      if (!ctx.model) {
        return toolError("No model configured for summarization.");
      }

      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
      if (!auth.ok) {
        return toolError(`Auth resolution failed: ${auth.error}`);
      }

      // The leaf at execute time is the assistant that just streamed the
      // rewind tool call. Its `usage.input` is the *minimum* of recent API
      // calls in this turn, so estimating from it understates what the
      // user just saw. Use the chain up to its parent (which has the prior
      // assistant's usage as baseline) so beforeTokens matches the value
      // `list` would have reported on the previous turn.
      const oldLeafEntry = sm.getEntry(oldLeaf);
      let beforeTokens: number;
      if (
        oldLeafEntry &&
        oldLeafEntry.type === "message" &&
        oldLeafEntry.message.role === "assistant" &&
        oldLeafEntry.parentId
      ) {
        const allEntries = sm.getEntries();
        const byId = new Map<string, SessionEntry>();
        for (const e of allEntries) byId.set(e.id, e);
        beforeTokens = estimateAtEntry(allEntries, oldLeafEntry.parentId, byId);
      } else {
        beforeTokens = estimateActiveBranchTokens(sm);
      }
      const contextWindow = ctx.model.contextWindow ?? 0;

      const { entries } = collectEntriesForBranchSummary(sm, oldLeaf, target);
      if (entries.length === 0) {
        return toolError(
          `No entries between leaf and '${p.labelStart}' — nothing to summarize.`,
        );
      }
      // Chained-rewind-no-turns guard: bail if the only message between
      // leaf and target matches our synthetic shape (single navigate_tree
      // toolCall block + stopReason "toolUse" + zero usage), meaning the
      // agent didn't append a real turn between rewinds. Keep only message
      // entries — label / compaction / branch_summary / etc. carry no
      // rewindable semantic content for this guard. The synthetic-shape
      // predicate avoids false-positives on real navigate_tree calls
      // (which have nonzero usage from the model).
      const messageEntries = entries.filter((e) => e.type === "message");
      if (messageEntries.length === 1) {
        const lone = messageEntries[0];
        if (lone.type === "message" && lone.message.role === "assistant") {
          const msg = lone.message as {
            content: Array<{ type?: string; name?: string }>;
            stopReason?: string;
            usage?: { input?: number; output?: number };
          };
          const block = msg.content[0];
          const isSyntheticShape =
            msg.stopReason === "toolUse" &&
            (msg.usage?.input ?? 0) === 0 &&
            (msg.usage?.output ?? 0) === 0;
          if (
            block &&
            block.type === "toolCall" &&
            block.name === "navigate_tree" &&
            isSyntheticShape
          ) {
            return toolError(
              `Already at synthetic boundary — no work to summarize. Append at least one turn between rewinds.`,
            );
          }
        }
      }

      const result = await summarize(entries, {
        // `auth.baseUrl` (OAuth/credential-derived endpoint, e.g.
        // githubCopilotOAuth) must be applied to the model, mirroring pi's
        // own `_getSummarizationRequestAuth` (`result.auth.baseUrl ?
        // { ...model, baseUrl: result.auth.baseUrl } : model`).
        model: auth.baseUrl
          ? { ...ctx.model, baseUrl: auth.baseUrl }
          : ctx.model,
        apiKey: auth.apiKey ?? "",
        headers: stripNullHeaders(auth.headers),
        ...(auth.env ? { env: auth.env } : {}),
        signal: signal ?? new AbortController().signal,
        customInstructions: p.summaryFocus,
        // Route through the composed provider's `streamSimple` (public
        // modelRegistry API) instead of the pi-ai compat registry.
        // Custom providers registered via `pi.registerProvider(name,
        // { api: <custom-id>, streamSimple })` are NOT visible to the
        // compat registry (which only knows builtin apis) — without this,
        // rewind fails with "No API provider registered for api:
        // <custom-id>" for any custom-api provider (e.g. commandcode 0.5.x
        // with api "commandcode-custom").
        ...(resolveProviderStreamFn(ctx.modelRegistry, ctx.model.provider) ??
          {}),
      });
      if (result.aborted) {
        return toolError("Summarization aborted.");
      }
      if (result.error || !result.summary) {
        return toolError(
          `Summarization failed: ${result.error ?? "no summary text"}`,
        );
      }

      // Move the tree.
      const summaryId = sm.branchWithSummary(target, result.summary, {
        readFiles: result.readFiles ?? [],
        modifiedFiles: result.modifiedFiles ?? [],
      });

      // Chain-validity invariants once `branchWithSummary` succeeds: a
      // synthetic must land on every path with toolCallId === this
      // in-flight call (so pi's appended tool_result pairs), and
      // stopReason: "toolUse" (survives Kiro's normalizeMessages filter
      // — see `buildSyntheticAssistant` JSDoc). The synthetic append
      // sits OUTSIDE the try so it runs exactly once regardless of
      // which earlier step threw. labelEnd write moves before clear,
      // mirroring `anchor`'s move-on-collision so duplicate anchors
      // can't survive a chained rewind.
      const fullLabelEnd = LABEL_PREFIX + p.labelEnd;
      let priorLabelEnd: ReturnType<typeof findLabeledEntry> = null;
      let tokensAtNewLeaf = 0;
      let originalErr: unknown;
      let salvageDetail = "";
      let failedStep:
        | "lookup"
        | "setLabelEnd"
        | "clearPrior"
        | "estimate"
        | null = null;
      try {
        failedStep = "lookup";
        priorLabelEnd = findLabeledEntry(sm, fullLabelEnd);
        failedStep = "setLabelEnd";
        pi.setLabel(summaryId, fullLabelEnd);
        // `summaryId` was freshly allocated by branchWithSummary above;
        // no pre-existing label can already point at it.
        if (priorLabelEnd) {
          failedStep = "clearPrior";
          pi.setLabel(priorLabelEnd, undefined);
        }

        // Compute afterTokens NOW — before we append the synthetic. This
        // captures the chain size at the new leaf (branch_summary) using
        // the prior real assistant's usage as the baseline.
        failedStep = "estimate";
        tokensAtNewLeaf = estimateActiveBranchTokens(sm);
        failedStep = null;
      } catch (err) {
        originalErr = err;
        // Best-effort retry of the specific failed step (pi.setLabel is
        // idempotent under re-application). Per-step recovery shape:
        //   - setLabelEnd: retry pi.setLabel(summaryId, fullLabelEnd).
        //   - clearPrior:  retry pi.setLabel(priorLabelEnd, undefined).
        //   - lookup / estimate: no retry — either prior state unknown
        //     or both labels already wrote; redundant retry would mask
        //     the real cause.
        if (failedStep === "setLabelEnd") {
          try {
            pi.setLabel(summaryId, fullLabelEnd);
          } catch (retryErr) {
            salvageDetail = `labelEnd retry failed: ${
              retryErr instanceof Error ? retryErr.message : String(retryErr)
            }`;
          }
        } else if (failedStep === "clearPrior" && priorLabelEnd) {
          try {
            pi.setLabel(priorLabelEnd, undefined);
          } catch (retryErr) {
            salvageDetail = `prior-clear retry failed: ${
              retryErr instanceof Error ? retryErr.message : String(retryErr)
            }`;
          }
        }
      }

      // Synthetic append: runs in BOTH the happy path and the salvage path.
      // If `originalErr` is set we use a degenerate synthetic
      // (totalTokens=0, since `tokensAtNewLeaf` may not have been computed).
      // The synthetic's matching toolCallId is the only structural
      // requirement for chain validity — pi's appended tool_result pairs
      // with this synthetic regardless of which earlier step threw.
      //
      // The full `summaryFocus` is already live in the LLM call to
      // `generateBranchSummary`; we only need a trimmed copy in the
      // synthetic's args (which pi will re-emit on every subsequent turn).
      // Truncate to MAX_SYNTHETIC_FOCUS_LENGTH so a long focus string
      // doesn't inflate every later turn indefinitely.
      const syntheticArgs: Record<string, unknown> = {
        ...(p as unknown as Record<string, unknown>),
      };
      if (
        typeof p.summaryFocus === "string" &&
        p.summaryFocus.length > MAX_SYNTHETIC_FOCUS_LENGTH
      ) {
        syntheticArgs.summaryFocus = `${p.summaryFocus.slice(0, MAX_SYNTHETIC_FOCUS_LENGTH)}… [truncated]`;
      }
      const syntheticMsg = buildSyntheticAssistant(
        toolCallId,
        "navigate_tree",
        syntheticArgs,
        ctx.model as
          | { api?: string; provider?: string; id?: string }
          | undefined,
        originalErr ? 0 : tokensAtNewLeaf,
      );
      const syntheticId = sm.appendMessage(syntheticMsg);

      // Refresh agent.state.messages so the next prompt() snapshot reflects
      // the rewound chain. Runs in both paths; `refreshAgentMessages`
      // already swallows internal throws, so it can't re-trigger salvage.
      const refreshed = refreshAgentMessages(sm);

      if (originalErr) {
        // Salvage path: synthetic landed (chain is valid), labelEnd retry
        // and refresh were best-effort. Re-throw the original error with
        // any salvage detail attached so the failure surfaces to the
        // agent and post-mortem reviewers can tell what was recovered.
        // Preserve the original via `Error.cause` (ES2022) so callers
        // doing `instanceof` checks against typed subclasses, or
        // post-mortem readers walking the cause chain, can recover the
        // original throw. Older runtimes silently ignore the options
        // bag, so this is forward-compatible without a feature gate.
        const baseMsg =
          originalErr instanceof Error
            ? originalErr.message
            : String(originalErr);
        throw new Error(
          salvageDetail ? `${baseMsg} (salvage: ${salvageDetail})` : baseMsg,
          { cause: originalErr },
        );
      }

      const afterTokens = tokensAtNewLeaf;

      return {
        content: [
          {
            type: "text",
            text:
              `[rewind '${p.labelStart}' → '${p.labelEnd}'] · ${formatContextDelta(beforeTokens, afterTokens, contextWindow)}\n\n` +
              `A branch_summary recording the work just collapsed has been appended to your context. Items under '### Done' are complete. Items under '### In Progress', '### Blocked', or '## Next Steps' are pending — execute them next without re-confirming with the user. Other branch_summary messages, if present, record earlier collapsed segments.` +
              (refreshed ? "" : `\n\n${REFLECTION_BOOTSTRAP_WARNING_REWIND}`),
          },
        ],
        details: {
          labelStart: p.labelStart,
          labelEnd: p.labelEnd,
          targetId: target,
          summaryId,
          syntheticAssistantId: syntheticId,
          collapsedEntries: entries.length,
          contextBefore: beforeTokens,
          contextAfter: afterTokens,
          contextWindow,
          agentMessagesRefreshed: refreshed,
          readFiles: result.readFiles ?? [],
          modifiedFiles: result.modifiedFiles ?? [],
        },
      };
    },
  });
}

/**
 * Non-stable testing-only hooks. **Do NOT import in production code.**
 *
 * The `__` prefix and individual member names are subject to change in any
 * release without a semver-major bump. Intended exclusively for hermetic
 * tests within this package; the hooks reach into module-internal state
 * (the `AgentSession.prototype` patch, the `seenSessions` WeakSet, the
 * `sessionInstances` array, the `prepareNextTurn` marker symbol) and are
 * not designed for external consumption.
 *
 * If you found this via `node_modules` archaeology, you're holding it
 * wrong — use the registered tool surface (`navigate_tree`) instead.
 */
export const __testHooks = {
  /**
   * Restore the original `AgentSession.prototype.prompt` (stashed by
   * `patchAgentSessionPrototype` under the `ORIG_PROMPT_KEY` symbol) and
   * drain the captured-session refs. Idempotent: a no-op if the patch was
   * never installed or has already been reset.
   */
  resetPrototype(): void {
    const proto = AgentSession.prototype as unknown as Record<
      PropertyKey,
      unknown
    >;
    const orig = proto[ORIG_PROMPT_KEY];
    if (typeof orig === "function") {
      proto.prompt = orig;
      delete proto[ORIG_PROMPT_KEY];
    }
    sessionInstances.length = 0;
    // WeakSet has no .clear(); rebind to a fresh instance so a test that
    // re-captures the SAME session identity post-reset isn't deduped by
    // stale state from the previous test's capture.
    seenSessions = new WeakSet();
  },
  /** Module-internal helpers exposed for hermetic unit tests. */
  buildSyntheticAssistant,
  findLabelHint,
  findLabeledEntry,
  buildContextMessages,
  refreshAgentMessages,
  captureSession,
  /** Read-only view of captured-session ref count for reaping assertions. */
  sessionRefCount(): number {
    return sessionInstances.length;
  },
  /**
   * The reflection-bootstrap-missing warning strings, split per site.
   * Exported so tests can pin the per-site verbatim wording (the `list`
   * site is read-only and uses the read-only phrasing; the `rewind` site
   * writes to disk and uses the write phrasing). Tests assert literal
   * containment at each site to catch drift.
   */
  REFLECTION_BOOTSTRAP_WARNING_LIST,
  REFLECTION_BOOTSTRAP_WARNING_REWIND,
};
