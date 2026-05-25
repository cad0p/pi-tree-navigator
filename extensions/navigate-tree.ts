/**
 * navigate-tree v3 — agent-callable session tree navigation that doesn't
 * corrupt the chain.
 *
 * The original navigate-tree extension produced a structural break every time
 * a rewind ran: pi unconditionally appends the rewind tool's tool_result to
 * the new branch, but the matching tool_use lives in the assistant entry that
 * called navigate_tree, which is on the abandoned branch. The result is a
 * tool_result whose tool_use_id has no matching tool_use anywhere in the kept
 * chain, and Anthropic 400s the next API call with "Improperly formed
 * request". Kiro forwards that as `context_length_exceeded`, which is
 * misleading.
 *
 * Fix: after `sm.branchWithSummary(target)`, append a synthetic assistant
 * message whose single tool_call has the same id as the in-flight tool call
 * (the first arg to `execute`). When pi then writes the real tool_result, it
 * lands as a child of the synthetic assistant, with a matching id. The chain
 * is structurally valid.
 *
 * The synthetic assistant is visible in /tree as a single-tool_call assistant
 * message right after the branch_summary. It carries no semantic content
 * beyond pairing the tool_result with a tool_use.
 *
 * Note: anchoring at an assistant message that has its own tool_calls (the
 * normal case, since `anchor` itself runs as a tool_call) leaves a dangling
 * tool_use on that anchor entry once the rewind cuts off its original
 * tool_results. Empirically, Anthropic accepts this — the dangling tool_use
 * is buffered behind the branch_summary's user-text rendering and the API
 * doesn't reject it. So no walk-up logic is required at anchor time.
 *
 * Within the same agent loop
 * --------------------------
 * Pi's `Agent` snapshots `state.messages` once at the start of `prompt()`
 * and pushes new messages onto its own in-loop array. A rewind issued from
 * a tool execute therefore doesn't reduce the next API call's size until
 * the user sends a fresh prompt — every assistant turn within the same
 * `prompt()` keeps paying the pre-rewind context cost.
 *
 * To make rewind take effect immediately for subsequent turns within the
 * same prompt, we wire `agent.prepareNextTurn` from the prompt patch. The
 * agent loop calls `prepareNextTurn` between every turn boundary, and we
 * return a fresh context built from `sm.buildSessionContext()`. After a
 * rewind, the next assistant turn in the same loop sees the rewound chain.
 *
 * Reflection bootstrap
 * --------------------
 * Pi exposes `navigateTree` only on `ExtensionCommandContext`, which is only
 * available in slash-command handlers — never in tool execute. We avoid the
 * user-facing arming step by monkey-patching `AgentSession.prototype.prompt`
 * at extension load time to capture every AgentSession instance into a
 * WeakRef array. From the tool's execute, we walk the array to find the
 * AgentSession whose `.sessionManager` matches `ctx.sessionManager`, then
 * mutate `session.agent.state.messages` directly — replicating the line that
 * pi's own `navigateTree` does:
 *
 *   this.agent.state.messages = this.sessionManager.buildSessionContext().messages;
 *
 * Without that mutation, `branchWithSummary` updates the session JSONL and
 * the leaf, but `agent.state.messages` (the cached message list pi snapshots
 * for the next prompt) stays stale, so the rewind isn't visible to the next
 * LLM call.
 *
 * Risks of the reflection approach:
 *   • If pi switches `agent`, `state`, or `messages` to ES `#` private fields
 *     in a future version, this breaks fundamentally.
 *   • If pi renames or restructures these fields, this breaks.
 *   • Patches the AgentSession prototype globally.
 *
 * Verified against pi 0.75.5.
 */

import { estimateContextTokens } from "@earendil-works/pi-agent-core";
import {
  AgentSession,
  buildSessionContext,
  collectEntriesForBranchSummary,
  type ExtensionAPI,
  generateBranchSummary,
  type SessionEntry,
  type SessionManager,
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
} from "./_lib/navigate-tree-helpers.ts";

const LABEL_PREFIX = "anchor:";
const MAX_SESSION_REFS = 16;
const MAX_HINT_WALK_DEPTH = 50;
const MIN_SUMMARY_FOCUS_LENGTH = 20;
const PNT_MARKER = Symbol.for("navigate-tree.pnt-installed");
const ORIG_PROMPT_KEY = "__navTreeOrigPrompt";

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
      systemPrompt: string;
      messages: unknown[];
      tools: unknown[];
    };
    prepareNextTurn?: unknown;
  };
  sessionManager: SessionManager;
}

function asInternals(session: AgentSession): PiInternals {
  return session as unknown as PiInternals;
}

type PntResult = {
  context?: unknown;
  model?: unknown;
  thinkingLevel?: unknown;
};
type PntFn = (signal?: AbortSignal) => Promise<PntResult> | PntResult;
type MarkedPntFn = PntFn & { [PNT_MARKER]?: boolean; __prior?: PntFn };

// =============================================================================
// Reflection bootstrap & in-loop refresh
// =============================================================================

const sessionInstances: WeakRef<AgentSession>[] = [];
const seenSessions = new WeakSet<AgentSession>();

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
  const proto = AgentSession.prototype as unknown as Record<string, unknown>;
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
    installPrepareNextTurn(this);
    return orig.apply(this, args);
  };
  proto.prompt = patched;
}

/**
 * Wire `agent.prepareNextTurn` so the in-flight agent loop refreshes its
 * context from sessionManager between turns within the same prompt() call.
 * Without this, the loop snapshots agent.state.messages once at prompt start
 * and pushes new messages onto its own array — a rewind issued mid-loop
 * doesn't reduce the next API call's size until the user sends a new prompt.
 *
 * The Agent class's `createLoopConfig` dereferences `this.prepareNextTurn`
 * at the closure call site, so the value here is read at every turn boundary.
 * But it gates the closure on `this.prepareNextTurn` being truthy at config
 * creation — so we have to set this BEFORE prompt() runs, hence wiring it
 * from inside the prompt patch.
 */
function installPrepareNextTurn(session: AgentSession): void {
  const internals = asInternals(session);
  const agent = internals.agent;
  if (!agent) return;

  const sm = internals.sessionManager;

  // If the existing prepareNextTurn was installed by a previous load of THIS
  // extension, recover the chain it captured (its `__prior`) so we don't
  // strand other extensions' closures across /reload. Preserve any other
  // extension's prepareNextTurn so we compose with them.
  const existing = agent.prepareNextTurn as MarkedPntFn | undefined;
  const prior: PntFn | undefined =
    typeof existing === "function" && existing[PNT_MARKER]
      ? existing.__prior
      : (existing as PntFn | undefined);

  const next: MarkedPntFn = async (signal?: AbortSignal) => {
    let priorResult: PntResult | undefined;
    if (typeof prior === "function") {
      priorResult = await prior(signal);
    }
    return {
      context: {
        systemPrompt: agent.state.systemPrompt,
        messages: sm.buildSessionContext().messages,
        tools: agent.state.tools,
      },
      model: priorResult?.model,
      thinkingLevel: priorResult?.thinkingLevel,
    };
  };
  next[PNT_MARKER] = true;
  next.__prior = prior;
  agent.prepareNextTurn = next;
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
    if (!agent || !agent.state) return false;
    agent.state.messages = sessionContext.messages;
    return true;
  } catch {
    return false;
  }
}

// =============================================================================
// Helpers (extension-internal; pure helpers live in _lib/)
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
  maxLen = 60,
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
 * Notes on the `usage` shape:
 *   - Pi's TUI footer iterates ALL session entries and reads
 *     `usage.{input,output,cacheRead,cacheWrite,cost.total}` to compute
 *     cumulative session totals. We leave those at 0 so this synthetic
 *     contributes nothing to cumulative cost — which is correct, since no
 *     real LLM call happened.
 *   - pi-agent-core's `estimateContextTokens` finds the LAST assistant with
 *     a usage block and uses its `calculateContextTokens(usage)` value as
 *     the baseline (`usage.totalTokens || input + output + cacheRead + cacheWrite`).
 *     We set `totalTokens` to the post-rewind chain size so the baseline is
 *     accurate. (Picking it up directly via the synthetic also avoids
 *     stopReason "error"/"aborted", which the kiro provider's
 *     `normalizeMessages` strips before sending to the API — that would
 *     re-orphan the tool_result and trip Anthropic's validation.)
 *   - The `totalTokens` we set is the chain size measured *before* the
 *     synthetic itself is appended. Once appended, the synthetic's
 *     toolCall block adds ~50 tokens that aren't reflected in the baseline.
 *     Future estimates therefore understate the chain by ~50 tokens —
 *     negligible at typical anchor cadence.
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

export default function (pi: ExtensionAPI) {
  patchAgentSessionPrototype();

  pi.registerTool({
    name: "navigate_tree",
    label: "Navigate Tree",
    description:
      "Long-session context management via the pi session tree. Anchor named milestones, then collapse work between them into a model-generated summary while preserving the full history on a sibling branch.\n\n" +
      "Operations (set the `action` parameter):\n" +
      "  • action='anchor', name='<kebab-name>': label the current point so you can rewind back to here later. Use at the start of a stage you'll summarize (e.g. 'design-start', 'impl-start').\n" +
      "  • action='rewind', labelStart='<existing>', labelEnd='<new>': collapse work between labelStart and the current leaf into a branch_summary. The new summary entry is itself labeled with labelEnd, so you can chain rewinds.\n" +
      "  • action='list': show all named labels on the active branch in chronological order.\n\n" +
      "`summaryFocus` is REQUIRED on rewind. It's appended to pi's branch-summary prompt as `Additional focus: …`; the summarizer LLM then rewrites the collapsed work into pi's structured summary format, biased by this focus. To preserve continuity, instruct the summarizer to keep: (1) the user's most recent message verbatim, (2) what's done in the collapsed segment, (3) what's left to do as a next action.",
    promptSnippet:
      "Use to anchor named milestones and rewind the conversation tree to a prior point with a model-generated summary, for token-efficient long autonomous sessions.",
    parameters: Type.Object({
      action: Type.Union([
        Type.Literal("anchor"),
        Type.Literal("rewind"),
        Type.Literal("list"),
      ]),
      name: Type.Optional(Type.String()),
      labelStart: Type.Optional(Type.String()),
      labelEnd: Type.Optional(Type.String()),
      summaryFocus: Type.Optional(Type.String()),
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
          if (lbl && lbl.startsWith(LABEL_PREFIX)) {
            const name = lbl.slice(LABEL_PREFIX.length);
            const tokensAt = estimateAtEntry(allEntries, e.id, byId);
            const pct = formatPct1(tokensAt, cw).padStart(5);
            const hint = findLabelHint(sm, e.id, 50);
            const hintPart = hint ? `  “${hint}”` : "";
            lines.push(`  ${pct}  ${name.padEnd(28)}${hintPart}`);
          }
        }

        const header = `[list] · ${lines.length} label${lines.length === 1 ? "" : "s"} · ctx ${formatPct1(totalTokens, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}${reflectionOk ? "" : " · ⚠ reflection bootstrap missing"}`;
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
        pi.setLabel(leafId, LABEL_PREFIX + p.name);
        const cw = ctx.model?.contextWindow ?? 0;
        const tokensHere = estimateActiveBranchTokens(sm);
        const labelHint = findLabelHint(sm, leafId);
        const positionLine = `${formatPct1(tokensHere, cw)}${cw > 0 ? ` of ${formatWindow(cw)}` : ""}`;
        const hintLine = labelHint ? ` (at: “${labelHint}”)` : "";
        return {
          content: [
            {
              type: "text",
              text:
                `[anchor '${p.name}'] set at ${positionLine}${hintLine}\n\n` +
                `When you finish this stage, call: navigate_tree(action='rewind', labelStart='${p.name}', labelEnd='<milestone-name>').`,
            },
          ],
          details: {
            label: p.name,
            entryId: leafId,
            contextTokens: tokensHere,
            labelHint,
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
        return toolError(
          `rewind requires a non-trivial \`summaryFocus\`. The user's most recent instruction (which triggered this rewind) lives on the chain that's about to be collapsed — if summaryFocus doesn't preserve it, the post-rewind turn won't know what's left to do.\n\n` +
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
      // calls in this turn (extended-thinking strips earlier-turn thinking),
      // so estimating from it understates what the user just saw. Use the
      // chain *up to its parent* (which has the prior assistant's usage as
      // baseline) so beforeTokens matches the value `list` would have
      // reported on the previous turn.
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

      const result = await generateBranchSummary(entries, {
        model: ctx.model,
        apiKey: auth.apiKey ?? "",
        headers: auth.headers,
        signal: signal ?? new AbortController().signal,
        customInstructions: p.summaryFocus,
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
      pi.setLabel(summaryId, LABEL_PREFIX + p.labelEnd);

      // Compute afterTokens NOW — before we append the synthetic. This
      // captures the chain size at the new leaf (branch_summary) using the
      // prior real assistant's usage as the baseline.
      const tokensAtNewLeaf = estimateActiveBranchTokens(sm);

      // Inject a synthetic assistant whose tool_call id matches this in-flight
      // call. When pi appends our tool_result after this returns, it pairs
      // cleanly with this synthetic tool_use rather than the abandoned one.
      // Set the synthetic's `usage.totalTokens` to `tokensAtNewLeaf` so that
      // future `estimateContextTokens` calls (which will pick up the synthetic
      // as the last assistant) report the correct context size.
      const syntheticMsg = buildSyntheticAssistant(
        toolCallId,
        "navigate_tree",
        p as unknown as Record<string, unknown>,
        ctx.model as
          | { api?: string; provider?: string; id?: string }
          | undefined,
        tokensAtNewLeaf,
      );
      const syntheticId = sm.appendMessage(syntheticMsg);

      // Refresh agent.state.messages so the next prompt() snapshot reflects
      // the rewound chain.
      const refreshed = refreshAgentMessages(sm);

      const afterTokens = tokensAtNewLeaf;

      return {
        content: [
          {
            type: "text",
            text:
              `[rewind '${p.labelStart}' → '${p.labelEnd}'] · ${formatContextDelta(beforeTokens, afterTokens, contextWindow)}\n\n` +
              `A branch_summary recording the work just collapsed has been appended to your context. Items under '## Done' are complete. Items under '## Next Steps' are still pending — execute them next without re-confirming with the user. Other branch_summary messages, if present, record earlier collapsed segments.` +
              (refreshed
                ? ""
                : `\n\n⚠ Reflection failed — next prompt() may still snapshot stale messages.`),
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
