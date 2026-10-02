/**
 * Tests for the local `estimateContextTokens` helper.
 *
 * The helper is a drop-in replacement for the `pi-agent-core` export that
 * pi 1.0.0 removed; it delegates the per-message math to the host's public
 * `estimateTokens`, so the table below pins the *semantics* that matters
 * (usage baseline + trailing estimates, invalid-usage fall-through,
 * per-message message shapes) without claiming cross-version numeric
 * parity — see `context-tokens.ts` file JSDoc.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { estimateTokens } from "@earendil-works/pi-coding-agent";
import { estimateContextTokens } from "./context-tokens.ts";

// -----------------------------------------------------------------------------
// Fixtures — chars/4 means every token value below is an exact multiple of 4
// chars per token, so the expectations stay readable.
// -----------------------------------------------------------------------------

const text = (chars: number) => "x".repeat(chars);

function user(chars: number): AgentMessage {
  return {
    role: "user",
    content: [{ type: "text", text: text(chars) }],
    timestamp: 1,
  } as unknown as AgentMessage;
}

function userWithImage(textChars: number): AgentMessage {
  return {
    role: "user",
    content: [
      { type: "text", text: text(textChars) },
      // estimateTokens accounts every image block as 4800 chars → 1200 tokens.
      { type: "image", data: "AAAA", mimeType: "image/png" },
    ],
    timestamp: 1,
  } as unknown as AgentMessage;
}

function assistant(
  chars: number,
  opts: { totalTokens?: number; stopReason?: string } = {},
): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text: text(chars) }],
    api: "anthropic",
    provider: "test",
    model: "test",
    stopReason: opts.stopReason ?? "endTurn",
    timestamp: 1,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: opts.totalTokens ?? 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  } as unknown as AgentMessage;
}

// -----------------------------------------------------------------------------
// Estimator table
// -----------------------------------------------------------------------------

describe("estimateContextTokens", () => {
  it("empty message list is 0", () => {
    assert.equal(estimateContextTokens([]), 0);
  });

  it("no usage anywhere: sums estimateTokens over every message", () => {
    const chain = [
      user(4), //     1
      assistant(8), // 2
      user(400), //  100
    ];
    assert.equal(
      estimateContextTokens(chain),
      estimateTokens(chain[0]) +
        estimateTokens(chain[1]) +
        estimateTokens(chain[2]),
    );
    assert.equal(estimateContextTokens(chain), 103);
  });

  it("uses the newest valid assistant usage as baseline + trailing estimates", () => {
    const chain = [
      user(4), //                          ignored (before the baseline)
      assistant(8, { totalTokens: 1000 }), // baseline 1000, content not re-counted
      user(400), //                         trailing 100
      user(400), //                         trailing 100
    ];
    assert.equal(estimateContextTokens(chain), 1000 + 100 + 100);
  });

  it("picks the newest valid usage, not the oldest", () => {
    const chain = [
      assistant(4, { totalTokens: 1000 }), // older valid usage — must be ignored
      user(400), //                           covered by the newest baseline
      assistant(4, { totalTokens: 2000 }), // newest valid baseline
      user(400), //                           trailing: 100
    ];
    assert.equal(estimateContextTokens(chain), 2000 + 100);
  });

  it("skips aborted/error/zero-usage baselines and falls through to the previous valid usage", () => {
    const chain = [
      user(4), //                                       ignored
      assistant(4, { totalTokens: 1000 }), //           baseline 1000 (endTurn)
      user(400), //                                      trailing 100
      assistant(400, { totalTokens: 900, stopReason: "aborted" }), // trailing 100
      assistant(400, { totalTokens: 800, stopReason: "error" }), //   trailing 100
      assistant(400, { totalTokens: 0 }), // zero usage → trailing 100
      user(400), //                                      trailing 100
    ];
    // A regression that took the newest usage regardless of stopReason
    // would return 0 + 100, or 800 + 200 when only zero-usage was skipped.
    assert.equal(estimateContextTokens(chain), 1000 + 100 * 5);
  });

  it("counts toolResult / custom / branchSummary shapes", () => {
    const toolResult = {
      role: "toolResult",
      toolCallId: "tc-1",
      toolName: "read",
      content: [{ type: "text", text: text(400) }],
      isError: false,
      timestamp: 1,
    } as unknown as AgentMessage;
    const custom = {
      role: "custom",
      customType: "note",
      content: [{ type: "text", text: text(400) }],
      display: false,
      timestamp: 1,
    } as unknown as AgentMessage;
    const branchSummary = {
      role: "branchSummary",
      summary: text(400),
      fromId: "entry-1",
      timestamp: 1,
    } as unknown as AgentMessage;

    assert.equal(estimateContextTokens([toolResult]), 100);
    assert.equal(estimateContextTokens([custom]), 100);
    assert.equal(estimateContextTokens([branchSummary]), 100);
    assert.equal(
      estimateContextTokens([toolResult, custom, branchSummary]),
      300,
    );
  });

  it("system message: delegates to the host estimateTokens (0.84.2 scores 0; 1.0.0 counts content+sections+toolsAdded)", () => {
    const system = {
      role: "system",
      content: [{ type: "text", text: text(400) }],
      sections: { appended: text(400) },
      toolsAdded: [{ name: "navigate_tree", description: text(400) }],
      timestamp: 1,
    } as unknown as AgentMessage;

    // Host-semantics pin, not a cross-version number: pi 1.0.0's
    // estimateTokens counts a system message's content + sections +
    // toolsAdded, while 0.84.2 (this repo's devDeps) has no system case and
    // returns 0. Asserting delegation keeps the unit suite correct on both
    // hosts and documents the divergence instead of pinning a false parity.
    const hostSystemTokens = estimateTokens(system);
    assert.equal(estimateContextTokens([system]), hostSystemTokens);
    assert.equal(
      estimateContextTokens([user(400), system]),
      100 + hostSystemTokens,
    );
  });

  it("image blocks are accounted as chars in the trailing estimate", () => {
    // 400 text chars + 4800 image chars = 5200 → 1300 tokens.
    assert.equal(estimateContextTokens([userWithImage(400)]), 1300);
  });

  it("circular toolCall arguments throw (plain JSON.stringify — no safeJsonStringify guard)", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const message = assistant(4, { totalTokens: 0 });
    (message as unknown as { content: unknown }).content = [
      { type: "toolCall", id: "tc", name: "x", arguments: circular },
    ];
    assert.throws(() => estimateContextTokens([message]), TypeError);
  });

  it("image blocks after the baseline are counted in the trailing estimate", () => {
    const chain = [
      assistant(4, { totalTokens: 500 }),
      userWithImage(400), // trailing 1300
    ];
    assert.equal(estimateContextTokens(chain), 500 + 1300);
  });
});
