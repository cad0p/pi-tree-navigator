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
import {
  estimateContextTokens,
  MAX_TAIL_RATIO,
  MIN_CALIBRATION_TAIL_TOKENS,
  MIN_TAIL_RATIO,
  measureTailRatio,
  piEstimateMessageTokens,
  requestContextTokens,
} from "./context-tokens.ts";

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
    // toolsAdded, while pre-1.0 hosts have no system case and return 0.
    // Asserting delegation keeps the unit suite correct on both hosts and
    // documents the divergence instead of pinning a false parity.
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

// -----------------------------------------------------------------------------
// pi-ai's own estimator (#107, round 3)
// -----------------------------------------------------------------------------

/**
 * pi-ai's real `estimateMessageTokens`. The package root does not export it
 * and the exports map blocks subpaths, so resolve the file beside the entry
 * (the same way the index tests reach `estimateContextTokens`).
 */
async function piAiEstimateMessageTokens(message: unknown): Promise<number> {
  const entry = import.meta.resolve("@earendil-works/pi-ai");
  const { estimateMessageTokens } = await import(
    new URL("./utils/estimate.js", entry).href
  );
  return estimateMessageTokens(message) as number;
}

describe("piEstimateMessageTokens", () => {
  const toolResult = (chars: number): AgentMessage =>
    ({
      role: "toolResult",
      toolCallId: "tc-1",
      toolName: "read",
      content: [{ type: "text", text: text(chars) }],
      isError: false,
      timestamp: 1,
    }) as unknown as AgentMessage;

  const system = (
    opts: {
      content?: number;
      sections?: Record<string, string | null>;
      toolsAdded?: unknown[];
      toolsRemoved?: unknown[];
    } = {},
  ): AgentMessage =>
    ({
      role: "system",
      content: [{ type: "text", text: text(opts.content ?? 0) }],
      sections: opts.sections,
      toolsAdded: opts.toolsAdded,
      toolsRemoved: opts.toolsRemoved,
      timestamp: 1,
    }) as unknown as AgentMessage;

  const assistantBlocks = (blocks: unknown[]): AgentMessage =>
    ({
      role: "assistant",
      content: blocks,
      api: "anthropic",
      provider: "test",
      model: "test",
      stopReason: "endTurn",
      timestamp: 1,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    }) as unknown as AgentMessage;

  it("matches pi-ai byte for byte on every message shape", async () => {
    const shapes: Array<[string, AgentMessage]> = [
      ["user text", user(400)],
      ["user string content", { role: "user", content: text(400) } as never],
      ["user text + image", userWithImage(400)],
      ["empty user", { role: "user", content: [] } as never],
      ["toolResult", toolResult(400)],
      ["assistant text", assistantBlocks([{ type: "text", text: text(400) }])],
      [
        "assistant text + thinking",
        assistantBlocks([
          { type: "text", text: text(400) },
          { type: "thinking", thinking: text(200) },
        ]),
      ],
      [
        "assistant tool call",
        assistantBlocks([
          {
            type: "toolCall",
            id: "c1",
            name: "read",
            arguments: { path: "a" },
          },
        ]),
      ],
      ["empty assistant", assistantBlocks([])],
      ["system content", system({ content: 400 })],
      [
        "system sections",
        system({ content: 400, sections: { appended: text(400), gone: null } }),
      ],
      [
        "system tool deltas",
        system({
          content: 400,
          toolsAdded: [{ name: "navigate_tree", description: text(200) }],
          toolsRemoved: [{ name: "gone", description: text(100) }],
        }),
      ],
      ["system empty", system()],
    ];
    for (const [name, message] of shapes) {
      assert.equal(
        piEstimateMessageTokens(message),
        await piAiEstimateMessageTokens(message),
        name,
      );
    }
  });

  it("pins the constants: chars/3.5 rounded up, 4800 chars per image", () => {
    assert.equal(piEstimateMessageTokens(user(400)), 115); // 400 / 3.5
    assert.equal(piEstimateMessageTokens(user(7)), 2); // 7 / 3.5 = 2 exactly
    assert.equal(piEstimateMessageTokens(user(4)), 2); // 1.14 -> 2
    // 400 text + 4800 image chars = 5200 -> 1486.
    assert.equal(piEstimateMessageTokens(userWithImage(400)), 1_486);
    assert.equal(piEstimateMessageTokens(assistantBlocks([])), 0);
    // A tool call counts its name and its JSON arguments, not its id.
    assert.equal(
      piEstimateMessageTokens(
        assistantBlocks([
          { type: "toolCall", id: "c1", name: "ab", arguments: { x: 1 } },
        ]),
      ),
      Math.ceil((2 + JSON.stringify({ x: 1 }).length) / 3.5),
    );
  });

  it("is 4/3.5 above pi-compaction's chars/4 estimate — the clamp gap (#107)", () => {
    const big = toolResult(48_000);
    assert.equal(estimateTokens(big), 12_000); // chars/4, the planner's tail
    assert.equal(piEstimateMessageTokens(big), 13_715); // chars/3.5, pi's clamp
    assert.equal(piEstimateMessageTokens(big), Math.ceil(48_000 / 3.5));
  });

  it("unserializable tool arguments degrade instead of throwing (pi's safeJsonStringify)", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const message = assistantBlocks([
      { type: "toolCall", id: "c1", name: "x", arguments: circular },
    ]);
    assert.equal(
      piEstimateMessageTokens(message),
      Math.ceil((1 + "[unserializable]".length) / 3.5),
    );
  });
});

// -----------------------------------------------------------------------------
// Tail calibration (#138)
// -----------------------------------------------------------------------------

describe("tail calibration (#138)", () => {
  // baseline 1000 (real), then a 10,000-token chars/4 tail.
  const chain = () => [
    user(4),
    assistant(8, { totalTokens: 1000 }),
    user(20_000),
    user(20_000),
  ];

  it("estimateContextTokens scales only the tail by the ratio", () => {
    assert.equal(estimateContextTokens(chain()), 11_000);
    assert.equal(estimateContextTokens(chain(), 1), 11_000);
    assert.equal(estimateContextTokens(chain(), 1.85), 1000 + 18_500);
    // Rounded to a whole token.
    assert.equal(estimateContextTokens(chain(), 1.00005), 11_001);
  });

  it("measures (real − baseline) / tail", () => {
    assert.equal(measureTailRatio(chain(), 1000 + 18_500), 1.85);
    assert.equal(measureTailRatio(chain(), 1000 + 7_000), 0.7);
  });

  it("clamps to [MIN_TAIL_RATIO, MAX_TAIL_RATIO]", () => {
    assert.equal(measureTailRatio(chain(), 1000 + 100_000), MAX_TAIL_RATIO);
    assert.equal(measureTailRatio(chain(), 1000 + 1), MIN_TAIL_RATIO);
  });

  it("is undefined when the real count does not exceed the baseline", () => {
    assert.equal(measureTailRatio(chain(), 1000), undefined);
    assert.equal(measureTailRatio(chain(), 500), undefined);
    assert.equal(measureTailRatio(chain(), Number.NaN), undefined);
  });

  it("is undefined below MIN_CALIBRATION_TAIL_TOKENS of tail", () => {
    const below = [
      assistant(8, { totalTokens: 1000 }),
      user((MIN_CALIBRATION_TAIL_TOKENS - 1) * 4),
    ];
    assert.equal(measureTailRatio(below, 50_000), undefined);
    const at = [
      assistant(8, { totalTokens: 1000 }),
      user(MIN_CALIBRATION_TAIL_TOKENS * 4),
    ];
    assert.equal(
      measureTailRatio(at, 1000 + MIN_CALIBRATION_TAIL_TOKENS * 2),
      2,
    );
  });

  it("with no usage anywhere the whole estimate is the tail", () => {
    assert.equal(measureTailRatio([user(40_000)], 20_000), 2);
  });

  it("requestContextTokens counts the input side only, never the reply's output", () => {
    assert.equal(
      requestContextTokens({
        input: 10,
        cacheRead: 200,
        cacheWrite: 3000,
        output: 40_000,
        totalTokens: 43_210,
      } as never),
      3210,
    );
    assert.equal(requestContextTokens({}), 0);
  });
});
