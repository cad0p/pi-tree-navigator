/**
 * Tests for the two-layer tree-navigator.json config loader.
 *
 * Node test runner. Run with: pnpm test
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files — so this test file is not loaded as a separate extension.
 * The loader's `readFile` is injected, so these tests touch no real fs and
 * pin the layer/merge/warning matrix without a session.
 *
 * Two fields with deliberately different failure semantics (issue #55):
 *   - `rewindHintReserveTokens`: field-level invalid disables the hint for the
 *     session with NO cross-layer fallback.
 *   - `anchorStartAfterTurns`: field-level invalid warns and falls through
 *     to the other layer's valid value, else the default 2.
 *
 * The removed `rewindHintAtPercent` key (0.3.0) only warns: it never
 * enables, disables, or invalidates the hint.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  type HintField,
  loadTreeNavigatorConfig,
  mergeTreeNavigatorConfig,
  type ParsedLayer,
  parseTreeNavigatorConfig,
  REWIND_HINT_MAX_RESERVE,
  REWIND_HINT_MIN_RESERVE,
  resolveRewindHintReserve,
  START_ANCHOR_DEFAULT_TURNS,
  START_ANCHOR_MAX_TURNS,
  START_ANCHOR_MIN_TURNS,
  type StartAnchorField,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";

const AGENT_DIR = "/agent";
const GLOBAL_PATH = `/agent/${TREE_NAVIGATOR_CONFIG_FILENAME}`;
const PROJECT_PATH = `/project/.pi/${TREE_NAVIGATOR_CONFIG_FILENAME}`;

// Byte-exact warning copy. Interpolating the same constants the source
// interpolates keeps the pins honest if a range is ever re-tuned.
const WARN_COULD_NOT_READ_GLOBAL = `navigate_tree: could not read global config at ${GLOBAL_PATH} — that layer was ignored.`;
const WARN_COULD_NOT_READ_PROJECT = `navigate_tree: could not read project config at ${PROJECT_PATH} — that layer was ignored.`;
const WARN_INVALID_JSON_GLOBAL = `navigate_tree: invalid JSON in global config at ${GLOBAL_PATH} — that layer was ignored.`;
const WARN_INVALID_JSON_PROJECT = `navigate_tree: invalid JSON in project config at ${PROJECT_PATH} — that layer was ignored.`;
const WARN_NOT_OBJECT_GLOBAL = `navigate_tree: global config at ${GLOBAL_PATH} must be a JSON object — that layer was ignored.`;
const WARN_NOT_OBJECT_PROJECT = `navigate_tree: project config at ${PROJECT_PATH} must be a JSON object — that layer was ignored.`;
const WARN_INVALID_HINT_GLOBAL = `navigate_tree: invalid rewindHintReserveTokens in global config at ${GLOBAL_PATH} — expected integer ${REWIND_HINT_MIN_RESERVE}-${REWIND_HINT_MAX_RESERVE} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;
const warnLegacyPercent = (layerName: string, path: string) =>
  `navigate_tree: rewindHintAtPercent in ${layerName} config at ${path} was removed in 0.3.0 and is ignored — use rewindHintReserveTokens (tokens left in the window when the hint fires, e.g. 50000).`;
const WARN_INVALID_HINT_PROJECT = `navigate_tree: invalid rewindHintReserveTokens in project config at ${PROJECT_PATH} — expected integer ${REWIND_HINT_MIN_RESERVE}-${REWIND_HINT_MAX_RESERVE} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;
const WARN_INVALID_START_GLOBAL = `navigate_tree: invalid anchorStartAfterTurns in global config at ${GLOBAL_PATH} — expected integer ${START_ANCHOR_MIN_TURNS}-${START_ANCHOR_MAX_TURNS}; falling back to the other layer's valid value, else the default ${START_ANCHOR_DEFAULT_TURNS}.`;
const WARN_INVALID_START_PROJECT = `navigate_tree: invalid anchorStartAfterTurns in project config at ${PROJECT_PATH} — expected integer ${START_ANCHOR_MIN_TURNS}-${START_ANCHOR_MAX_TURNS}; falling back to the other layer's valid value, else the default ${START_ANCHOR_DEFAULT_TURNS}.`;

/** In-memory `readFile` double: returns `files[path]` or throws ENOENT. */
function readerFor(files: Record<string, string>): {
  readFile: (path: string) => Promise<string>;
  paths: string[];
} {
  const paths: string[] = [];
  return {
    paths,
    async readFile(path: string) {
      paths.push(path);
      if (path in files) return files[path];
      const err = new Error(
        `ENOENT: no such file or directory, open '${path}'`,
      ) as NodeJS.ErrnoException;
      err.code = "ENOENT";
      throw err;
    },
  };
}

/** A `readFile` that throws a fixed errno for every path. */
function throwingReader(code: string): (path: string) => Promise<string> {
  return async (path: string) => {
    const err = new Error(
      `${code}: cannot open '${path}'`,
    ) as NodeJS.ErrnoException;
    err.code = code;
    throw err;
  };
}

function configJson(rewindHintReserveTokens: unknown): string {
  return JSON.stringify({ rewindHintReserveTokens });
}

function configJsonStart(anchorStartAfterTurns: unknown): string {
  return JSON.stringify({ anchorStartAfterTurns });
}

/** Build a `ParsedLayer` with the fields the test cares about. */
function layer(partial: {
  hint?: HintField;
  startAnchor?: StartAnchorField;
  modelHints?: Array<[string, HintField]>;
}): ParsedLayer {
  return {
    hint: partial.hint ?? { kind: "absent" },
    startAnchor: partial.startAnchor ?? { kind: "absent" },
    modelHints: new Map(partial.modelHints ?? []),
    modelOverridesProblems: {
      notObject: false,
      nonObjectEntries: [],
      paddedKeys: [],
      entriesWithoutReserve: [],
    },
  };
}

/** The loader's expected config object (start defaults unless overridden). */
function cfg(
  hint: number | null,
  start = START_ANCHOR_DEFAULT_TURNS,
  byModel: Array<[string, number | null]> = [],
) {
  return {
    rewindHintReserveTokens: hint,
    rewindHintModelReserveTokens: new Map(byModel),
    anchorStartAfterTurns: start,
  };
}

const DEEPSEEK = "opencode-go/deepseek-v4.1-flash";
const HAIKU = "claude-bridge/claude-haiku-5-5";

function warnInvalidModelHint(
  which: "global" | "project",
  path: string,
  key: string,
): string {
  return `navigate_tree: invalid modelOverrides[${JSON.stringify(key)}].rewindHintReserveTokens in ${which} config at ${path} — expected integer ${REWIND_HINT_MIN_RESERVE}-${REWIND_HINT_MAX_RESERVE} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for that model.`;
}

/** Parse a root expected to be an object and return its parsed layer. */
function parsedLayer(raw: unknown): ParsedLayer {
  const parsed = parseTreeNavigatorConfig(raw);
  assert.equal(parsed.kind, "fields");
  if (parsed.kind !== "fields") throw new Error("unreachable");
  return parsed.layer;
}

describe("parseTreeNavigatorConfig", () => {
  describe("rewindHintReserveTokens: accepts integer reserves (number and string forms)", () => {
    const accepted: Array<[string, unknown, number]> = [
      ["number", 50000, 50000],
      ["numeric string", "50000", 50000],
      ["trimmed numeric string", " 50000 ", 50000],
      ["min edge number", REWIND_HINT_MIN_RESERVE, REWIND_HINT_MIN_RESERVE],
      ["max edge number", REWIND_HINT_MAX_RESERVE, REWIND_HINT_MAX_RESERVE],
      [
        "min edge string",
        `${REWIND_HINT_MIN_RESERVE}`,
        REWIND_HINT_MIN_RESERVE,
      ],
      [
        "max edge string",
        `${REWIND_HINT_MAX_RESERVE}`,
        REWIND_HINT_MAX_RESERVE,
      ],
    ];
    for (const [label, raw, want] of accepted) {
      it(label, () => {
        assert.deepEqual(parsedLayer({ rewindHintReserveTokens: raw }).hint, {
          kind: "ok",
          value: want,
        });
      });
    }
  });

  describe("rewindHintReserveTokens: treats explicit-disable sentinels as disabled", () => {
    const disabling: unknown[] = [
      null,
      false,
      "off",
      "OFF",
      " Off ",
      "disabled",
      "DISABLED",
      "Disabled",
    ];
    for (const raw of disabling) {
      it(JSON.stringify(raw), () => {
        assert.deepEqual(parsedLayer({ rewindHintReserveTokens: raw }).hint, {
          kind: "disabled",
        });
      });
    }
  });

  describe("rewindHintReserveTokens: rejects unusable field values as invalid", () => {
    const invalid: Array<[string, unknown]> = [
      ["one below min", REWIND_HINT_MIN_RESERVE - 1],
      ["one above max", REWIND_HINT_MAX_RESERVE + 1],
      ["zero", 0],
      ["negative", -1],
      ["decimal", 50000.5],
      ["decimal string", "50000.5"],
      ["non-numeric string", "ninety"],
      ["empty string", ""],
      ["whitespace string", "   "],
      ["boolean true", true],
      ["object", {}],
      ["array", []],
      ["NaN number", Number.NaN],
      ["NaN string", "NaN"],
      ["negative string", "-1"],
      ["plus-signed string", "+50000"],
      ["decimal-zero string", "1000.0"],
      ["exponent string", "1e3"],
      ["plus-signed min string", "+1000"],
      ["string below min", "999"],
      ["string above max", "1000001"],
    ];
    for (const [label, raw] of invalid) {
      it(label, () => {
        assert.deepEqual(parsedLayer({ rewindHintReserveTokens: raw }).hint, {
          kind: "invalid",
        });
      });
    }
  });

  describe("anchorStartAfterTurns: accepts integer turn bounds (number and string forms)", () => {
    const accepted: Array<[string, unknown, number]> = [
      ["default", 2, 2],
      ["numeric string", "2", 2],
      ["trimmed numeric string", " 2 ", 2],
      ["min edge number", START_ANCHOR_MIN_TURNS, START_ANCHOR_MIN_TURNS],
      ["max edge number", START_ANCHOR_MAX_TURNS, START_ANCHOR_MAX_TURNS],
      ["min edge string", `${START_ANCHOR_MIN_TURNS}`, START_ANCHOR_MIN_TURNS],
      ["max edge string", `${START_ANCHOR_MAX_TURNS}`, START_ANCHOR_MAX_TURNS],
      ["leading zeroes", "02", 2],
    ];
    for (const [label, raw, want] of accepted) {
      it(label, () => {
        assert.deepEqual(
          parsedLayer({ anchorStartAfterTurns: raw }).startAnchor,
          { kind: "ok", value: want },
        );
      });
    }
  });

  describe("anchorStartAfterTurns: rejects unusable field values as invalid", () => {
    const invalid: Array<[string, unknown]> = [
      ["zero (no turn 0)", 0],
      ["one above max", START_ANCHOR_MAX_TURNS + 1],
      ["negative", -1],
      ["decimal", 2.5],
      ["decimal string", "2.5"],
      ["non-numeric string", "two"],
      ["empty string", ""],
      ["whitespace string", "   "],
      ["boolean true", true],
      ["boolean false (no disable sentinel)", false],
      ["null (no disable sentinel)", null],
      ["off string (no disable sentinel)", "off"],
      ["disabled string (no sentinel)", "disabled"],
      ["object", {}],
      ["array", []],
      ["NaN number", Number.NaN],
      ["NaN string", "NaN"],
      ["negative string", "-1"],
      ["plus-signed string", "+2"],
      ["string above max", `${START_ANCHOR_MAX_TURNS + 1}`],
    ];
    for (const [label, raw] of invalid) {
      it(label, () => {
        assert.deepEqual(
          parsedLayer({ anchorStartAfterTurns: raw }).startAnchor,
          { kind: "invalid" },
        );
      });
    }
  });

  it("parses each field independently (invalid hint leaves start valid)", () => {
    const parsed = parsedLayer({
      rewindHintReserveTokens: true,
      anchorStartAfterTurns: 5,
    });
    assert.deepEqual(parsed.hint, { kind: "invalid" });
    assert.deepEqual(parsed.startAnchor, { kind: "ok", value: 5 });
  });

  it("parses each field independently (invalid start leaves hint valid)", () => {
    const parsed = parsedLayer({
      rewindHintReserveTokens: 50000,
      anchorStartAfterTurns: "two",
    });
    assert.deepEqual(parsed.hint, { kind: "ok", value: 50000 });
    assert.deepEqual(parsed.startAnchor, { kind: "invalid" });
  });

  it("returns absent for each missing key", () => {
    assert.deepEqual(parsedLayer({}), layer({}));
    assert.deepEqual(parsedLayer({ other: 1 }), layer({}));
    // A present hint key leaves start absent (absent ≠ invalid).
    assert.deepEqual(
      parsedLayer({ rewindHintReserveTokens: 50000 }).startAnchor,
      {
        kind: "absent",
      },
    );
    assert.deepEqual(parsedLayer({ anchorStartAfterTurns: 4 }).hint, {
      kind: "absent",
    });
  });

  it("ignores the removed rewindHintAtPercent key (the loader warns)", () => {
    for (const raw of ["90", 90, null, "off", true]) {
      assert.deepEqual(parsedLayer({ rewindHintAtPercent: raw }), layer({}));
    }
  });

  it("returns unusable for a non-object root (file-level failure)", () => {
    for (const root of [null, [], 42, "x", true, undefined]) {
      assert.deepEqual(parseTreeNavigatorConfig(root), { kind: "unusable" });
    }
  });
});

describe("mergeTreeNavigatorConfig", () => {
  describe("hint field (issue #44 semantics preserved)", () => {
    const okGlobal: HintField = { kind: "ok", value: 50000 };
    const okProject: HintField = { kind: "ok", value: 30000 };
    const cases: Array<{
      name: string;
      global: HintField;
      project: HintField;
      want: HintField;
    }> = [
      {
        name: "absent + absent stays absent",
        global: { kind: "absent" },
        project: { kind: "absent" },
        want: { kind: "absent" },
      },
      {
        name: "project absent → global applies",
        global: okGlobal,
        project: { kind: "absent" },
        want: okGlobal,
      },
      {
        name: "global absent → project applies",
        global: { kind: "absent" },
        project: okProject,
        want: okProject,
      },
      {
        name: "project value wins over global value",
        global: okGlobal,
        project: okProject,
        want: okProject,
      },
      {
        name: "project disabled sentinel wins",
        global: okGlobal,
        project: { kind: "disabled" },
        want: { kind: "disabled" },
      },
      {
        name: "global disabled wins when project is absent",
        global: { kind: "disabled" },
        project: { kind: "absent" },
        want: { kind: "disabled" },
      },
      {
        name: "project invalid wins (no cross-layer fallback)",
        global: okGlobal,
        project: { kind: "invalid" },
        want: { kind: "invalid" },
      },
      {
        name: "global invalid is overridden by a valid project value",
        global: { kind: "invalid" },
        project: okProject,
        want: okProject,
      },
    ];
    for (const c of cases) {
      it(c.name, () => {
        assert.deepEqual(
          mergeTreeNavigatorConfig(
            layer({ hint: c.global }),
            layer({ hint: c.project }),
          ).hint,
          c.want,
        );
      });
    }
  });

  describe("start-anchor field (issue #55: invalid falls through)", () => {
    const okGlobal: StartAnchorField = { kind: "ok", value: 5 };
    const okProject: StartAnchorField = { kind: "ok", value: 3 };
    const cases: Array<{
      name: string;
      global: StartAnchorField;
      project: StartAnchorField;
      want: StartAnchorField;
    }> = [
      {
        name: "absent + absent stays absent (resolves to default)",
        global: { kind: "absent" },
        project: { kind: "absent" },
        want: { kind: "absent" },
      },
      {
        name: "project absent → global applies",
        global: okGlobal,
        project: { kind: "absent" },
        want: okGlobal,
      },
      {
        name: "global absent → project applies",
        global: { kind: "absent" },
        project: okProject,
        want: okProject,
      },
      {
        name: "project value wins over global value",
        global: okGlobal,
        project: okProject,
        want: okProject,
      },
      {
        name: "project invalid + valid global → global applies",
        global: okGlobal,
        project: { kind: "invalid" },
        want: okGlobal,
      },
      {
        name: "project invalid + absent global → invalid (resolves to default)",
        global: { kind: "absent" },
        project: { kind: "invalid" },
        want: { kind: "invalid" },
      },
      {
        name: "project invalid + invalid global → invalid (resolves to default)",
        global: { kind: "invalid" },
        project: { kind: "invalid" },
        want: { kind: "invalid" },
      },
      {
        name: "project ok overrides an invalid global",
        global: { kind: "invalid" },
        project: okProject,
        want: okProject,
      },
      {
        name: "project absent + invalid global → invalid (resolves to default)",
        global: { kind: "invalid" },
        project: { kind: "absent" },
        want: { kind: "invalid" },
      },
    ];
    for (const c of cases) {
      it(c.name, () => {
        assert.deepEqual(
          mergeTreeNavigatorConfig(
            layer({ startAnchor: c.global }),
            layer({ startAnchor: c.project }),
          ).startAnchor,
          c.want,
        );
      });
    }
  });

  it("merges the two fields independently", () => {
    const merged = mergeTreeNavigatorConfig(
      layer({
        hint: { kind: "ok", value: 50000 },
        startAnchor: { kind: "ok", value: 5 },
      }),
      layer({ hint: { kind: "invalid" } }),
    );
    assert.deepEqual(merged.hint, { kind: "invalid" });
    assert.deepEqual(merged.startAnchor, { kind: "ok", value: 5 });
  });
});

describe("loadTreeNavigatorConfig", () => {
  it("reads both layers and lets a trusted project override global per field", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintReserveTokens: "50000",
        anchorStartAfterTurns: 5,
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintReserveTokens: 30000,
        anchorStartAfterTurns: 3,
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(30000, 3), warnings: [] });
    assert.deepEqual(reader.paths, [GLOBAL_PATH, PROJECT_PATH]);
  });

  it("never reads the project layer when the project is untrusted", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintReserveTokens: "50000",
        anchorStartAfterTurns: 5,
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintReserveTokens: 30000,
        anchorStartAfterTurns: 3,
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(50000, 5), warnings: [] });
    assert.deepEqual(reader.paths, [GLOBAL_PATH]);
  });

  it("is silent when both files are missing (ENOENT is the normal state)", async () => {
    const reader = readerFor({});
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    // Start anchor is non-optional: no file still resolves to the default 2
    // with no warning; the hint stays off.
    assert.deepEqual(out, {
      config: cfg(null, START_ANCHOR_DEFAULT_TURNS),
      warnings: [],
    });
  });

  it("is silent when the file exists but neither field is present", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: JSON.stringify({ other: 1 }) });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null, START_ANCHOR_DEFAULT_TURNS),
      warnings: [],
    });
  });

  it("warns on the removed rewindHintAtPercent key and leaves the hint off", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({ rewindHintAtPercent: "90" }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [warnLegacyPercent("global", GLOBAL_PATH)],
    });
  });

  it("warns on a legacy project key while the global reserve still applies", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({ rewindHintReserveTokens: 50000 }),
      [PROJECT_PATH]: JSON.stringify({ rewindHintAtPercent: 80 }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(50000),
      warnings: [warnLegacyPercent("project", PROJECT_PATH)],
    });
  });

  it("trims and accepts numeric-string values through the loader", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintReserveTokens: " 50000 ",
        anchorStartAfterTurns: " 4 ",
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(50000, 4), warnings: [] });
  });

  it("warns on a non-ENOENT read error and ignores that layer (EACCES)", async () => {
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: throwingReader("EACCES"),
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [WARN_COULD_NOT_READ_GLOBAL, WARN_COULD_NOT_READ_PROJECT],
    });
  });

  it("warns on EISDIR and still applies the other layer", async () => {
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: readerFor({ [PROJECT_PATH]: configJson("63000") }).readFile,
    });
    // Global read hits the injected ENOENT (silent), project applies.
    assert.deepEqual(out, { config: cfg(63000), warnings: [] });

    const dirReader = throwingReader("EISDIR");
    const dirOut = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: dirReader,
    });
    assert.deepEqual(dirOut, {
      config: cfg(null),
      warnings: [WARN_COULD_NOT_READ_GLOBAL],
    });
  });

  it("warns on malformed, empty, and whitespace-only JSON", async () => {
    for (const raw of ["{ not json", "", "   \n  "]) {
      const reader = readerFor({ [GLOBAL_PATH]: raw });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: false,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(null),
        warnings: [WARN_INVALID_JSON_GLOBAL],
      });
    }
  });

  it("warns when the root is not a JSON object", async () => {
    for (const raw of ["[]", "42", "null", '"x"']) {
      const reader = readerFor({ [GLOBAL_PATH]: raw });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: false,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(null),
        warnings: [WARN_NOT_OBJECT_GLOBAL],
      });
    }
  });

  it("warns once on an invalid hint value and disables the hint (no fallback)", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: configJson(true) });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [WARN_INVALID_HINT_GLOBAL],
    });
  });

  it("an invalid project hint disables despite a valid global (no fallback)", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson(50000),
      [PROJECT_PATH]: configJson({ nope: true }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [WARN_INVALID_HINT_PROJECT],
    });
  });

  it("a project hint disable sentinel overrides a valid global", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson(50000),
      [PROJECT_PATH]: configJson(null),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(null), warnings: [] });
  });

  it("an invalid project start value falls through to a valid global (issue #55)", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJsonStart(5),
      [PROJECT_PATH]: configJsonStart("abc"),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null, 5),
      warnings: [WARN_INVALID_START_PROJECT],
    });
  });

  it("an invalid global start value falls through to a valid project (issue #55)", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJsonStart(51),
      [PROJECT_PATH]: configJsonStart("3"),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null, 3),
      warnings: [WARN_INVALID_START_GLOBAL],
    });
  });

  it("an invalid start value with no valid layer warns and uses the default", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: configJsonStart(true) });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null, START_ANCHOR_DEFAULT_TURNS),
      warnings: [WARN_INVALID_START_GLOBAL],
    });
  });

  it("both fields invalid in one layer emit one scoped warning each", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintReserveTokens: true,
        anchorStartAfterTurns: "lots",
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [WARN_INVALID_HINT_GLOBAL, WARN_INVALID_START_GLOBAL],
    });
  });

  it("a file-level failure warns once and suppresses per-field warnings", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: "{ not json" });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [WARN_INVALID_JSON_GLOBAL],
    });
  });

  it("orders warnings global-then-project across fields", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintReserveTokens: "nope",
        anchorStartAfterTurns: "nope",
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintReserveTokens: { bad: true },
        anchorStartAfterTurns: [],
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out.warnings, [
      WARN_INVALID_HINT_GLOBAL,
      WARN_INVALID_START_GLOBAL,
      WARN_INVALID_HINT_PROJECT,
      WARN_INVALID_START_PROJECT,
    ]);
    assert.equal(out.config.rewindHintReserveTokens, null);
    assert.equal(out.config.anchorStartAfterTurns, START_ANCHOR_DEFAULT_TURNS);
  });

  it("a malformed project file warns but a valid global still applies", async () => {
    for (const raw of ["{ not json", "[]"]) {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          rewindHintReserveTokens: 50000,
          anchorStartAfterTurns: 5,
        }),
        [PROJECT_PATH]: raw,
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: true,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(50000, 5),
        warnings: [
          raw === "[]" ? WARN_NOT_OBJECT_PROJECT : WARN_INVALID_JSON_PROJECT,
        ],
      });
    }
  });

  it("a malformed project file is silent when the project is untrusted", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson(50000),
      [PROJECT_PATH]: "{ not json",
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(50000), warnings: [] });
    assert.deepEqual(reader.paths, [GLOBAL_PATH]);
  });

  it("emits the file-level warning then the surviving field warning (per layer)", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: "{ not json",
      [PROJECT_PATH]: configJson("nope"),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out.warnings, [
      WARN_INVALID_JSON_GLOBAL,
      WARN_INVALID_HINT_PROJECT,
    ]);
    assert.equal(out.config.rewindHintReserveTokens, null);
    assert.equal(out.config.anchorStartAfterTurns, START_ANCHOR_DEFAULT_TURNS);
  });

  it("uses the real default readFile when none is injected (ENOENT on a fresh temp dir)", async () => {
    // Hermetic-by-absence: point agentDir at a path that cannot exist and
    // assert the default fs reader surfaces ENOENT as a silent absent layer.
    const out = await loadTreeNavigatorConfig({
      agentDir: "/nonexistent-navigate-tree-test-agent-dir",
      projectPath:
        "/nonexistent-navigate-tree-test-project/.pi/tree-navigator.json",
      projectTrusted: false,
    });
    assert.deepEqual(out, {
      config: cfg(null),
      warnings: [],
    });
  });
});

describe("modelOverrides (#86)", () => {
  describe("parse", () => {
    it("parses per-model reserves with the base field's rules", () => {
      const parsed = parsedLayer({
        modelOverrides: {
          [DEEPSEEK]: { rewindHintReserveTokens: 50000 },
          [HAIKU]: { rewindHintReserveTokens: "off" },
          "x/bad": { rewindHintReserveTokens: 5 },
          "x/str": { rewindHintReserveTokens: " 30000 " },
          // No reserve key (e.g. pi's `reserveTokens`): a problem, not an
          // override.
          "x/none": { reserveTokens: 1 },
          " x/padded ": { rewindHintReserveTokens: 30000 },
        },
      });
      assert.deepEqual(
        parsed.modelHints,
        new Map<string, HintField>([
          [DEEPSEEK, { kind: "ok", value: 50000 }],
          [HAIKU, { kind: "disabled" }],
          ["x/bad", { kind: "invalid" }],
          ["x/str", { kind: "ok", value: 30000 }],
        ]),
      );
      assert.deepEqual(parsed.modelOverridesProblems, {
        notObject: false,
        nonObjectEntries: [],
        paddedKeys: [" x/padded "],
        entriesWithoutReserve: ["x/none"],
      });
    });

    it("records a non-object modelOverrides and non-object entries", () => {
      for (const raw of [null, 5, "x", [], true]) {
        const parsed = parsedLayer({ modelOverrides: raw });
        assert.equal(parsed.modelHints.size, 0);
        assert.equal(parsed.modelOverridesProblems.notObject, true);
      }
      const parsed = parsedLayer({
        modelOverrides: { a: 1, b: null, c: [], [DEEPSEEK]: {} },
      });
      assert.equal(parsed.modelHints.size, 0);
      assert.deepEqual(parsed.modelOverridesProblems, {
        notObject: false,
        nonObjectEntries: ["a", "b", "c"],
        paddedKeys: [],
        entriesWithoutReserve: [DEEPSEEK],
      });
    });

    it("leaves an absent modelOverrides problem-free", () => {
      const parsed = parsedLayer({ rewindHintReserveTokens: 40000 });
      assert.equal(parsed.modelHints.size, 0);
      assert.deepEqual(parsed.modelOverridesProblems, {
        notObject: false,
        nonObjectEntries: [],
        paddedKeys: [],
        entriesWithoutReserve: [],
      });
    });
  });

  describe("merge", () => {
    it("merges per key: a project entry wins, others pass through", () => {
      const merged = mergeTreeNavigatorConfig(
        layer({
          modelHints: [
            [DEEPSEEK, { kind: "ok", value: 50000 }],
            [HAIKU, { kind: "ok", value: 20000 }],
          ],
        }),
        layer({ modelHints: [[HAIKU, { kind: "disabled" }]] }),
      );
      assert.deepEqual(
        merged.modelHints,
        new Map<string, HintField>([
          [DEEPSEEK, { kind: "ok", value: 50000 }],
          [HAIKU, { kind: "disabled" }],
        ]),
      );
    });
  });

  describe("resolveRewindHintReserve", () => {
    const config = {
      rewindHintReserveTokens: 40000,
      rewindHintModelReserveTokens: new Map<string, number | null>([
        [DEEPSEEK, 50000],
        [HAIKU, null],
      ]),
    };

    it("prefers a matching entry, including an off entry", () => {
      assert.equal(resolveRewindHintReserve(config, DEEPSEEK), 50000);
      assert.equal(resolveRewindHintReserve(config, HAIKU), null);
    });

    it("falls back to the base for an unmatched or unknown model", () => {
      assert.equal(resolveRewindHintReserve(config, "x/other"), 40000);
      assert.equal(resolveRewindHintReserve(config, undefined), 40000);
    });

    it("an entry enables a model while the base is off", () => {
      assert.equal(
        resolveRewindHintReserve(
          { ...config, rewindHintReserveTokens: null },
          DEEPSEEK,
        ),
        50000,
      );
    });

    it("matches keys exactly", () => {
      assert.equal(
        resolveRewindHintReserve(config, DEEPSEEK.toUpperCase()),
        40000,
      );
    });
  });

  describe("loader", () => {
    it("loads the approved snippet", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          rewindHintReserveTokens: 40000,
          modelOverrides: {
            [DEEPSEEK]: { rewindHintReserveTokens: 50000 },
          },
        }),
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: false,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(40000, START_ANCHOR_DEFAULT_TURNS, [[DEEPSEEK, 50000]]),
        warnings: [],
      });
    });

    it("a global model entry beats a project base (pi precedent)", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          modelOverrides: { [DEEPSEEK]: { rewindHintReserveTokens: 50000 } },
        }),
        [PROJECT_PATH]: JSON.stringify({
          rewindHintReserveTokens: 30000,
          modelOverrides: { [HAIKU]: { rewindHintReserveTokens: false } },
        }),
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: true,
        readFile: reader.readFile,
      });
      assert.deepEqual(out.warnings, []);
      assert.equal(resolveRewindHintReserve(out.config, DEEPSEEK), 50000);
      assert.equal(resolveRewindHintReserve(out.config, HAIKU), null);
      assert.equal(resolveRewindHintReserve(out.config, "x/y"), 30000);
    });

    it("ignores the untrusted project's entries", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({ rewindHintReserveTokens: 40000 }),
        [PROJECT_PATH]: JSON.stringify({
          modelOverrides: { [DEEPSEEK]: { rewindHintReserveTokens: 50000 } },
        }),
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: false,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, { config: cfg(40000), warnings: [] });
    });

    it("warns on shape problems and invalid entries; an invalid entry turns that model off", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          rewindHintReserveTokens: 40000,
          modelOverrides: {
            [DEEPSEEK]: { rewindHintReserveTokens: 999 },
            "x/num": 5,
          },
        }),
        [PROJECT_PATH]: JSON.stringify({ modelOverrides: [] }),
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: true,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(40000, START_ANCHOR_DEFAULT_TURNS, [[DEEPSEEK, null]]),
        warnings: [
          `navigate_tree: modelOverrides["x/num"] in global config at ${GLOBAL_PATH} must be an object — ignored.`,
          warnInvalidModelHint("global", GLOBAL_PATH, DEEPSEEK),
          `navigate_tree: modelOverrides in project config at ${PROJECT_PATH} must be an object keyed by "provider/modelId" — ignored.`,
        ],
      });
    });

    it("warns on a padded key and on an entry without rewindHintReserveTokens", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          rewindHintReserveTokens: 40000,
          modelOverrides: {
            [` ${DEEPSEEK}`]: { rewindHintReserveTokens: 50000 },
            [HAIKU]: { reserveTokens: 50000 },
          },
        }),
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: false,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(40000),
        warnings: [
          `navigate_tree: modelOverrides key " ${DEEPSEEK}" in global config at ${GLOBAL_PATH} has surrounding whitespace and can never match a model — ignored.`,
          `navigate_tree: modelOverrides["${HAIKU}"] in global config at ${GLOBAL_PATH} sets no rewindHintReserveTokens — ignored.`,
        ],
      });
    });

    it("a project file-level problem keeps the global entries", async () => {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          modelOverrides: { [DEEPSEEK]: { rewindHintReserveTokens: 50000 } },
        }),
        [PROJECT_PATH]: "{not json",
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: true,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: cfg(null, START_ANCHOR_DEFAULT_TURNS, [[DEEPSEEK, 50000]]),
        warnings: [WARN_INVALID_JSON_PROJECT],
      });
    });
  });
});
