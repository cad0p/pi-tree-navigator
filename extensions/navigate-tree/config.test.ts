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
 *   - `rewindHintAtPercent`: field-level invalid disables the hint for the
 *     session with NO cross-layer fallback.
 *   - `anchorStartAfterTurns`: field-level invalid warns and falls through
 *     to the other layer's valid value, else the default 2.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  type HintField,
  loadTreeNavigatorConfig,
  mergeTreeNavigatorConfig,
  type ParsedLayer,
  parseTreeNavigatorConfig,
  REWIND_HINT_MAX_PERCENT,
  REWIND_HINT_MIN_PERCENT,
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
const WARN_INVALID_HINT_GLOBAL = `navigate_tree: invalid rewindHintAtPercent in global config at ${GLOBAL_PATH} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;
const WARN_INVALID_HINT_PROJECT = `navigate_tree: invalid rewindHintAtPercent in project config at ${PROJECT_PATH} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;
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

function configJson(rewindHintAtPercent: unknown): string {
  return JSON.stringify({ rewindHintAtPercent });
}

function configJsonStart(anchorStartAfterTurns: unknown): string {
  return JSON.stringify({ anchorStartAfterTurns });
}

/** Build a `ParsedLayer` with the fields the test cares about. */
function layer(partial: {
  hint?: HintField;
  startAnchor?: StartAnchorField;
}): ParsedLayer {
  return {
    hint: partial.hint ?? { kind: "absent" },
    startAnchor: partial.startAnchor ?? { kind: "absent" },
  };
}

/** The loader's expected config object (start defaults unless overridden). */
function cfg(hint: number | null, start = START_ANCHOR_DEFAULT_TURNS) {
  return { rewindHintAtPercent: hint, anchorStartAfterTurns: start };
}

/** Parse a root expected to be an object and return its parsed layer. */
function parsedLayer(raw: unknown): ParsedLayer {
  const parsed = parseTreeNavigatorConfig(raw);
  assert.equal(parsed.kind, "fields");
  if (parsed.kind !== "fields") throw new Error("unreachable");
  return parsed.layer;
}

describe("parseTreeNavigatorConfig", () => {
  describe("rewindHintAtPercent: accepts integer thresholds (number and string forms)", () => {
    const accepted: Array<[string, unknown, number]> = [
      ["number", 90, 90],
      ["numeric string", "90", 90],
      ["trimmed numeric string", " 90 ", 90],
      ["min edge number", REWIND_HINT_MIN_PERCENT, REWIND_HINT_MIN_PERCENT],
      ["max edge number", REWIND_HINT_MAX_PERCENT, REWIND_HINT_MAX_PERCENT],
      [
        "min edge string",
        `${REWIND_HINT_MIN_PERCENT}`,
        REWIND_HINT_MIN_PERCENT,
      ],
      [
        "max edge string",
        `${REWIND_HINT_MAX_PERCENT}`,
        REWIND_HINT_MAX_PERCENT,
      ],
    ];
    for (const [label, raw, want] of accepted) {
      it(label, () => {
        assert.deepEqual(parsedLayer({ rewindHintAtPercent: raw }).hint, {
          kind: "ok",
          value: want,
        });
      });
    }
  });

  describe("rewindHintAtPercent: treats explicit-disable sentinels as disabled", () => {
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
        assert.deepEqual(parsedLayer({ rewindHintAtPercent: raw }).hint, {
          kind: "disabled",
        });
      });
    }
  });

  describe("rewindHintAtPercent: rejects unusable field values as invalid", () => {
    const invalid: Array<[string, unknown]> = [
      ["one below min", REWIND_HINT_MIN_PERCENT - 1],
      ["one above max", REWIND_HINT_MAX_PERCENT + 1],
      ["zero", 0],
      ["negative", -1],
      ["decimal", 90.5],
      ["decimal string", "90.5"],
      ["non-numeric string", "ninety"],
      ["empty string", ""],
      ["whitespace string", "   "],
      ["boolean true", true],
      ["object", {}],
      ["array", []],
      ["NaN number", Number.NaN],
      ["NaN string", "NaN"],
      ["negative string", "-1"],
      ["plus-signed string", "+90"],
      ["string below min", "19"],
      ["string above max", "96"],
    ];
    for (const [label, raw] of invalid) {
      it(label, () => {
        assert.deepEqual(parsedLayer({ rewindHintAtPercent: raw }).hint, {
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
      rewindHintAtPercent: true,
      anchorStartAfterTurns: 5,
    });
    assert.deepEqual(parsed.hint, { kind: "invalid" });
    assert.deepEqual(parsed.startAnchor, { kind: "ok", value: 5 });
  });

  it("parses each field independently (invalid start leaves hint valid)", () => {
    const parsed = parsedLayer({
      rewindHintAtPercent: 90,
      anchorStartAfterTurns: "two",
    });
    assert.deepEqual(parsed.hint, { kind: "ok", value: 90 });
    assert.deepEqual(parsed.startAnchor, { kind: "invalid" });
  });

  it("returns absent for each missing key", () => {
    assert.deepEqual(parsedLayer({}), layer({}));
    assert.deepEqual(parsedLayer({ other: 1 }), layer({}));
    // A present hint key leaves start absent (absent ≠ invalid).
    assert.deepEqual(parsedLayer({ rewindHintAtPercent: 90 }).startAnchor, {
      kind: "absent",
    });
    assert.deepEqual(parsedLayer({ anchorStartAfterTurns: 4 }).hint, {
      kind: "absent",
    });
  });

  it("returns unusable for a non-object root (file-level failure)", () => {
    for (const root of [null, [], 42, "x", true, undefined]) {
      assert.deepEqual(parseTreeNavigatorConfig(root), { kind: "unusable" });
    }
  });
});

describe("mergeTreeNavigatorConfig", () => {
  describe("hint field (issue #44 semantics preserved)", () => {
    const okGlobal: HintField = { kind: "ok", value: 90 };
    const okProject: HintField = { kind: "ok", value: 50 };
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
        hint: { kind: "ok", value: 90 },
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
        rewindHintAtPercent: "90",
        anchorStartAfterTurns: 5,
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintAtPercent: 50,
        anchorStartAfterTurns: 3,
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(50, 3), warnings: [] });
    assert.deepEqual(reader.paths, [GLOBAL_PATH, PROJECT_PATH]);
  });

  it("never reads the project layer when the project is untrusted", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintAtPercent: "90",
        anchorStartAfterTurns: 5,
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintAtPercent: 50,
        anchorStartAfterTurns: 3,
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(90, 5), warnings: [] });
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

  it("trims and accepts numeric-string values through the loader", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: JSON.stringify({
        rewindHintAtPercent: " 90 ",
        anchorStartAfterTurns: " 4 ",
      }),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(90, 4), warnings: [] });
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
      readFile: readerFor({ [PROJECT_PATH]: configJson("63") }).readFile,
    });
    // Global read hits the injected ENOENT (silent), project applies.
    assert.deepEqual(out, { config: cfg(63), warnings: [] });

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
      [GLOBAL_PATH]: configJson(90),
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
      [GLOBAL_PATH]: configJson(90),
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
        rewindHintAtPercent: true,
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
        rewindHintAtPercent: "nope",
        anchorStartAfterTurns: "nope",
      }),
      [PROJECT_PATH]: JSON.stringify({
        rewindHintAtPercent: { bad: true },
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
    assert.equal(out.config.rewindHintAtPercent, null);
    assert.equal(out.config.anchorStartAfterTurns, START_ANCHOR_DEFAULT_TURNS);
  });

  it("a malformed project file warns but a valid global still applies", async () => {
    for (const raw of ["{ not json", "[]"]) {
      const reader = readerFor({
        [GLOBAL_PATH]: JSON.stringify({
          rewindHintAtPercent: 90,
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
        config: cfg(90, 5),
        warnings: [
          raw === "[]" ? WARN_NOT_OBJECT_PROJECT : WARN_INVALID_JSON_PROJECT,
        ],
      });
    }
  });

  it("a malformed project file is silent when the project is untrusted", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson(90),
      [PROJECT_PATH]: "{ not json",
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, { config: cfg(90), warnings: [] });
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
    assert.equal(out.config.rewindHintAtPercent, null);
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
