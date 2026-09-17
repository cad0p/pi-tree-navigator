/**
 * Tests for the two-layer tree-navigator.json config loader.
 *
 * Node test runner. Run with: pnpm test
 *
 * The pi extension loader treats `./index.ts` as the entry point and ignores
 * sibling files — so this test file is not loaded as a separate extension.
 * The loader's `readFile` is injected, so these tests touch no real fs and
 * pin the layer/merge/warning matrix without a session.
 */

import * as assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  loadTreeNavigatorConfig,
  mergeTreeNavigatorConfig,
  type ParsedLayer,
  parseTreeNavigatorConfig,
  REWIND_HINT_MAX_PERCENT,
  REWIND_HINT_MIN_PERCENT,
  TREE_NAVIGATOR_CONFIG_FILENAME,
} from "./config.ts";

const AGENT_DIR = "/agent";
const GLOBAL_PATH = `/agent/${TREE_NAVIGATOR_CONFIG_FILENAME}`;
const PROJECT_PATH = `/project/.pi/${TREE_NAVIGATOR_CONFIG_FILENAME}`;

// Byte-exact warning copy (frozen by issue #44). Interpolating the same
// constants the source interpolates keeps the pins honest if the range is
// ever re-tuned.
const WARN_COULD_NOT_READ_GLOBAL = `navigate_tree: could not read global config at ${GLOBAL_PATH} — that layer was ignored.`;
const WARN_COULD_NOT_READ_PROJECT = `navigate_tree: could not read project config at ${PROJECT_PATH} — that layer was ignored.`;
const WARN_INVALID_JSON_GLOBAL = `navigate_tree: invalid JSON in global config at ${GLOBAL_PATH} — that layer was ignored.`;
const WARN_INVALID_JSON_PROJECT = `navigate_tree: invalid JSON in project config at ${PROJECT_PATH} — that layer was ignored.`;
const WARN_NOT_OBJECT_GLOBAL = `navigate_tree: global config at ${GLOBAL_PATH} must be a JSON object — that layer was ignored.`;
const WARN_NOT_OBJECT_PROJECT = `navigate_tree: project config at ${PROJECT_PATH} must be a JSON object — that layer was ignored.`;
const WARN_INVALID_FIELD_GLOBAL = `navigate_tree: invalid rewindHintAtPercent in global config at ${GLOBAL_PATH} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;
const WARN_INVALID_FIELD_PROJECT = `navigate_tree: invalid rewindHintAtPercent in project config at ${PROJECT_PATH} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`;

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

describe("parseTreeNavigatorConfig", () => {
  describe("accepts integer thresholds (number and string forms)", () => {
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
        assert.deepEqual(
          parseTreeNavigatorConfig({ rewindHintAtPercent: raw }),
          {
            kind: "ok",
            value: want,
          },
        );
      });
    }
  });

  describe("treats explicit-disable sentinels as disabled", () => {
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
        assert.deepEqual(
          parseTreeNavigatorConfig({ rewindHintAtPercent: raw }),
          { kind: "disabled" },
        );
      });
    }
  });

  describe("rejects unusable field values as invalid", () => {
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
        assert.deepEqual(
          parseTreeNavigatorConfig({ rewindHintAtPercent: raw }),
          { kind: "invalid" },
        );
      });
    }
  });

  it("returns absent when the key is missing", () => {
    assert.deepEqual(parseTreeNavigatorConfig({}), { kind: "absent" });
    assert.deepEqual(parseTreeNavigatorConfig({ other: 1 }), {
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
  const okGlobal: ParsedLayer = { kind: "ok", value: 90 };
  const okProject: ParsedLayer = { kind: "ok", value: 50 };
  const cases: Array<{
    name: string;
    global: ParsedLayer;
    project: ParsedLayer;
    want: ParsedLayer;
  }> = [
    {
      name: "absent + absent stays absent",
      global: { kind: "absent" },
      project: { kind: "absent" },
      want: { kind: "absent" },
    },
    {
      name: "unusable global falls through to an absent project",
      global: { kind: "unusable" },
      project: { kind: "absent" },
      want: { kind: "unusable" },
    },
    {
      name: "project absent → global applies",
      global: okGlobal,
      project: { kind: "absent" },
      want: okGlobal,
    },
    {
      name: "project unusable → global applies (file-level fallthrough)",
      global: okGlobal,
      project: { kind: "unusable" },
      want: okGlobal,
    },
    {
      name: "global absent → project applies",
      global: { kind: "absent" },
      project: okProject,
      want: okProject,
    },
    {
      name: "global unusable → project applies",
      global: { kind: "unusable" },
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
      assert.deepEqual(mergeTreeNavigatorConfig(c.global, c.project), c.want);
    });
  }
});

describe("loadTreeNavigatorConfig", () => {
  it("reads both layers and lets a trusted project override global", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson("90"),
      [PROJECT_PATH]: configJson(50),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: 50 },
      warnings: [],
    });
    assert.deepEqual(reader.paths, [GLOBAL_PATH, PROJECT_PATH]);
  });

  it("never reads the project layer when the project is untrusted", async () => {
    const reader = readerFor({
      [GLOBAL_PATH]: configJson("90"),
      [PROJECT_PATH]: configJson(50),
    });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: 90 },
      warnings: [],
    });
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
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: null },
      warnings: [],
    });
  });

  it("trims and accepts a numeric-string threshold through the loader", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: configJson(" 90 ") });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: 90 },
      warnings: [],
    });
  });

  it("warns on a non-ENOENT read error and ignores that layer (EACCES)", async () => {
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: true,
      readFile: throwingReader("EACCES"),
    });
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: null },
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
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: 63 },
      warnings: [],
    });

    const dirReader = throwingReader("EISDIR");
    const dirOut = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: dirReader,
    });
    assert.deepEqual(dirOut, {
      config: { rewindHintAtPercent: null },
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
        config: { rewindHintAtPercent: null },
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
        config: { rewindHintAtPercent: null },
        warnings: [WARN_NOT_OBJECT_GLOBAL],
      });
    }
  });

  it("warns once on an invalid field value and disables the session", async () => {
    const reader = readerFor({ [GLOBAL_PATH]: configJson(true) });
    const out = await loadTreeNavigatorConfig({
      agentDir: AGENT_DIR,
      projectPath: PROJECT_PATH,
      projectTrusted: false,
      readFile: reader.readFile,
    });
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: null },
      warnings: [WARN_INVALID_FIELD_GLOBAL],
    });
  });

  it("an invalid project value disables despite a valid global (no fallback)", async () => {
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
      config: { rewindHintAtPercent: null },
      warnings: [WARN_INVALID_FIELD_PROJECT],
    });
  });

  it("a project disable sentinel overrides a valid global", async () => {
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
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: null },
      warnings: [],
    });
  });

  it("a malformed project file warns but a valid global still applies", async () => {
    for (const raw of ["{ not json", "[]"]) {
      const reader = readerFor({
        [GLOBAL_PATH]: configJson(90),
        [PROJECT_PATH]: raw,
      });
      const out = await loadTreeNavigatorConfig({
        agentDir: AGENT_DIR,
        projectPath: PROJECT_PATH,
        projectTrusted: true,
        readFile: reader.readFile,
      });
      assert.deepEqual(out, {
        config: { rewindHintAtPercent: 90 },
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
    assert.deepEqual(out, {
      config: { rewindHintAtPercent: 90 },
      warnings: [],
    });
    assert.deepEqual(reader.paths, [GLOBAL_PATH]);
  });

  it("emits at most one warning per layer (global file-level + project field-level)", async () => {
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
      WARN_INVALID_FIELD_PROJECT,
    ]);
    assert.equal(out.config.rewindHintAtPercent, null);
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
      config: { rewindHintAtPercent: null },
      warnings: [],
    });
  });
});
