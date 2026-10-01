/**
 * tree-navigator.json — two-layer config loader for the navigate-tree
 * extension.
 *
 * Pure parse/merge functions plus a thin loader with an injectable
 * `readFile`. No pi runtime imports: the caller computes `projectPath` from
 * `CONFIG_DIR_NAME` (so `.pi` stays rebrandable and `PI_CODING_AGENT_DIR`
 * stays honored), and the loader stays unit-testable without a session.
 *
 * Layers, in application order:
 *   - global:  join(agentDir, "tree-navigator.json")
 *   - project: join(cwd, CONFIG_DIR_NAME, "tree-navigator.json"), read only
 *     when the project is trusted (untrusted → ignored silently).
 *
 * Fields (each parsed and merged independently):
 *   - `rewindHintAtPercent` (issue #44): opt-in context-pressure hint, or a
 *     disable sentinel.
 *   - `anchorStartAfterTurns` (issue #55): turn bound for the automatic
 *     `anchor:start` write. Non-optional — an absent key still fires.
 *
 * Failure semantics (issues #44/#55; authoritative):
 *   - File-level problems (non-ENOENT read failure, invalid JSON, non-object
 *     root) make that layer contribute nothing — every field falls through
 *     to the OTHER layer, which still applies.
 *   - HINT field-level problems (`rewindHintAtPercent` present but unusable)
 *     disable the hint for the session with NO cross-layer fallback: the
 *     user touched the field, so firing at the other layer's threshold would
 *     be an unintended rewind.
 *   - START-ANCHOR field-level problems (`anchorStartAfterTurns` present but
 *     unusable) warn and fall through to the other layer's valid value, else
 *     the default: a touched-but-invalid value must not disable a
 *     non-optional feature, and any unintended turn index only places a
 *     benign label.
 *   - A missing file is the normal "not configured" state → silent. Warning
 *     on it would toast every session of every user without a config.
 *
 * Each invalid field emits its own warning, scoped to that field. These
 * helpers do not import pi or log: warnings are returned as strings so
 * `index.ts` can route them through `ctx.ui.notify` (gated on `hasUI`).
 */

import { readFile as fsReadFile } from "node:fs/promises";
import { join } from "node:path";

export const TREE_NAVIGATOR_CONFIG_FILENAME = "tree-navigator.json";
export const REWIND_HINT_MIN_PERCENT = 20;
export const REWIND_HINT_MAX_PERCENT = 95;
export const START_ANCHOR_MIN_TURNS = 1;
export const START_ANCHOR_MAX_TURNS = 50;
/**
 * Effective turn bound when `anchorStartAfterTurns` is absent everywhere
 * (or invalid in every layer that touched it). Unlike the hint, the start
 * anchor is non-optional: a missing config file still fires at this default.
 */
export const START_ANCHOR_DEFAULT_TURNS = 2;

/**
 * Per-field parse outcome for `rewindHintAtPercent`.
 *
 *   - `absent`:   no key — contributes nothing (falls through).
 *   - `disabled`: explicit sentinel — wins the merge.
 *   - `ok`:       valid threshold — wins the merge.
 *   - `invalid`:  key present, value unusable — wins the merge (disables
 *                 the session; no cross-layer fallback).
 */
export type HintField =
  | { kind: "absent" }
  | { kind: "disabled" }
  | { kind: "ok"; value: number }
  | { kind: "invalid" };

/**
 * Per-field parse outcome for `anchorStartAfterTurns`.
 *
 *   - `absent`:   no key — contributes nothing (falls through to the other
 *                 layer, then the default).
 *   - `ok`:       valid turn bound — wins the merge.
 *   - `invalid`:  key present, value unusable — falls through to the other
 *                 layer's valid value, else the default (see merge/resolve).
 */
export type StartAnchorField =
  | { kind: "absent" }
  | { kind: "ok"; value: number }
  | { kind: "invalid" };

/** One config layer's parsed fields (file-level failures are all-absent). */
export interface ParsedLayer {
  hint: HintField;
  startAnchor: StartAnchorField;
}

/**
 * Root parse result. A non-object root (null, array, primitive) is a
 * file-level failure so the other layer can still apply every field.
 */
export type ParsedRoot =
  | { kind: "unusable" }
  | { kind: "fields"; layer: ParsedLayer };

/** Case-insensitive string sentinels that explicitly disable the hint. */
const DISABLE_SENTINELS = new Set(["off", "disabled"]);

function parseHintPercent(n: number): HintField {
  return Number.isInteger(n) &&
    n >= REWIND_HINT_MIN_PERCENT &&
    n <= REWIND_HINT_MAX_PERCENT
    ? { kind: "ok", value: n }
    : { kind: "invalid" };
}

function parseStartTurns(n: number): StartAnchorField {
  return Number.isInteger(n) &&
    n >= START_ANCHOR_MIN_TURNS &&
    n <= START_ANCHOR_MAX_TURNS
    ? { kind: "ok", value: n }
    : { kind: "invalid" };
}

function parseHintField(record: Record<string, unknown>): HintField {
  if (!("rewindHintAtPercent" in record)) return { kind: "absent" };
  const value = record.rewindHintAtPercent;
  if (value === null || value === false) return { kind: "disabled" };
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (DISABLE_SENTINELS.has(trimmed.toLowerCase())) {
      return { kind: "disabled" };
    }
    // Digits only: decimals, signs, exponents, and empty strings reject.
    return /^\d+$/.test(trimmed)
      ? parseHintPercent(Number(trimmed))
      : { kind: "invalid" };
  }
  if (typeof value === "number") return parseHintPercent(value);
  return { kind: "invalid" };
}

function parseStartAnchorField(
  record: Record<string, unknown>,
): StartAnchorField {
  if (!("anchorStartAfterTurns" in record)) return { kind: "absent" };
  const value = record.anchorStartAfterTurns;
  if (typeof value === "string") {
    const trimmed = value.trim();
    // Digits only, mirroring the hint's string tolerance. There are no
    // disable sentinels for this field (non-optional), so null/false fall
    // through to `invalid` below.
    return /^\d+$/.test(trimmed)
      ? parseStartTurns(Number(trimmed))
      : { kind: "invalid" };
  }
  if (typeof value === "number") return parseStartTurns(value);
  return { kind: "invalid" };
}

/**
 * Parse one already-JSON.parsed config root. A non-object root (null, array,
 * primitive) is `unusable` — a file-level failure so the other layer can
 * still apply. Fields are parsed independently: an invalid hint does not
 * affect the start anchor, and vice versa.
 */
export function parseTreeNavigatorConfig(raw: unknown): ParsedRoot {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { kind: "unusable" };
  }
  const record = raw as Record<string, unknown>;
  return {
    kind: "fields",
    layer: {
      hint: parseHintField(record),
      startAnchor: parseStartAnchorField(record),
    },
  };
}

/**
 * Merge a single field. For the hint, `absent` is the only value that falls
 * through: a project layer that is absent (or file-level unusable, which
 * leaves every field absent) contributes nothing, while `ok`, `disabled`,
 * and field-level `invalid` all win — a present project key is the user's
 * explicit intent for this project.
 */
function mergeHintField(
  globalField: HintField,
  projectField: HintField,
): HintField {
  return projectField.kind === "absent" ? globalField : projectField;
}

/**
 * Merge the start-anchor field. `absent` falls through; `invalid` falls
 * through only to a VALID other layer (warning + fall through per issue
 * #55) — otherwise it stays invalid and the resolver applies the default.
 */
function mergeStartAnchorField(
  globalField: StartAnchorField,
  projectField: StartAnchorField,
): StartAnchorField {
  if (projectField.kind === "ok") return projectField;
  if (projectField.kind === "invalid") {
    return globalField.kind === "ok" ? globalField : projectField;
  }
  return globalField;
}

/**
 * Merge the two layers field by field. File-level failures arrive as
 * all-absent layers (loadLayer maps them that way), so each field's
 * documented precedence applies independently.
 */
export function mergeTreeNavigatorConfig(
  globalLayer: ParsedLayer,
  projectLayer: ParsedLayer,
): ParsedLayer {
  return {
    hint: mergeHintField(globalLayer.hint, projectLayer.hint),
    startAnchor: mergeStartAnchorField(
      globalLayer.startAnchor,
      projectLayer.startAnchor,
    ),
  };
}

type LayerName = "global" | "project";

const ABSENT_LAYER: ParsedLayer = {
  hint: { kind: "absent" },
  startAnchor: { kind: "absent" },
};

interface LayerLoadResult {
  layer: ParsedLayer;
  warnings: string[];
}

function isEnoent(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    (err as { code?: unknown }).code === "ENOENT"
  );
}

async function loadLayer(
  layer: LayerName,
  path: string,
  readFile: (path: string) => Promise<string>,
): Promise<LayerLoadResult> {
  const fileWarning = (message: string): LayerLoadResult => ({
    layer: ABSENT_LAYER,
    warnings: [message],
  });
  let text: string;
  try {
    text = await readFile(path);
  } catch (err) {
    if (isEnoent(err)) return { layer: ABSENT_LAYER, warnings: [] };
    return fileWarning(
      `navigate_tree: could not read ${layer} config at ${path} — that layer was ignored.`,
    );
  }
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch {
    return fileWarning(
      `navigate_tree: invalid JSON in ${layer} config at ${path} — that layer was ignored.`,
    );
  }
  const parsed = parseTreeNavigatorConfig(root);
  if (parsed.kind === "unusable") {
    return {
      layer: ABSENT_LAYER,
      warnings: [
        `navigate_tree: ${layer} config at ${path} must be a JSON object — that layer was ignored.`,
      ],
    };
  }
  const warnings: string[] = [];
  if (parsed.layer.hint.kind === "invalid") {
    warnings.push(
      `navigate_tree: invalid rewindHintAtPercent in ${layer} config at ${path} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`,
    );
  }
  if (parsed.layer.startAnchor.kind === "invalid") {
    warnings.push(
      `navigate_tree: invalid anchorStartAfterTurns in ${layer} config at ${path} — expected integer ${START_ANCHOR_MIN_TURNS}-${START_ANCHOR_MAX_TURNS}; falling back to the other layer's valid value, else the default ${START_ANCHOR_DEFAULT_TURNS}.`,
    );
  }
  return { layer: parsed.layer, warnings };
}

/**
 * Load and merge both config layers.
 *
 * `projectPath` is passed in rather than derived from `cwd` so this module
 * needs no `CONFIG_DIR_NAME` import (the no-pi-imports constraint; the
 * caller computes `join(cwd, CONFIG_DIR_NAME, TREE_NAVIGATOR_CONFIG_FILENAME)`).
 * Warnings are ordered global-then-project; each invalid field contributes
 * its own scoped warning.
 */
export async function loadTreeNavigatorConfig(opts: {
  agentDir: string;
  projectPath: string;
  projectTrusted: boolean;
  readFile?: (path: string) => Promise<string>;
}): Promise<{
  config: {
    rewindHintAtPercent: number | null;
    anchorStartAfterTurns: number;
  };
  warnings: string[];
}> {
  const readFile =
    opts.readFile ?? ((path: string) => fsReadFile(path, "utf8"));

  const warnings: string[] = [];
  const globalResult = await loadLayer(
    "global",
    join(opts.agentDir, TREE_NAVIGATOR_CONFIG_FILENAME),
    readFile,
  );
  warnings.push(...globalResult.warnings);

  let projectLayer: ParsedLayer = ABSENT_LAYER;
  if (opts.projectTrusted) {
    const projectResult = await loadLayer(
      "project",
      opts.projectPath,
      readFile,
    );
    warnings.push(...projectResult.warnings);
    projectLayer = projectResult.layer;
  }

  const effective = mergeTreeNavigatorConfig(globalResult.layer, projectLayer);
  return {
    config: {
      rewindHintAtPercent:
        effective.hint.kind === "ok" ? effective.hint.value : null,
      anchorStartAfterTurns:
        effective.startAnchor.kind === "ok"
          ? effective.startAnchor.value
          : START_ANCHOR_DEFAULT_TURNS,
    },
    warnings,
  };
}
