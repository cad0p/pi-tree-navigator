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
 * Failure semantics (issue #44; authoritative):
 *   - File-level problems (non-ENOENT read failure, invalid JSON, non-object
 *     root) make that layer contribute nothing — the OTHER layer still
 *     applies.
 *   - FIELD-level problems (`rewindHintAtPercent` present but unusable)
 *     disable the feature for the session with NO cross-layer fallback: the
 *     user touched the field, so firing at the other layer's threshold would
 *     be an unintended rewind.
 *   - A missing file is the normal "not configured" state → silent. Warning
 *     on it would toast every session of every user without a config.
 *
 * These helpers do not import pi or log: warnings are returned as strings so
 * `index.ts` can route them through `ctx.ui.notify` (gated on `hasUI`).
 */

import { readFile as fsReadFile } from "node:fs/promises";
import { join } from "node:path";

export const TREE_NAVIGATOR_CONFIG_FILENAME = "tree-navigator.json";
export const REWIND_HINT_MIN_PERCENT = 20;
export const REWIND_HINT_MAX_PERCENT = 95;

/**
 * The outcome of parsing one config layer.
 *
 *   - `absent`:   no file / no key — contributes nothing (falls through).
 *   - `unusable`: file-level failure — contributes nothing (falls through).
 *   - `disabled`: explicit sentinel — wins the merge.
 *   - `ok`:       valid threshold — wins the merge.
 *   - `invalid`:  key present, value unusable — wins the merge (disables the
 *                 session; no cross-layer fallback).
 */
export type ParsedLayer =
  | { kind: "absent" }
  | { kind: "unusable" }
  | { kind: "disabled" }
  | { kind: "ok"; value: number }
  | { kind: "invalid" };

/** Case-insensitive string sentinels that explicitly disable the hint. */
const DISABLE_SENTINELS = new Set(["off", "disabled"]);

function parseThreshold(n: number): ParsedLayer {
  return Number.isInteger(n) &&
    n >= REWIND_HINT_MIN_PERCENT &&
    n <= REWIND_HINT_MAX_PERCENT
    ? { kind: "ok", value: n }
    : { kind: "invalid" };
}

/**
 * Parse one already-JSON.parsed config root. A non-object root (null, array,
 * primitive) is `unusable` — a file-level failure so the other layer can
 * still apply.
 */
export function parseTreeNavigatorConfig(raw: unknown): ParsedLayer {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { kind: "unusable" };
  }
  const record = raw as Record<string, unknown>;
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
      ? parseThreshold(Number(trimmed))
      : { kind: "invalid" };
  }
  if (typeof value === "number") return parseThreshold(value);
  return { kind: "invalid" };
}

/**
 * Merge the two layers. A project layer that is `absent` or `unusable`
 * contributes nothing (global applies); anything else wins — `ok`,
 * `disabled`, and field-level `invalid` — because a present project key is
 * the user's explicit intent for this project.
 */
export function mergeTreeNavigatorConfig(
  globalLayer: ParsedLayer,
  projectLayer: ParsedLayer,
): ParsedLayer {
  if (projectLayer.kind === "absent" || projectLayer.kind === "unusable") {
    return globalLayer;
  }
  return projectLayer;
}

type LayerName = "global" | "project";

interface LayerLoadResult {
  layer: ParsedLayer;
  warning: string | null;
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
  let text: string;
  try {
    text = await readFile(path);
  } catch (err) {
    if (isEnoent(err)) return { layer: { kind: "absent" }, warning: null };
    return {
      layer: { kind: "unusable" },
      warning: `navigate_tree: could not read ${layer} config at ${path} — that layer was ignored.`,
    };
  }
  let root: unknown;
  try {
    root = JSON.parse(text);
  } catch {
    return {
      layer: { kind: "unusable" },
      warning: `navigate_tree: invalid JSON in ${layer} config at ${path} — that layer was ignored.`,
    };
  }
  const parsed = parseTreeNavigatorConfig(root);
  if (parsed.kind === "unusable") {
    return {
      layer: parsed,
      warning: `navigate_tree: ${layer} config at ${path} must be a JSON object — that layer was ignored.`,
    };
  }
  if (parsed.kind === "invalid") {
    return {
      layer: parsed,
      warning: `navigate_tree: invalid rewindHintAtPercent in ${layer} config at ${path} — expected integer ${REWIND_HINT_MIN_PERCENT}-${REWIND_HINT_MAX_PERCENT} or a disable sentinel (null, false, "off", "disabled"); rewind hint disabled for this session.`,
    };
  }
  return { layer: parsed, warning: null };
}

/**
 * Load and merge both config layers.
 *
 * `projectPath` is passed in rather than derived from `cwd` so this module
 * needs no `CONFIG_DIR_NAME` import (the no-pi-imports constraint; the
 * caller computes `join(cwd, CONFIG_DIR_NAME, TREE_NAVIGATOR_CONFIG_FILENAME)`).
 * Warnings are ordered global-then-project, at most one per layer.
 */
export async function loadTreeNavigatorConfig(opts: {
  agentDir: string;
  projectPath: string;
  projectTrusted: boolean;
  readFile?: (path: string) => Promise<string>;
}): Promise<{
  config: { rewindHintAtPercent: number | null };
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
  if (globalResult.warning) warnings.push(globalResult.warning);

  let projectLayer: ParsedLayer = { kind: "absent" };
  if (opts.projectTrusted) {
    const projectResult = await loadLayer(
      "project",
      opts.projectPath,
      readFile,
    );
    if (projectResult.warning) warnings.push(projectResult.warning);
    projectLayer = projectResult.layer;
  }

  const effective = mergeTreeNavigatorConfig(globalResult.layer, projectLayer);
  return {
    config: {
      rewindHintAtPercent: effective.kind === "ok" ? effective.value : null,
    },
    warnings,
  };
}
