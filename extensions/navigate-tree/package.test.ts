// Every runtime import of the shipped extension is a peer dependency (pi
// provides it), and the dev copies pin the floor tests run against.
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(
  readFileSync(join(here, "..", "..", "package.json"), "utf8"),
) as {
  peerDependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};

/** Package names imported by the shipped (non-test) sources. */
function shippedImports(): Set<string> {
  const names = new Set<string>();
  for (const file of readdirSync(here)) {
    if (!file.endsWith(".ts") || file.endsWith(".test.ts")) continue;
    const source = readFileSync(join(here, file), "utf8");
    for (const match of source.matchAll(/from "([^"./][^"]*)"/g)) {
      const spec = match[1];
      if (spec.startsWith("node:")) continue;
      const parts = spec.split("/");
      names.add(spec.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0]);
    }
  }
  return names;
}

describe("package.json dependencies", () => {
  it("every package the shipped sources import is a peer dependency", () => {
    const peers = pkg.peerDependencies ?? {};
    for (const name of shippedImports()) {
      assert.ok(name in peers, `${name} must be a peerDependency`);
    }
  });

  it("@earendil-works peers share one range", () => {
    const ranges = Object.entries(pkg.peerDependencies ?? {})
      .filter(([name]) => name.startsWith("@earendil-works/"))
      .map(([, range]) => range);
    assert.ok(ranges.length > 0);
    assert.deepEqual(new Set(ranges), new Set([">=0.87.0"]));
  });

  it("pi-tui dev copy is pinned to the version pi-coding-agent resolves to (the floor tests run against)", () => {
    const resolved = (name: string) =>
      (
        JSON.parse(
          readFileSync(
            join(here, "..", "..", "node_modules", name, "package.json"),
            "utf8",
          ),
        ) as { version: string }
      ).version;
    const agent = resolved("@earendil-works/pi-coding-agent");
    assert.equal(pkg.devDependencies?.["@earendil-works/pi-tui"], agent);
    assert.equal(resolved("@earendil-works/pi-tui"), agent);
  });
});
