// TypeScript types for the engine's contracts (contracts/*.schema.json), for the backend and the dashboard.
//
//   node scripts/contract_types.mjs <out.ts> [<out2.ts> …]
//   (needs json-schema-to-typescript: NODE_PATH=<dir with it>/node_modules)
//
// One exported type per route (DashHome, DashPortrait, AdminLive, …). Regenerate after the contracts change.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { compile } = require("json-schema-to-typescript");
const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.join(here, "..", "contracts");
const outs = process.argv.slice(2);
if (!outs.length) throw new Error("usage: node scripts/contract_types.mjs <out.ts> …");

const pascal = (s) => s.split(/[^a-zA-Z0-9]+/).filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join("");
const parts = [
  "// Generated from kawach-ai-engine contracts/*.schema.json by scripts/contract_types.mjs. Do not edit by hand.",
  "// The shapes are learnt from the synthetic test world; a field marked optional was missing in some responses.",
  "",
];
for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".schema.json")).sort()) {
  const schema = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
  const name = pascal(file.replace(/\.schema\.json$/, ""));
  delete schema.$schema;
  schema.title = name;
  const ts = await compile(schema, name, { bannerComment: "", additionalProperties: false, unknownAny: true, strictIndexSignatures: false,
    format: false, declareExternallyReferenced: true });
  // an object never seen with keys: any object (eslint forbids the bare {} type)
  parts.push(ts.trim().replace(/\{\s*\}/g, "Record<string, unknown>"), "");
}
for (const out of outs) {
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, parts.join("\n"));
  console.log(`wrote ${out}`);
}
