// tools/runtime/preflight.mjs
// Preflight check to ensure no absolute paths have crept back in
// Runs before dev/build/start to catch regressions early

import { readdirSync, statSync, readFileSync } from "node:fs";
import { join, sep } from "node:path";

const ROOT = process.env.IRIS_ROOT || process.cwd();
const EXCLUDE = [
  `${sep}node_modules${sep}`, 
  `${sep}.git${sep}`, 
  `${sep}dist${sep}`, 
  `${sep}build${sep}`,
  `${sep}.venv${sep}`, 
  `${sep}venv${sep}`,
  `${sep}__pycache__${sep}`,
  `${sep}.cache${sep}`,
  `${sep}.pytest_cache${sep}`,
  `${sep}tools${sep}dawn${sep}`, 
  `${sep}docs${sep}conversations${sep}`, // Preserve historical references
  `${sep}conversations${sep}`, // Preserve all conversation logs
  `${sep}docs${sep}shiporder.txt`, // Old documentation
  `RELEASE_GATE_COMPLETE.md`, // Documentation about the refactoring
  `tools${sep}runtime${sep}README.md` // Documentation about the refactoring
];

const MAX = 2_000_000; // 2 MB per file
const NEEDLE = "C:\\Users\\jason\\Desktop\\tori\\kha";
const exts = new Set([".py",".ts",".tsx",".js",".jsx",".svelte",".wgsl",".txt",".json",".md",".yaml",".yml"]);

function shouldSkip(p) { 
  return EXCLUDE.some(x => p.includes(x)); 
}

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    try {
      const st = statSync(p);
      if (st.isDirectory()) { 
        if (!shouldSkip(p)) yield* walk(p); 
      } else {
        const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
        if (exts.has(ext) && st.size <= MAX && !shouldSkip(p)) yield p;
      }
    } catch { 
      // Ignore permission errors, etc.
    }
  }
}

console.log("Running preflight check for absolute paths...");
const hits = [];
let checked = 0;

for (const f of walk(ROOT)) {
  try {
    checked++;
    if (checked % 1000 === 0) {
      process.stdout.write(`\rChecked ${checked} files...`);
    }
    const content = readFileSync(f, { encoding: "utf8" });
    if (content.includes(NEEDLE)) {
      hits.push(f);
    }
  } catch {
    // Ignore read errors
  }
}

console.log(`\rChecked ${checked} files total.`);

if (hits.length) {
  console.error("\n❌ ERROR: Absolute path references found!");
  console.error("The following files contain hard-coded paths:");
  console.error("-".repeat(60));
  for (const h of hits.slice(0, 50)) {
    console.error(" •", h.replace(ROOT + sep, ""));
  }
  if (hits.length > 50) {
    console.error(` ... and ${hits.length - 50} more files`);
  }
  console.error("-".repeat(60));
  console.error(`Total files with absolute paths: ${hits.length}`);
  console.error("\nPlease use ${IRIS_ROOT} or {PROJECT_ROOT} instead.");
  console.error("Run: python tools\\refactor\\refactor_continue.py");
  process.exit(2);
} else {
  console.log("✅ Preflight OK — no absolute path references found.");
}
