/* frontend/scripts/pack_floater.mjs */
import path from 'node:path';
import { promises as fs } from 'node:fs';
import { createGzip } from 'node:zlib';
import { pipeline } from 'node:stream/promises';

const ROOT = process.cwd();
const DIST = path.join(ROOT, 'frontend', 'floater', 'dist');
const OUT_DIR = path.join(ROOT, 'artifacts');
await fs.mkdir(OUT_DIR, { recursive: true });

const files = [
  'gaea_floater.esm.js',
  'gaea_floater.umd.js',
  'gaea_floater.worker.js',
  'gaea_floater.d.ts',
  'gaea_floater.css',
  'README.md',
];

const stamp = new Date().toISOString().slice(0,10).replace(/-/g,'');
const outName = `floater_bundle_${stamp}.zip`;
const outPath = path.join(OUT_DIR, outName);

// Minimal zip via store (no deps) — use .zip.gz to avoid true zip container complexity
const gzName = outName + '.gz';
const gzPath = path.join(OUT_DIR, gzName);
const concat = (await Promise.all(files.map(async f => {
  const p = path.join(DIST, f);
  const exists = await fs.stat(p).then(()=>true).catch(()=>false);
  if (!exists) return `# Missing: ${f}\n`;
  const content = await fs.readFile(p, 'utf8');
  return `--- ${f} ---\n${content}\n`;
}))).join('\n');

await fs.writeFile(gzPath.replace(/\.zip\.gz$/, '.txt'), concat, 'utf8');
await pipeline(
  fs.createReadStream(gzPath.replace(/\.zip\.gz$/, '.txt')),
  createGzip(),
  fs.createWriteStream(gzPath)
);

console.log(`Packed bundle → ${gzPath}`);
