import path from 'node:path';
import { promises as fs } from 'node:fs';

const ROOT = process.cwd();
const DIST = path.join(ROOT, 'frontend', 'floater', 'dist');
const OUT_DIR = path.join(ROOT, 'artifacts');

await fs.mkdir(OUT_DIR, { recursive: true });

const files = [
  'gaea_floater.esm.js',
  'gaea_floater.worker.js'
];

const stamp = new Date().toISOString().slice(0,10).replace(/-/g,'');
const outZip = path.join(OUT_DIR, `floater_bundle_${stamp}.zip`);

// For now, just create a manifest file
const manifest = {
  name: '@holographic/display',
  version: '0.1.0',
  files: [],
  timestamp: new Date().toISOString()
};

for (const file of files) {
  const fullPath = path.join(DIST, file);
  try {
    const stats = await fs.stat(fullPath);
    manifest.files.push({ name: file, size: stats.size });
  } catch (err) {
    console.warn(`File not found: ${file}`);
  }
}

await fs.writeFile(
  path.join(OUT_DIR, `floater_manifest_${stamp}.json`),
  JSON.stringify(manifest, null, 2),
  'utf8'
);

console.log(`Floater manifest created → artifacts/floater_manifest_${stamp}.json`);
console.log('Files included:', manifest.files);
