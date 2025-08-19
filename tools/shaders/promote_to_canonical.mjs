#!/usr/bin/env node
/**
 * promote_to_canonical.mjs
 * Promotes/moves specific shaders to canonical location
 */
import fs from 'fs';
import path from 'path';

const args = new Map(process.argv.slice(2).map(a => a.includes('=') ? a.split('=') : [a, true]));
const REPO = path.resolve(String(args.get('--repo') || process.cwd()));
const NAMES = args.get('--names') ? String(args.get('--names')).split(',') : [];

const CANONICAL = path.join(REPO, 'frontend', 'lib', 'webgpu', 'shaders');
const SEARCH_PATHS = [
  path.join(REPO, 'frontend', 'hybrid'),
  path.join(REPO, 'frontend', 'hybrid', 'wgsl'),
  path.join(REPO, 'frontend', 'public', 'hybrid', 'wgsl'),
  path.join(REPO, 'frontend', 'shaders')
];

console.log('Promoting shaders to canonical location...');
console.log(`Canonical: ${path.relative(REPO, CANONICAL)}`);

// Ensure canonical exists
if (!fs.existsSync(CANONICAL)) {
  fs.mkdirSync(CANONICAL, { recursive: true });
}

const promoted = [];
const skipped = [];

for (const name of NAMES) {
  const filename = name.endsWith('.wgsl') ? name : `${name}.wgsl`;
  const canonicalPath = path.join(CANONICAL, filename);
  
  if (fs.existsSync(canonicalPath)) {
    skipped.push(filename);
    continue;
  }
  
  // Find in search paths
  let found = false;
  for (const searchPath of SEARCH_PATHS) {
    const sourcePath = path.join(searchPath, filename);
    if (fs.existsSync(sourcePath)) {
      fs.copyFileSync(sourcePath, canonicalPath);
      promoted.push({
        name: filename,
        from: path.relative(REPO, sourcePath),
        to: path.relative(REPO, canonicalPath)
      });
      found = true;
      break;
    }
  }
  
  if (!found) {
    console.log(`  ⚠️ ${filename} not found in any search path`);
  }
}

if (promoted.length > 0) {
  console.log(`\n✅ Promoted ${promoted.length} shaders:`);
  promoted.forEach(p => console.log(`  ${p.name}: ${p.from} → ${p.to}`));
}

if (skipped.length > 0) {
  console.log(`\n⏭️ Skipped ${skipped.length} shaders (already in canonical):`);
  skipped.forEach(s => console.log(`  ${s}`));
}
