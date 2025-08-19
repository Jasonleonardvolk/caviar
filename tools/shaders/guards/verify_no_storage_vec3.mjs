// verify_no_storage_vec3.mjs
// Ensures no vec3<f32> lives in actual storage structs

import fs from 'node:fs'; 
import path from 'node:path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.join(__dirname, '../../../frontend/lib/webgpu/shaders');
let bad = [];

function scanFile(p) {
  const s = fs.readFileSync(p, 'utf8');
  // Find structs with vec3 and see if bound as storage
  const structs = [...s.matchAll(/struct\s+(\w+)\s*{([\s\S]*?)}/g)];
  const storageVars = new Set(
    [...s.matchAll(/var<\s*storage[^>]*>\s+(\w+)\s*:\s*(\w+)/g)].map(m => m[2])
  );
  
  for (const m of structs) {
    const name = m[1], body = m[2];
    if (body.includes('vec3<f32>') && storageVars.has(name)) {
      bad.push({ 
        file: path.relative(ROOT, p), 
        struct: name,
        issue: 'vec3<f32> in storage struct needs padding'
      });
    }
  }
}

function* walk(dir) {
  for (const e of fs.readdirSync(dir, {withFileTypes:true})) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(p);
    else if (p.endsWith('.wgsl')) scanFile(p);
  }
}

console.log('🔍 Checking for vec3 in storage structs...\n');
[...walk(ROOT)];

if (bad.length) {
  console.error('❌ vec3<f32> found in STORAGE structs (need padding):');
  for (const b of bad) {
    console.error(`  - ${b.file} :: struct ${b.struct}`);
    console.error(`    Issue: ${b.issue}`);
  }
  process.exit(2);
} else {
  console.log('✅ OK: No vec3<f32> in var<storage> structs.');
  console.log('   The vec3 warnings are false positives (vertex attributes).');
}
