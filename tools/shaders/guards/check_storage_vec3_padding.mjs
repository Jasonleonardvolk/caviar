// check_storage_vec3_padding.mjs
// Guards against storage vec3 without padding

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const root = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');
const offenders = [];

function scan(file) {
  const src = fs.readFileSync(file, 'utf8');
  const structRe = /struct\s+([A-Za-z_]\w*)\s*{([\s\S]*?)}/g;
  
  for (const m of src.matchAll(structRe)) {
    const structName = m[1];
    const body = m[2];
    
    // Check if this is likely a storage struct (not vertex/fragment IO)
    const isStorageStruct = !body.includes('@location') && !body.includes('@builtin');
    
    if (isStorageStruct) {
      // naive: flag any vec3<f32> not immediately followed by f32 or another vec3
      const lines = body.split('\n').map(l => l.trim()).filter(Boolean);
      
      for (let i = 0; i < lines.length; i++) {
        if (/\bvec3<f32>\b/.test(lines[i])) {
          const next = lines[i + 1] || '';
          if (!/\bf32\b/.test(next) && !/\bvec3<f32>\b/.test(next)) {
            offenders.push({
              file: path.relative(root, file),
              struct: structName,
              line: i + 1,
              snippet: lines[i]
            });
          }
        }
      }
    }
  }
}

function* walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      yield* walk(p);
    } else if (p.endsWith('.wgsl')) {
      scan(p);
    }
  }
}

// Run the scan
[...walk(root)];

if (offenders.length) {
  console.error('⚠️  vec3<f32> in storage struct without padding:');
  offenders.forEach(o => {
    console.error(`  - ${o.file} [${o.struct}]:${o.line}  ${o.snippet}`);
  });
  process.exit(2);
}

console.log('✅ vec3 padding guard passed - no unpadded vec3 in storage structs');
