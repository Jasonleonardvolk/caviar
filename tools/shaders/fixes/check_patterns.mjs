#!/usr/bin/env node
/**
 * Check what patterns the validator sees in our files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

console.log('🔍 Checking what we actually have in the files...\n');

// Check normalize.wgsl
const normalizePath = path.join(shadersDir, 'normalize.wgsl');
const content = fs.readFileSync(normalizePath, 'utf8');

// Look for array accesses
const arrayPattern = /(\w+)\[[^\]]+\]/g;
const matches = content.match(arrayPattern);

if (matches) {
    console.log('normalize.wgsl array accesses:');
    const unique = [...new Set(matches)];
    unique.forEach(m => {
        if (m.includes('clamp_index_dyn')) {
            console.log(`  ✅ SAFE: ${m}`);
        } else if (m.includes('[') && !m.startsWith('vec') && !m.startsWith('mat')) {
            console.log(`  ⚠️ UNSAFE: ${m}`);
        }
    });
}

// Check if clamp_index_dyn is defined
if (content.includes('fn clamp_index_dyn')) {
    console.log('\n✅ clamp_index_dyn helper is present');
} else {
    console.log('\n❌ clamp_index_dyn helper is MISSING');
}
