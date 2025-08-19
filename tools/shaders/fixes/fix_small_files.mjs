#!/usr/bin/env node
/**
 * Quick fix for the 2 smallest files - normalize.wgsl and fftShift.wgsl
 * Only 4 warnings total - easy wins!
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// Fix normalize.wgsl (2 warnings)
console.log('🔧 Fixing normalize.wgsl...');
let normalizePath = path.join(shadersDir, 'normalize.wgsl');
let content = fs.readFileSync(normalizePath, 'utf8');

content = content.replace(/input\[([^\]]+)\]/g, (match, idx) => {
    if (match.includes('clamp_index_dyn')) return match;
    return `input[clamp_index_dyn(${idx}, arrayLength(&input))]`;
});

content = content.replace(/output\[([^\]]+)\]/g, (match, idx) => {
    if (match.includes('clamp_index_dyn')) return match;
    return `output[clamp_index_dyn(${idx}, arrayLength(&output))]`;
});

fs.writeFileSync(normalizePath, content);
console.log('  ✅ Fixed 2 array accesses');

// Fix fftShift.wgsl (2 warnings)
console.log('\n🔧 Fixing fftShift.wgsl...');
let fftPath = path.join(shadersDir, 'fftShift.wgsl');
content = fs.readFileSync(fftPath, 'utf8');

content = content.replace(/input\[([^\]]+)\]/g, (match, idx) => {
    if (match.includes('clamp_index_dyn')) return match;
    return `input[clamp_index_dyn(${idx}, arrayLength(&input))]`;
});

content = content.replace(/output\[([^\]]+)\]/g, (match, idx) => {
    if (match.includes('clamp_index_dyn')) return match;
    return `output[clamp_index_dyn(${idx}, arrayLength(&output))]`;
});

fs.writeFileSync(fftPath, content);
console.log('  ✅ Fixed 2 array accesses');

console.log('\n✅ Total: Fixed 4 array accesses');
console.log('Run: npm run shaders:sync ; npm run shaders:gate:iphone');
