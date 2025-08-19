#!/usr/bin/env node
/**
 * Fix the 13 const/let warnings
 * Simple replacements for immutable variables
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

const CONST_FIXES = [
    { file: 'lenticularInterlace.wgsl', vars: ['subpixel_width'] },
    { file: 'multiDepthWaveSynth.wgsl', vars: ['lambda'] },
    { file: 'multiViewSynthesis.wgsl', vars: ['phase_tilt_y', 'a', 'b', 'c', 'd', 'e'] },
    { file: 'propagation.wgsl', vars: ['view_angle'] },
    { file: 'topologicalOverlay.wgsl', vars: ['s', 'v'] },
    { file: 'velocityField.wgsl', vars: ['momentum', 'value'] }
];

console.log('🔧 Fixing const/let warnings (13 total)\n');

for (const fix of CONST_FIXES) {
    const filePath = path.join(shadersDir, fix.file);
    if (!fs.existsSync(filePath)) continue;
    
    console.log(`${fix.file}:`);
    let content = fs.readFileSync(filePath, 'utf8');
    
    for (const varName of fix.vars) {
        // Match let declarations followed by = (assignment)
        const regex = new RegExp(`(\\s+)let\\s+(${varName}\\s*=)`, 'g');
        if (content.match(regex)) {
            content = content.replace(regex, '$1const $2');
            console.log(`  ✅ let ${varName} → const ${varName}`);
        }
    }
    
    fs.writeFileSync(filePath, content);
}

console.log('\n✅ Done! Run: npm run shaders:sync && npm run shaders:gate:iphone');
