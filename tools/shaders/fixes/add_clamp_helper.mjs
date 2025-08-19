#!/usr/bin/env node
/**
 * Add clamp_index_dyn helper to files with array warnings
 * This just adds the helper - you'll still need to wrap the arrays manually
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// Files that need the clamp helper (have array warnings)
const FILES_NEEDING_CLAMP = [
    'bitReversal.wgsl',
    'butterflyStage.wgsl',
    'fftShift.wgsl',
    'multiDepthWaveSynth.wgsl',
    'multiViewSynthesis.wgsl',
    'normalize.wgsl',
    'phaseOcclusion.wgsl',
    'propagation.wgsl',
    'topologicalOverlay.wgsl',
    'transpose.wgsl',
    'velocityField.wgsl',
    'wavefieldEncoder.wgsl',
    'wavefieldEncoder_optimized.wgsl'
];

const CLAMP_HELPER = `
fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}`;

console.log('🔧 Adding clamp_index_dyn helper to files with array warnings\n');

for (const fileName of FILES_NEEDING_CLAMP) {
    const filePath = path.join(shadersDir, fileName);
    if (!fs.existsSync(filePath)) continue;
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Skip if already has the helper
    if (content.includes('fn clamp_index_dyn')) {
        console.log(`${fileName}: ⏭️  Already has clamp helper`);
        continue;
    }
    
    // Find insertion point - after structs/constants, before first function
    let insertPos = 0;
    
    // Try to find first @compute/@vertex/@fragment or fn declaration
    const fnMatch = content.match(/(@compute|@vertex|@fragment|fn\s+\w+)/);
    if (fnMatch) {
        insertPos = fnMatch.index;
    } else {
        // Fall back to after initial comments and structs
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line.startsWith('//') && 
                !line.startsWith('struct') && 
                !line.startsWith('const') &&
                line.length > 0) {
                insertPos = content.indexOf(lines[i]);
                break;
            }
        }
    }
    
    // Insert the helper
    content = content.slice(0, insertPos) + CLAMP_HELPER + '\n\n' + content.slice(insertPos);
    fs.writeFileSync(filePath, content);
    console.log(`${fileName}: ✅ Added clamp_index_dyn helper`);
}

console.log('\n✅ Helpers added!');
console.log('\nNext steps for each file:');
console.log('1. For workgroup arrays (tile, shared_data): use clamp_index_dyn(idx, TILE_SIZE)');
console.log('2. For storage arrays: use clamp_index_dyn(idx, arrayLength(&array))');
console.log('3. For fixed arrays: use clamp_index_dyn(idx, KNOWN_SIZE)');
console.log('\nExample: input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]');
