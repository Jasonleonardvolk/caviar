#!/usr/bin/env node
/**
 * Manually check and fix the 6 broken shader files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// First, let's see what's actually in these files
const problemFiles = [
    'hybridWavefieldBlend.wgsl',
    'lightFieldComposer.wgsl',
    'multiDepthWaveSynth.wgsl',
    'multiViewSynthesis.wgsl',
    'phaseOcclusion.wgsl',
    'topologicalOverlay.wgsl'
];

console.log('🔍 Checking the actual content of problem files...\n');

for (const fileName of problemFiles) {
    const filePath = path.join(shadersDir, fileName);
    console.log(`📄 ${fileName}`);
    
    if (!fs.existsSync(filePath)) {
        console.log('   ❌ File not found!');
        continue;
    }
    
    const content = fs.readFileSync(filePath, 'utf8');
    const firstChar = content[0];
    const firstLine = content.split('\n')[0];
    
    console.log(`   First char: "${firstChar}" (code: ${firstChar.charCodeAt(0)})`);
    console.log(`   First line: ${firstLine.substring(0, 60)}...`);
    
    // Check for specific issues
    if (firstChar === '{') {
        console.log('   ❌ FILE STARTS WITH { - This is corrupted/JSON!');
    } else if (firstChar.charCodeAt(0) === 0xFEFF) {
        console.log('   ⚠️  File has BOM character');
    } else {
        console.log('   ✅ File starts correctly');
    }
    
    // Check for specific syntax issues
    if (fileName === 'topologicalOverlay.wgsl') {
        if (content.includes('var<storage, read> charges')) {
            console.log('   ⚠️  Has read-only storage buffer that needs write access');
        }
    }
    
    if (fileName === 'lightFieldComposer.wgsl') {
        const match = content.match(/textureLoad\([^)]+\)/);
        if (match) {
            const args = match[0].split(',').length - 1;
            console.log(`   textureLoad has ${args} arguments (needs 3 + mip level = 4)`);
        }
    }
    
    console.log('');
}