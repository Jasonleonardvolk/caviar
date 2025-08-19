#!/usr/bin/env node
/**
 * Find what's actually causing vec3 warnings
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

console.log('🔍 Finding vec3 fields in storage buffers...\n');

const files = ['avatarShader.wgsl', 'topologicalOverlay.wgsl'];

for (const fileName of files) {
    const filePath = path.join(shadersDir, fileName);
    if (!fs.existsSync(filePath)) continue;
    
    const content = fs.readFileSync(filePath, 'utf8');
    console.log(`\n${fileName}:`);
    
    // Find all vec3<f32> declarations
    const vec3Pattern = /(\w+)\s*:\s*vec3<f32>/g;
    let match;
    while ((match = vec3Pattern.exec(content)) !== null) {
        const fieldName = match[1];
        const lineNum = content.substring(0, match.index).split('\n').length;
        console.log(`  Line ${lineNum}: ${fieldName}: vec3<f32>`);
        
        // Check if it's in a storage buffer context
        const beforeMatch = content.substring(Math.max(0, match.index - 500), match.index);
        if (beforeMatch.includes('var<storage') || beforeMatch.includes('struct')) {
            console.log(`    ⚠️  Might be in storage buffer context`);
        }
    }
}
