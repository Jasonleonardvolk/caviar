#!/usr/bin/env node
/**
 * Find and fix the 2 remaining vec3 warnings in avatarShader.wgsl
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const filePath = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders', 'avatarShader.wgsl');

console.log('🔍 Finding vec3 fields in avatarShader.wgsl...\n');

let content = fs.readFileSync(filePath, 'utf8');

// Find all vec3<f32> declarations in structs
const structPattern = /struct\s+(\w+)\s*\{([^}]+)\}/g;
let match;

while ((match = structPattern.exec(content)) !== null) {
    const structName = match[1];
    const structBody = match[2];
    
    if (structBody.includes('vec3<f32>')) {
        console.log(`Found struct ${structName} with vec3 fields:`);
        
        // Find each vec3 field
        const fieldPattern = /(\w+)\s*:\s*vec3<f32>/g;
        let fieldMatch;
        while ((fieldMatch = fieldPattern.exec(structBody)) !== null) {
            console.log(`  - ${fieldMatch[1]}: vec3<f32>`);
        }
        
        // Check if this struct is used in storage
        const storagePattern = new RegExp(`var<storage[^>]*>\\s+\\w+\\s*:\\s*(array<)?${structName}`, 'g');
        if (storagePattern.test(content)) {
            console.log(`  ⚠️ ${structName} is used in storage buffer - needs vec4 conversion!\n`);
            
            // Convert vec3 to vec4 in this struct
            const newStructBody = structBody.replace(/(\w+\s*:\s*)vec3<f32>/g, '$1vec4<f32>');
            const newStruct = `struct ${structName} {${newStructBody}}`;
            content = content.replace(match[0], newStruct);
            console.log(`  ✅ Converted all vec3<f32> to vec4<f32> in ${structName}`);
        }
    }
}

fs.writeFileSync(filePath, content);
console.log('\nDone! Run: npm run shaders:sync ; npm run shaders:gate:iphone');
