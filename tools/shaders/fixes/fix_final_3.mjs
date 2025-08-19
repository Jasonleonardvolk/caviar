#!/usr/bin/env node
/**
 * Fix the final 3 shader errors
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

console.log('🔧 Fixing the final 3 shader errors...\n');

// Fix 1: lightFieldComposer.wgsl - textureLoad needs different fix
const file1 = path.join(shadersDir, 'lightFieldComposer.wgsl');
console.log('1. lightFieldComposer.wgsl - Fix textureLoad arguments');
if (fs.existsSync(file1)) {
    let content = fs.readFileSync(file1, 'utf8');
    // The error shows it has vec2<i32>(local_x, local_y, 0) which is wrong - vec2 takes 2 args not 3
    // It should be textureLoad(tex, coords, arrayIndex, mipLevel)
    content = content.replace(
        /textureLoad\(([^,]+),\s*vec2<i32>\(([^,]+),\s*([^,]+),\s*0\),\s*([^)]+)\)/g,
        'textureLoad($1, vec2<i32>($2, $3), $4, 0)'
    );
    fs.writeFileSync(file1, content);
    console.log('   ✅ Fixed textureLoad calls\n');
}

// Fix 2: multiDepthWaveSynth.wgsl - storage write → read_write
const file2 = path.join(shadersDir, 'multiDepthWaveSynth.wgsl');
console.log('2. multiDepthWaveSynth.wgsl - Fix storage buffer access');
if (fs.existsSync(file2)) {
    let content = fs.readFileSync(file2, 'utf8');
    content = content.replace(
        'var<storage, write>',
        'var<storage, read_write>'
    );
    fs.writeFileSync(file2, content);
    console.log('   ✅ Changed write to read_write\n');
}

// Fix 3: phaseOcclusion.wgsl - storage write → read_write
const file3 = path.join(shadersDir, 'phaseOcclusion.wgsl');
console.log('3. phaseOcclusion.wgsl - Fix storage buffer access');
if (fs.existsSync(file3)) {
    let content = fs.readFileSync(file3, 'utf8');
    content = content.replace(
        'var<storage, write>',
        'var<storage, read_write>'
    );
    fs.writeFileSync(file3, content);
    console.log('   ✅ Changed write to read_write\n');
}

console.log('✅ Done! Now run: npm run shaders:gate:iphone');
