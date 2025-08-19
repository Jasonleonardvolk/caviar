#!/usr/bin/env node
/**
 * Fix the 3 vec3 alignment warnings
 * Super targeted - just 2 files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

console.log('🔧 Fixing vec3 alignment (3 warnings total)\n');

// Fix 1: avatarShader.wgsl
const avatar = path.join(shadersDir, 'avatarShader.wgsl');
if (fs.existsSync(avatar)) {
    console.log('avatarShader.wgsl:');
    let content = fs.readFileSync(avatar, 'utf8');
    
    // Find and replace pos and vel in Particle struct
    content = content.replace(/(\s+)(pos\s*:\s*)vec3<f32>/g, '$1$2vec4<f32>');
    content = content.replace(/(\s+)(vel\s*:\s*)vec3<f32>/g, '$1$2vec4<f32>');
    
    fs.writeFileSync(avatar, content);
    console.log('  ✅ Changed pos: vec3<f32> → vec4<f32>');
    console.log('  ✅ Changed vel: vec3<f32> → vec4<f32>');
}

// Fix 2: topologicalOverlay.wgsl
const topo = path.join(shadersDir, 'topologicalOverlay.wgsl');
if (fs.existsSync(topo)) {
    console.log('\ntopologicalOverlay.wgsl:');
    let content = fs.readFileSync(topo, 'utf8');
    
    // Find and replace position in ChargeData struct
    content = content.replace(/(\s+)(position\s*:\s*)vec3<f32>/g, '$1$2vec4<f32>');
    
    fs.writeFileSync(topo, content);
    console.log('  ✅ Changed position: vec3<f32> → vec4<f32>');
}

console.log('\n✅ Done! Run: npm run shaders:sync && npm run shaders:gate:iphone');
