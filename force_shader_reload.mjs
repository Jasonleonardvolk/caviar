#!/usr/bin/env node
// Force shader reload by clearing ShaderLoader cache

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Read the shader file
const shaderPath = join(__dirname, 'frontend/lib/webgpu/shaders/wavefieldEncoder_optimized.wgsl');
const shaderContent = readFileSync(shaderPath, 'utf8');

// Check if it has the correct binding
if (shaderContent.includes('var<storage, read>')) {
    console.log('✅ Shader file has correct storage binding');
} else {
    console.log('❌ ERROR: Shader still using uniform binding!');
    process.exit(1);
}

// Force timestamp update to invalidate cache
const shaderWithTimestamp = shaderContent.replace(
    /\/\/ Enhanced wavefield encoder/,
    `// Enhanced wavefield encoder - Updated ${new Date().toISOString()}`
);

writeFileSync(shaderPath, shaderWithTimestamp);
console.log('✅ Updated shader timestamp to force reload');

// Also check the other location
const shaderPath2 = join(__dirname, 'frontend/shaders/wavefieldEncoder_optimized.wgsl');
try {
    const shader2Content = readFileSync(shaderPath2, 'utf8');
    const shader2WithTimestamp = shader2Content.replace(
        /\/\/ Enhanced wavefield encoder/,
        `// Enhanced wavefield encoder - Updated ${new Date().toISOString()}`
    );
    writeFileSync(shaderPath2, shader2WithTimestamp);
    console.log('✅ Updated second shader timestamp');
} catch (e) {
    console.log('⚠️  Second shader location not found');
}

console.log('\n🔄 Now restart TORI to load the updated shaders');
