#!/usr/bin/env node
// fix_ts_errors.js - Run with: node fix_ts_errors.js

const fs = require('fs');
const path = require('path');

// Fix 1: Remove Uint8Array wrapping from buffer writes
function fixBufferWrites(filePath) {
    if (!fs.existsSync(filePath)) return;
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Remove new Uint8Array(...buffer) wrapping
    content = content.replace(/new Uint8Array\((.*?)\.buffer\)/g, '$1');
    
    fs.writeFileSync(filePath, content);
    console.log(`Fixed buffer writes in ${filePath}`);
}

// Fix 2: Add ort namespace import
function fixOrtImports(filePath) {
    if (!fs.existsSync(filePath)) return;
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Check if already has ort import
    if (!content.includes('import * as ort')) {
        // Replace the import line
        content = content.replace(
            /import\s*\{([^}]+)\}\s*from\s*['"]onnxruntime-web['"]/,
            "import * as ort from 'onnxruntime-web';\nimport {$1} from 'onnxruntime-web'"
        );
    }
    
    fs.writeFileSync(filePath, content);
    console.log(`Fixed ORT imports in ${filePath}`);
}

// Fix 3: Rename performance variable
function fixPerformanceVar(filePath) {
    if (!fs.existsSync(filePath)) return;
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Fix the specific conflict
    content = content.replace(
        /const performance = this\.calculatePerformance/g,
        'const perfMetrics = this.calculatePerformance'
    );
    content = content.replace(
        /return performance;(\s*\/\/ PerformanceMetrics)/g,
        'return perfMetrics;$1'
    );
    
    fs.writeFileSync(filePath, content);
    console.log(`Fixed performance variable in ${filePath}`);
}

// Fix 4: Fix compilationInfo type
function fixCompilationInfo(filePath) {
    if (!fs.existsSync(filePath)) return;
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Add type assertion
    content = content.replace(
        /const info = await shaderModule\.compilationInfo\(\);/g,
        'const info = await (shaderModule as any).compilationInfo();'
    );
    
    fs.writeFileSync(filePath, content);
    console.log(`Fixed compilationInfo in ${filePath}`);
}

// Apply fixes
console.log('Applying TypeScript fixes...\n');

// Buffer write fixes
fixBufferWrites('frontend/lib/webgpu/quilt/WebGPUQuiltGenerator.ts');
fixBufferWrites('tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts');

// ORT imports
fixOrtImports('frontend/lib/webgpu/kernels/onnxWaveOpRunner.ts');

// Performance variable
fixPerformanceVar('frontend/lib/webgpu/kernels/schrodingerBenchmark.ts');

// Compilation info
fixCompilationInfo('frontend/lib/webgpu/fftCompute.ts');

console.log('\nFixes applied! Run: npx tsc -p frontend/tsconfig.json --noEmit');
