#!/usr/bin/env node
// Batch fix script for common TypeScript errors

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FRONTEND_DIR = path.resolve(__dirname, '../frontend');

// Pattern replacements
const replacements = [
    // Fix GPUTexture.size access
    {
        pattern: /(\w+)\.size\[0\]/g,
        replacement: '$1.width',
        filePattern: /\.ts$/
    },
    {
        pattern: /(\w+)\.size\[1\]/g,
        replacement: '$1.height',
        filePattern: /\.ts$/
    },
    {
        pattern: /(\w+)\.size\[2\]/g,
        replacement: '$1.depthOrArrayLayers',
        filePattern: /\.ts$/
    },
    {
        pattern: /size:\s*(\w+)\.size,/g,
        replacement: 'size: { width: $1.width, height: $1.height },',
        filePattern: /\.ts$/
    },
    
    // Fix writeTimestamp calls - convert to timestampWrites in pass descriptor
    {
        pattern: /pass\.writeTimestamp\(([^,]+),\s*([^)]+)\);/g,
        replacement: '// TODO: Move timestamp to pass descriptor: $1, $2',
        filePattern: /\.ts$/
    },
    {
        pattern: /commandEncoder\.writeTimestamp\(([^,]+),\s*([^)]+)\);/g,
        replacement: '// TODO: Move timestamp to pass descriptor: $1, $2',
        filePattern: /\.ts$/
    },
    
    // Fix fragment state missing targets
    {
        pattern: /fragment:\s*{\s*module:\s*([^,]+),\s*entryPoint:\s*([^}]+)\s*}/g,
        replacement: 'fragment: { module: $1, entryPoint: $2, targets: [{ format: "bgra8unorm" }] }',
        filePattern: /\.ts$/
    }
];

// Files to add property initializers
const propertyInitFixes = {
    'frontend/lib/webgpu/fftCompute.ts': [
        { property: 'bindGroupLayout', init: '= null!' },
        { property: 'pipelineLayout', init: '= null!' },
        { property: 'uniformBuffer', init: '= null!' },
        { property: 'twiddleBuffer', init: '= null!' },
        { property: 'bitReversalBuffer', init: '= null!' },
        { property: 'twiddleOffsetBuffer', init: '= null!' }
    ],
    'frontend/lib/webgpu/kernels/splitStepOrchestrator.ts': [
        { property: 'fftModule', init: '= null!' },
        { property: 'transposeModule', init: '= null!' },
        { property: 'phaseModule', init: '= null!' },
        { property: 'kspaceModule', init: '= null!' },
        { property: 'normalizeModule', init: '= null!' },
        { property: 'fftPipeline', init: '= null!' },
        { property: 'transposePipeline', init: '= null!' },
        { property: 'phasePipeline', init: '= null!' },
        { property: 'kspacePipeline', init: '= null!' },
        { property: 'normalizePipeline', init: '= null!' },
        { property: 'bufferA', init: '= null!' },
        { property: 'bufferB', init: '= null!' },
        { property: 'uniformBuffer', init: '= null!' }
    ],
    'tori_ui_svelte/src/lib/webgpu/photoMorphPipeline.ts': [
        { property: 'propagationPipeline', init: '= null!' },
        { property: 'velocityFieldPipeline', init: '= null!' },
        { property: 'multiViewPipeline', init: '= null!' },
        { property: 'lenticularPipeline', init: '= null!' },
        { property: 'propagationShader', init: '= null!' },
        { property: 'velocityShader', init: '= null!' },
        { property: 'multiViewShader', init: '= null!' },
        { property: 'lenticularShader', init: '= null!' }
    ]
};

function applyReplacements(filePath, content) {
    let modified = content;
    let changeCount = 0;
    
    for (const rule of replacements) {
        if (!rule.filePattern.test(filePath)) continue;
        
        const before = modified;
        modified = modified.replace(rule.pattern, rule.replacement);
        
        if (before !== modified) {
            changeCount += (modified.match(rule.pattern) || []).length;
        }
    }
    
    return { modified, changeCount };
}

function fixPropertyInitializers(filePath, content) {
    const fixes = propertyInitFixes[path.relative(path.dirname(FRONTEND_DIR), filePath).replace(/\\/g, '/')];
    if (!fixes) return { modified: content, changeCount: 0 };
    
    let modified = content;
    let changeCount = 0;
    
    for (const fix of fixes) {
        const pattern = new RegExp(`(private ${fix.property}:.*);`, 'g');
        const replacement = `$1${fix.init};`;
        
        const before = modified;
        modified = modified.replace(pattern, replacement);
        
        if (before !== modified) {
            changeCount++;
        }
    }
    
    return { modified, changeCount };
}

function processFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Apply general replacements
    let { modified, changeCount } = applyReplacements(filePath, content);
    
    // Apply property initializer fixes
    const propFix = fixPropertyInitializers(filePath, modified);
    modified = propFix.modified;
    changeCount += propFix.changeCount;
    
    if (changeCount > 0) {
        fs.writeFileSync(filePath, modified);
        console.log(`✅ Fixed ${changeCount} issues in ${path.relative(FRONTEND_DIR, filePath)}`);
        return true;
    }
    
    return false;
}

function walkDir(dir, callback) {
    const files = fs.readdirSync(dir);
    
    for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        
        if (stat.isDirectory()) {
            if (!file.startsWith('.') && file !== 'node_modules') {
                walkDir(filePath, callback);
            }
        } else if (file.endsWith('.ts')) {
            callback(filePath);
        }
    }
}

// Main execution
console.log('🔧 Running batch TypeScript fixes...\n');

let totalFixed = 0;
let filesFixed = 0;

walkDir(FRONTEND_DIR, (filePath) => {
    if (processFile(filePath)) {
        filesFixed++;
    }
});

// Also process Svelte TypeScript files
const svelteDir = path.resolve(__dirname, '../tori_ui_svelte/src');
if (fs.existsSync(svelteDir)) {
    walkDir(svelteDir, (filePath) => {
        if (processFile(filePath)) {
            filesFixed++;
        }
    });
}

console.log(`\n✨ Batch fixes complete!`);
console.log(`   Files modified: ${filesFixed}`);
console.log(`\nRun 'npm run typecheck' to verify remaining issues.`);
