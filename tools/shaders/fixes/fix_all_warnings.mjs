#!/usr/bin/env node
/**
 * Fix all 135 shader warnings in one go
 * - 3 vec3 alignment issues
 * - 13 const/let issues  
 * - 119 dynamic array bounds issues
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// Helper function to add bounds checking
const CLAMP_HELPER = `
fn clamp_index(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}`;

// Files with vec3 issues
const VEC3_FIXES = {
    'avatarShader.wgsl': ['pos', 'vel'],
    'topologicalOverlay.wgsl': ['position']
};

// Files with const/let issues
const CONST_FIXES = {
    'lenticularInterlace.wgsl': ['subpixel_width'],
    'multiDepthWaveSynth.wgsl': ['lambda'],
    'multiViewSynthesis.wgsl': ['phase_tilt_y', 'a', 'b', 'c', 'd', 'e'],
    'propagation.wgsl': ['view_angle'],
    'topologicalOverlay.wgsl': ['s', 'v'],
    'velocityField.wgsl': ['momentum', 'value']
};

function fixVec3Alignment(filePath) {
    const fileName = path.basename(filePath);
    if (!VEC3_FIXES[fileName]) return false;
    
    console.log(`  Fixing vec3 alignment in ${fileName}`);
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    
    for (const field of VEC3_FIXES[fileName]) {
        // Replace vec3<f32> with vec4<f32> in struct definitions
        const regex = new RegExp(`(${field}\\s*:\\s*)vec3<f32>`, 'g');
        if (regex.test(content)) {
            content = content.replace(regex, '$1vec4<f32>');
            console.log(`    ✓ Changed ${field}: vec3<f32> → vec4<f32>`);
            modified = true;
        }
    }
    
    if (modified) {
        fs.writeFileSync(filePath, content);
        return true;
    }
    return false;
}

function fixConstLet(filePath) {
    const fileName = path.basename(filePath);
    if (!CONST_FIXES[fileName]) return false;
    
    console.log(`  Fixing const/let in ${fileName}`);
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    
    for (const varName of CONST_FIXES[fileName]) {
        // Replace let with const for these specific variables
        const regex = new RegExp(`\\blet\\s+${varName}\\b`, 'g');
        if (regex.test(content)) {
            content = content.replace(regex, `const ${varName}`);
            console.log(`    ✓ let ${varName} → const ${varName}`);
            modified = true;
        }
    }
    
    if (modified) {
        fs.writeFileSync(filePath, content);
        return true;
    }
    return false;
}

function addClampHelper(content) {
    // Check if clamp_index already exists
    if (content.includes('fn clamp_index')) {
        return content;
    }
    
    // Find good insertion point - after structs, before first function
    const fnMatch = content.match(/(@vertex|@fragment|@compute|fn\s+\w+)/);
    if (fnMatch) {
        const insertPos = fnMatch.index;
        return content.slice(0, insertPos) + CLAMP_HELPER + '\n\n' + content.slice(insertPos);
    }
    
    // Fallback: add after initial comments
    const lines = content.split('\n');
    let insertLine = 0;
    for (let i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith('//') && lines[i].trim() !== '') {
            insertLine = i;
            break;
        }
    }
    lines.splice(insertLine, 0, '', CLAMP_HELPER, '');
    return lines.join('\n');
}

function fixDynamicArrayBounds(filePath) {
    const fileName = path.basename(filePath);
    console.log(`  Fixing dynamic array bounds in ${fileName}`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Add clamp_index helper if needed
    content = addClampHelper(content);
    
    // Common array patterns to fix
    const patterns = [
        // Simple array access: array[idx]
        { 
            regex: /(\w+)\[([a-zA-Z_]\w*(?:\s*[+\-*/]\s*\w+)*)\]/g,
            replacement: (match, arr, idx) => {
                // Skip if already clamped or is a constant
                if (match.includes('clamp_index') || /^\d+u?$/.test(idx.trim())) {
                    return match;
                }
                // Skip vec/mat constructors
                if (['vec2', 'vec3', 'vec4', 'mat2x2', 'mat3x3', 'mat4x4'].includes(arr)) {
                    return match;
                }
                // Determine array size based on common patterns
                let size = 'arrayLength(&' + arr + ')';
                if (arr === 'tile' || arr === 'shared_data') size = '256u';
                if (arr === 'depths') size = 'MAX_LAYERS';
                if (arr === 'wavelengths' || arr === 'spectral_weights') size = '3u';
                if (arr === 'dispersion_factors') size = '3u';
                if (arr.includes('shared_')) size = '256u';
                
                return `${arr}[clamp_index(${idx}, ${size})]`;
            }
        }
    ];
    
    let modified = false;
    for (const pattern of patterns) {
        const matches = content.match(pattern.regex);
        if (matches && matches.length > 0) {
            content = content.replace(pattern.regex, pattern.replacement);
            console.log(`    ✓ Added bounds checking to ${matches.length} array accesses`);
            modified = true;
        }
    }
    
    if (modified) {
        fs.writeFileSync(filePath, content);
        return true;
    }
    return false;
}

async function main() {
    console.log('🔧 Fixing all 135 shader warnings...\n');
    
    const files = fs.readdirSync(shadersDir)
        .filter(f => f.endsWith('.wgsl'))
        .map(f => path.join(shadersDir, f));
    
    let totalFixed = 0;
    
    // Phase 1: Fix vec3 alignment (3 warnings)
    console.log('Phase 1: Vec3 Alignment Issues');
    for (const file of files) {
        if (fixVec3Alignment(file)) totalFixed++;
    }
    
    // Phase 2: Fix const/let (13 warnings)
    console.log('\nPhase 2: Const/Let Issues');
    for (const file of files) {
        if (fixConstLet(file)) totalFixed++;
    }
    
    // Phase 3: Fix dynamic array bounds (119 warnings)
    console.log('\nPhase 3: Dynamic Array Bounds');
    for (const file of files) {
        if (fixDynamicArrayBounds(file)) totalFixed++;
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`✅ Fixed ${totalFixed} files`);
    console.log('\n📋 Next: Validate to confirm all warnings are gone:');
    console.log('   npm run shaders:sync');
    console.log('   npm run shaders:gate:iphone');
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
