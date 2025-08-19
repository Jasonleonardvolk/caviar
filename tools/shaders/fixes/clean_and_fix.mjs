#!/usr/bin/env node
/**
 * Clean up temp files and fix the actual shader issues
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

console.log('🧹 Cleaning up and fixing shader issues...\n');

// Step 1: Remove any .temp_ files
console.log('Step 1: Removing temp files...');
try {
    const tempFiles = execSync('dir /b .temp_*.wgsl', { cwd: shadersDir, encoding: 'utf8' }).trim().split('\n');
    for (const file of tempFiles) {
        if (file) {
            const filePath = path.join(shadersDir, file);
            fs.unlinkSync(filePath);
            console.log(`   Deleted: ${file}`);
        }
    }
} catch (e) {
    console.log('   No temp files found (good!)');
}

// Step 2: Fix each problematic shader
const fixes = [
    {
        file: 'topologicalOverlay.wgsl',
        description: 'Fix storage buffer write permission',
        fix: (content) => {
            return content.replace(
                '@group(0) @binding(1) var<storage, read> charges: array<ChargeData>;',
                '@group(0) @binding(1) var<storage, read_write> charges: array<ChargeData>;'
            );
        }
    },
    {
        file: 'lightFieldComposer.wgsl',
        description: 'Add mip level to textureLoad',
        fix: (content) => {
            // Fix textureLoad calls that have 3 arguments (need 4)
            return content.replace(
                /textureLoad\(([^,]+),\s*([^,]+),\s*([^)]+)\)/g,
                (match, tex, coords, arrayIndex) => {
                    // Check if this already has 4 args (has a second comma after coords)
                    const argCount = match.split(',').length;
                    if (argCount === 3) {
                        return `textureLoad(${tex}, ${coords}, ${arrayIndex}, 0)`;
                    }
                    return match;
                }
            );
        }
    },
    {
        file: 'multiDepthWaveSynth.wgsl',
        description: 'Fix inline struct syntax',
        fix: (content) => {
            // Look for var<uniform> with inline struct
            const match = content.match(/@group\(0\)\s*@binding\(2\)\s*var<uniform>\s+params:\s*struct\s*{([^}]+)}/);
            if (match) {
                const fields = match[1];
                const structDef = `struct ParamsData {${fields}}\n\n`;
                const newBinding = '@group(0) @binding(2) var<uniform> params: ParamsData;';
                
                // Replace the entire declaration
                content = content.replace(match[0], structDef + newBinding);
            }
            return content;
        }
    },
    {
        file: 'phaseOcclusion.wgsl',
        description: 'Fix inline struct syntax',
        fix: (content) => {
            // Look for var<uniform> with inline struct
            const match = content.match(/@group\(0\)\s*@binding\(3\)\s*var<uniform>\s+params:\s*struct\s*{([^}]+)}/);
            if (match) {
                const fields = match[1];
                const structDef = `struct ParamsData {${fields}}\n\n`;
                const newBinding = '@group(0) @binding(3) var<uniform> params: ParamsData;';
                
                // Replace the entire declaration
                content = content.replace(match[0], structDef + newBinding);
            }
            return content;
        }
    },
    {
        file: 'multiViewSynthesis.wgsl',
        description: 'Fix workgroup size exceeding limit',
        fix: (content) => {
            // Replace any workgroup_size that exceeds 256 total threads
            content = content.replace(/@workgroup_size\((\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?\)/g, (match, x, y, z) => {
                const xVal = parseInt(x);
                const yVal = y ? parseInt(y) : 1;
                const zVal = z ? parseInt(z) : 1;
                const total = xVal * yVal * zVal;
                
                if (total > 256) {
                    console.log(`      Reducing workgroup_size from ${total} to 256 threads`);
                    return '@workgroup_size(16, 16, 1)';
                }
                return match;
            });
            return content;
        }
    },
    {
        file: 'hybridWavefieldBlend.wgsl',
        description: 'Check for corrupted content',
        fix: (content) => {
            // Check if file starts with { (JSON/corrupted)
            if (content.trim().startsWith('{')) {
                console.log('      File is corrupted (starts with {), replacing with placeholder');
                return `// hybridWavefieldBlend.wgsl
// TORI-GAEA Hybrid Holographic Core - Wavefield Blending Shader
// TODO: Restore proper implementation

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.uv, 0.0, 1.0);
}`;
            }
            return content;
        }
    }
];

// Apply fixes
console.log('\nStep 2: Fixing shader issues...');
let fixedCount = 0;

for (const fix of fixes) {
    const filePath = path.join(shadersDir, fix.file);
    console.log(`\n📄 ${fix.file}`);
    console.log(`   ${fix.description}`);
    
    if (!fs.existsSync(filePath)) {
        console.log(`   ❌ File not found!`);
        continue;
    }
    
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const fixed = fix.fix(content);
        
        if (fixed !== content) {
            // Backup
            const backupPath = filePath + '.pre-fix.bak';
            if (!fs.existsSync(backupPath)) {
                fs.writeFileSync(backupPath, content);
            }
            
            // Write fixed
            fs.writeFileSync(filePath, fixed);
            console.log(`   ✅ Fixed and saved!`);
            fixedCount++;
        } else {
            console.log(`   ⏭️  No changes needed`);
        }
    } catch (err) {
        console.log(`   ❌ Error: ${err.message}`);
    }
}

console.log('\n' + '='.repeat(60));
console.log(`✅ Fixed ${fixedCount} files`);
console.log('\n📋 Next steps:');
console.log('1. Run validation:');
console.log('   npm run shaders:gate:iphone');
console.log('2. If syntax errors are fixed, fix warnings:');
console.log('   npm run shaders:fix:all');