#!/usr/bin/env node
/**
 * Fixes the 6 shader syntax errors that are preventing compilation
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

const fixes = [
    {
        file: 'topologicalOverlay.wgsl',
        name: 'Storage buffer write permission',
        fix: (content) => {
            // Change from read to read_write
            return content.replace(
                '@group(0) @binding(1) var<storage, read> charges: array<ChargeData>;',
                '@group(0) @binding(1) var<storage, read_write> charges: array<ChargeData>;'
            );
        }
    },
    {
        file: 'lightFieldComposer.wgsl',
        name: 'textureLoad missing mip level',
        fix: (content) => {
            // Add mip level (0) as 4th argument
            return content.replace(
                /textureLoad\(([^,]+),\s*([^,]+),\s*([^)]+)\)/g,
                'textureLoad($1, $2, $3, 0)'
            );
        }
    },
    {
        file: 'multiDepthWaveSynth.wgsl',
        name: 'Invalid struct syntax',
        fix: (content) => {
            // Fix inline struct declaration
            const pattern = /var<uniform>\s+(\w+):\s*struct\s*{([^}]+)}/g;
            return content.replace(pattern, (match, varName, fields) => {
                // Generate struct name
                const structName = varName.charAt(0).toUpperCase() + varName.slice(1) + 'Data';
                // Create proper struct declaration
                return `struct ${structName} {${fields}}\n\n@group(0) @binding(2) var<uniform> ${varName}: ${structName}`;
            });
        }
    },
    {
        file: 'phaseOcclusion.wgsl',
        name: 'Invalid struct syntax',
        fix: (content) => {
            // Same fix as multiDepthWaveSynth
            const pattern = /var<uniform>\s+(\w+):\s*struct\s*{([^}]+)}/g;
            return content.replace(pattern, (match, varName, fields) => {
                const structName = varName.charAt(0).toUpperCase() + varName.slice(1) + 'Data';
                return `struct ${structName} {${fields}}\n\n@group(0) @binding(3) var<uniform> ${varName}: ${structName}`;
            });
        }
    },
    {
        file: 'multiViewSynthesis.wgsl',
        name: 'Workgroup size exceeds limit',
        fix: (content) => {
            // Change from 512 to 256 (16x16x1)
            return content
                .replace('@workgroup_size(512)', '@workgroup_size(256)')
                .replace('@workgroup_size(32, 16, 1)', '@workgroup_size(16, 16, 1)')
                .replace('@workgroup_size(16, 32, 1)', '@workgroup_size(16, 16, 1)');
        }
    },
    {
        file: 'hybridWavefieldBlend.wgsl',
        name: 'File starts with { (corrupted)',
        fix: (content) => {
            // Check if file starts with { (JSON or corrupted)
            if (content.trim().startsWith('{')) {
                console.log('    File appears to be JSON or corrupted. Creating minimal valid shader...');
                // Return a minimal valid shader
                return `// hybridWavefieldBlend.wgsl
// Placeholder shader - original was corrupted
// TODO: Restore proper implementation

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    // Simple fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Placeholder - just output UV as color
    return vec4<f32>(input.uv, 0.0, 1.0);
}`;
            }
            return content;
        }
    }
];

async function main() {
    console.log('🔧 Fixing 6 shader syntax errors...\n');
    
    let fixedCount = 0;
    let failedCount = 0;
    
    for (const fix of fixes) {
        const filePath = path.join(shadersDir, fix.file);
        console.log(`📄 ${fix.file}`);
        console.log(`   Issue: ${fix.name}`);
        
        if (!fs.existsSync(filePath)) {
            console.log(`   ❌ File not found!`);
            failedCount++;
            continue;
        }
        
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const fixed = fix.fix(content);
            
            if (fixed !== content) {
                // Backup original
                const backupPath = filePath + '.pre-syntax-fix.bak';
                if (!fs.existsSync(backupPath)) {
                    fs.writeFileSync(backupPath, content);
                }
                
                // Write fixed
                fs.writeFileSync(filePath, fixed);
                console.log(`   ✅ Fixed!`);
                fixedCount++;
            } else {
                console.log(`   ⏭️  No changes needed`);
            }
        } catch (err) {
            console.log(`   ❌ Error: ${err.message}`);
            failedCount++;
        }
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`✅ Fixed: ${fixedCount} files`);
    if (failedCount > 0) {
        console.log(`❌ Failed: ${failedCount} files`);
    }
    
    console.log('\n📋 Next steps:');
    console.log('1. Run validation to verify syntax fixes:');
    console.log('   npm run shaders:gate:iphone');
    console.log('2. If syntax errors are fixed, run warning fixes:');
    console.log('   npm run shaders:fix:all');
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}