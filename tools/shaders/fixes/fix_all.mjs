#!/usr/bin/env node
/**
 * Master shader fixer - runs all fixes in the correct order
 * 1. vec3 alignment (structural changes)
 * 2. bounds checking (safety)
 * 3. const/let (optimization)
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const fixes = [
    {
        name: 'Vec3 Storage Alignment',
        script: 'fix_vec3_alignment.mjs',
        description: 'Converts vec3 to vec4 in storage buffers for proper alignment'
    },
    {
        name: 'Bounds Checking',
        script: 'fix_bounds_checking.mjs',
        description: 'Adds clamp_index() to prevent array out-of-bounds access'
    },
    {
        name: 'Const vs Let',
        script: 'fix_const_let.mjs',
        description: 'Changes immutable let declarations to const'
    }
];

async function runFix(fix) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🔧 Running: ${fix.name}`);
    console.log(`   ${fix.description}`);
    console.log('='.repeat(60) + '\n');
    
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, fix.script);
        const child = spawn('node', [scriptPath], {
            stdio: 'inherit',
            shell: process.platform === 'win32'
        });
        
        child.on('close', (code) => {
            if (code === 0) {
                console.log(`✅ ${fix.name} completed successfully`);
                resolve();
            } else {
                console.error(`❌ ${fix.name} failed with code ${code}`);
                reject(new Error(`Fix failed: ${fix.name}`));
            }
        });
        
        child.on('error', (err) => {
            console.error(`❌ Failed to run ${fix.name}:`, err);
            reject(err);
        });
    });
}

async function main() {
    console.log('🚀 SHADER AUTO-FIXER');
    console.log('='.repeat(60));
    console.log('This will fix all shader validation warnings:');
    console.log('  1. Storage buffer vec3 alignment issues');
    console.log('  2. Dynamic array bounds checking');
    console.log('  3. Const vs let optimizations');
    console.log('='.repeat(60));
    
    const startTime = Date.now();
    
    try {
        for (const fix of fixes) {
            await runFix(fix);
        }
        
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        console.log('\n' + '='.repeat(60));
        console.log('🎉 ALL FIXES COMPLETE!');
        console.log(`   Time: ${elapsed}s`);
        console.log('='.repeat(60));
        
        console.log('\n📁 Backup files created:');
        console.log('  - *.pre-vec4.bak     (before vec3→vec4 conversion)');
        console.log('  - *.pre-bounds.bak   (before bounds checking)');
        console.log('  - *.pre-const.bak    (before const/let fixes)');
        
        console.log('\n🧪 Next steps:');
        console.log('  1. Sync to public:     npm run shaders:sync');
        console.log('  2. Validate fixes:     npm run shaders:gate:iphone');
        console.log('  3. Check all profiles: npm run shaders:validate:all');
        
        console.log('\n💡 To restore originals:');
        console.log('  powershell: Get-ChildItem -Filter "*.bak" | ForEach { Copy-Item $_.FullName ($_.FullName -replace "\\.bak$", "") }');
        
    } catch (err) {
        console.error('\n❌ Fix process failed:', err.message);
        console.error('\nCheck the output above for details.');
        process.exit(1);
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}