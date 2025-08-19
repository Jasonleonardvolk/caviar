#!/usr/bin/env node
/**
 * Add package.json scripts for shader management
 * Run this once to update package.json with new shader scripts
 */

import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const packagePath = join(__dirname, '..', '..', 'package.json');

async function updatePackageJson() {
    console.log('📦 Updating package.json with shader scripts...');
    
    try {
        // Read current package.json
        const content = await fs.readFile(packagePath, 'utf8');
        const pkg = JSON.parse(content);
        
        // Ensure scripts object exists
        if (!pkg.scripts) {
            pkg.scripts = {};
        }
        
        // Add new shader scripts
        const newScripts = {
            'shaders:sync': 'node tools/shaders/copy_canonical_to_public.mjs',
            'shaders:validate': 'node tools/shaders/validate_and_report.mjs --dir=frontend --strict',
            'shaders:gate': 'node tools/shaders/validate_and_report.mjs --dir=frontend --limits=tools/shaders/device_limits.iphone15.json --targets=msl,hlsl,spirv --strict',
            'shaders:check-uniforms': 'node tools/shaders/guards/check_uniform_arrays.mjs',
            'shaders:fix-phase2': 'powershell -ExecutionPolicy Bypass -File tools/shaders/Fix-WGSL-Phase2.ps1',
            'shaders:fix-phase3': 'powershell -ExecutionPolicy Bypass -File tools/shaders/Fix-WGSL-Phase3.ps1',
            'shaders:fix-all': 'npm run shaders:fix-phase2 -- -Apply && npm run shaders:fix-phase3 -- -Apply',
            'shaders:full-check': 'npm run shaders:check-uniforms && npm run shaders:sync && npm run shaders:gate',
            'prebuild': 'npm run shaders:sync',
            'predev': 'npm run shaders:sync'
        };
        
        // Merge with existing scripts (don't overwrite existing ones)
        for (const [key, value] of Object.entries(newScripts)) {
            if (!pkg.scripts[key]) {
                pkg.scripts[key] = value;
                console.log(`  ✅ Added script: ${key}`);
            } else {
                console.log(`  ⏭️  Script exists: ${key}`);
            }
        }
        
        // Write back to package.json
        const updatedContent = JSON.stringify(pkg, null, 2) + '\n';
        await fs.writeFile(packagePath, updatedContent, 'utf8');
        
        console.log('\n✅ package.json updated successfully!');
        console.log('\nNew scripts available:');
        console.log('  npm run shaders:sync         - Sync canonical to public');
        console.log('  npm run shaders:validate     - Basic validation');
        console.log('  npm run shaders:gate         - Full validation with device limits');
        console.log('  npm run shaders:check-uniforms - Check for uniform array violations');
        console.log('  npm run shaders:fix-all      - Apply all mechanical fixes');
        console.log('  npm run shaders:full-check   - Complete validation pipeline');
        
    } catch (err) {
        console.error('❌ Failed to update package.json:', err);
        process.exit(1);
    }
}

// Also update .gitignore
async function updateGitignore() {
    const gitignorePath = join(__dirname, '..', '..', '.gitignore');
    
    try {
        let content = '';
        try {
            content = await fs.readFile(gitignorePath, 'utf8');
        } catch (err) {
            console.log('📝 Creating new .gitignore');
        }
        
        const additions = [
            '\n# Shader build outputs (canonical sources are in frontend/lib/webgpu/shaders/)',
            '/frontend/public/hybrid/wgsl/',
            '',
            '# Shader validation reports',
            '/tools/shaders/reports/',
            '!tools/shaders/reports/.gitkeep',
            '',
            '# Shader backups',
            '/frontend/shaders.bak/',
            ''
        ];
        
        // Check if already contains our entries
        if (!content.includes('/frontend/public/hybrid/wgsl/')) {
            content += additions.join('\n');
            await fs.writeFile(gitignorePath, content, 'utf8');
            console.log('\n✅ Updated .gitignore');
        } else {
            console.log('\n⏭️  .gitignore already updated');
        }
        
    } catch (err) {
        console.error('⚠️  Could not update .gitignore:', err.message);
    }
}

// Run both updates
async function main() {
    await updatePackageJson();
    await updateGitignore();
    
    console.log('\n🎉 Setup complete!');
    console.log('\nNext steps:');
    console.log('  1. Run: npm run shaders:full-check');
    console.log('  2. Commit: git add -A && git commit -m "feat: shader validation infrastructure"');
    console.log('  3. Tag: git tag -a shaders-pass-2025-08-08 -m "First green run"');
    console.log('  4. Push: git push && git push --tags');
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
