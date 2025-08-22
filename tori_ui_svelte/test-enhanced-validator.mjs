#!/usr/bin/env node
// Test the enhanced texture validator features

import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
const manifestPath = path.join(projectRoot, 'assets/3d/luxury/ASSET_MANIFEST.json');
const validatorPath = path.join(projectRoot, 'tools/assets/validate-manifest.mjs');

console.log('=====================================');
console.log('  Testing Enhanced Validator v3.0');
console.log('=====================================');
console.log('');
console.log('Features being tested:');
console.log('  - Texture format allow-list (PNG, JPG, WebP only)');
console.log('  - Alpha channel detection (PNG/WebP)');
console.log('  - Alpha texture limits (2048px, 8MB)');
console.log('  - Opaque texture limits (4096px, 16MB)');
console.log('  - Normal map limits (2048px max)');
console.log('  - Total asset set size (50MB)');
console.log('');

try {
    // Run validator with all parameters
    const command = `node "${validatorPath}" "${manifestPath}" --maxTris=100000 --maxTexDim=4096 --maxTexBytes=16777216 --maxAlphaDim=2048 --maxAlphaBytes=8388608 --maxSetBytes=52428800`;
    
    console.log('Running validation with strict limits...');
    console.log('Command:', command.replace(projectRoot, '.'));
    console.log('');
    
    const output = execSync(command, {
        encoding: 'utf8',
        stdio: 'pipe'
    });
    
    console.log(output);
    console.log('[OK] Validation passed with enhanced texture policies!');
    console.log('');
    console.log('What this means:');
    console.log('  - All textures are in allowed formats');
    console.log('  - Alpha textures are within 2K/8MB limits');
    console.log('  - Normal maps are within 2K limit');
    console.log('  - Total size is under 50MB');
    
} catch (error) {
    if (error.stdout) {
        console.log(error.stdout);
    }
    
    if (error.stderr) {
        console.error(error.stderr);
    }
    
    if (error.status === 1) {
        console.log('[!] Validation completed with errors');
        console.log('');
        console.log('Common issues:');
        console.log('  - Texture format not in allow-list (use PNG/JPG/WebP)');
        console.log('  - Alpha texture too large (max 2048px, 8MB)');
        console.log('  - Normal map too large (max 2048px)');
        console.log('  - Total size exceeds 50MB');
    } else {
        console.error('[X] Validator failed to run');
        console.error(error.message);
    }
}

console.log('');
console.log('=====================================');