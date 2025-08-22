#!/usr/bin/env node
// Quick test to verify manifest validator is working

import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
const manifestPath = path.join(projectRoot, 'assets/3d/luxury/ASSET_MANIFEST.json');
const validatorPath = path.join(projectRoot, 'tools/assets/validate-manifest.mjs');

console.log('=====================================');
console.log('   Testing Manifest Validator');
console.log('=====================================');
console.log('');

try {
    // Run validator
    const output = execSync(`node "${validatorPath}" "${manifestPath}" --maxTris=100000`, {
        encoding: 'utf8',
        stdio: 'pipe'
    });
    
    console.log(output);
    console.log('[OK] Validator is working!');
    console.log('Note: Warnings about missing model files are expected');
    console.log('      until the 3D models are downloaded.');
    
} catch (error) {
    if (error.stdout) {
        console.log(error.stdout);
    }
    
    if (error.stderr) {
        console.error(error.stderr);
    }
    
    // Exit code 0 = success, 1 = validation errors
    if (error.status === 1) {
        console.log('[!] Validation completed with errors (expected if models not downloaded)');
    } else {
        console.error('[X] Validator failed to run');
        console.error(error.message);
    }
}

console.log('');
console.log('=====================================');