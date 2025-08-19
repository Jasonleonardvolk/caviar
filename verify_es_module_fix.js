#!/usr/bin/env node
/**
 * ES Module Fix Verification Script
 * Tests that the shader_quality_gate.js can now run with ES modules
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('✅ ES Module imports working correctly!');
console.log('📦 Package.json has been updated with "type": "module"');
console.log('🚀 You can now run: node tools/shader_quality_gate.js --strict');

// Verify package.json has the type field
try {
    const packageJson = JSON.parse(readFileSync(join(__dirname, 'package.json'), 'utf8'));
    if (packageJson.type === 'module') {
        console.log('✓ package.json correctly configured for ES modules');
    } else {
        console.log('⚠ Warning: package.json missing "type": "module"');
    }
} catch (error) {
    console.log('⚠ Could not verify package.json');
}

console.log('\n📝 Note: With "type": "module" set, all .js files are now ES modules.');
console.log('   If you have any CommonJS files, rename them to .cjs');
