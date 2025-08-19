// Quick fix for common TypeScript errors
const fs = require('fs');
const path = require('path');

// Fix 1: Generate SvelteKit types
console.log('Generating SvelteKit types...');
const { execSync } = require('child_process');
try {
    execSync('npm run sync', {
        cwd: 'D:\\Dev\\kha\\tori_ui_svelte',
        stdio: 'inherit'
    });
    console.log('✓ SvelteKit types generated');
} catch (e) {
    console.log('⚠ Could not generate SvelteKit types');
}

// Fix 2: Check for missing type packages
console.log('\nChecking for missing @types packages...');
const packagesToCheck = [
    '@types/webxr',
    '@types/three', 
    '@webgpu/types'
];

packagesToCheck.forEach(pkg => {
    try {
        require.resolve(pkg);
        console.log(`✓ ${pkg} is installed`);
    } catch {
        console.log(`✗ ${pkg} is NOT installed - run: npm install -D ${pkg}`);
    }
});

console.log('\nNext: Run ANALYZE_ERRORS.bat to see specific issues');
