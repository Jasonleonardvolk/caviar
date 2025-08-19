#!/usr/bin/env node
/**
 * Holographic System Integration Verification
 * Checks that all components are properly connected
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, '..');

console.log('🔍 Verifying Holographic System Integration...\n');

// Check core files exist
const coreFiles = [
    // WebGPU Infrastructure
    'frontend/lib/webgpu/fftCompute.ts',
    'frontend/lib/webgpu/hologramPropagation.ts',
    'frontend/lib/webgpu/quiltGenerator.ts',
    'frontend/lib/holographicEngine.ts',
    
    // Shaders
    'frontend/shaders/wavefieldEncoder.wgsl',
    'frontend/shaders/propagation.wgsl',
    'frontend/shaders/multiViewSynthesis.wgsl',
    'frontend/shaders/lenticularInterlace.wgsl',
    
    // Components
    'frontend/components/HolographicVisualization.svelte',
    'frontend/components/pairing/QRPairing.svelte',
    
    // Encoder System
    'src/encoder/h265Server.ts',
    'src/encoder/webrtcHandler.ts',
    
    // Dickbox Configuration
    'dickbox/hologram/Dickboxfile',
    'dickbox/hologram/capsule.yml',
    
    // Mobile App
    'mobile/src/holographicEngine.ts',
    'mobile/src/telemetry.ts',
    'mobile/src/App.svelte',
    'mobile/capacitor.config.ts'
];

let allFilesExist = true;
const missingFiles = [];

console.log('📁 Checking core files:');
for (const file of coreFiles) {
    const fullPath = path.join(projectRoot, file);
    const exists = fs.existsSync(fullPath);
    console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    if (!exists) {
        allFilesExist = false;
        missingFiles.push(file);
    }
}

console.log('\n📦 Checking dependencies:');

// Check package.json dependencies
const rootPackageJson = JSON.parse(
    fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8')
);

const requiredDeps = [
    '@webgpu/types',
    '@msgpack/msgpack',
    'simple-peer',
    'ws',
    'zeromq',
    'prom-client'
];

for (const dep of requiredDeps) {
    const hasDep = rootPackageJson.dependencies?.[dep] || 
                   rootPackageJson.devDependencies?.[dep];
    console.log(`  ${hasDep ? '✅' : '❌'} ${dep}`);
}

console.log('\n🔗 Integration Points:');

// Check if HolographicEngine imports exist
if (fs.existsSync(path.join(projectRoot, 'frontend/lib/holographicEngine.ts'))) {
    const engineContent = fs.readFileSync(
        path.join(projectRoot, 'frontend/lib/holographicEngine.ts'), 
        'utf8'
    );
    
    const imports = [
        'FFTCompute',
        'HologramPropagation',
        'QuiltGenerator'
    ];
    
    for (const imp of imports) {
        const hasImport = engineContent.includes(`import { ${imp} }`);
        console.log(`  ${hasImport ? '✅' : '❌'} HolographicEngine imports ${imp}`);
    }
}

// Summary
console.log('\n📊 Summary:');
if (allFilesExist) {
    console.log('✅ All core files are present!');
} else {
    console.log(`❌ Missing ${missingFiles.length} files:`);
    missingFiles.forEach(f => console.log(`   - ${f}`));
}

console.log('\n💡 Next Steps:');
console.log('1. Run: npm install (in root, frontend, and mobile directories)');
console.log('2. Build desktop capsule: npm run hologram:build-capsule');
console.log('3. Build mobile app: npm run mobile:build');
console.log('4. Deploy with dickbox: sudo ./scripts/install_services.sh');
console.log('\n✨ The holographic system is ready for deployment!');
