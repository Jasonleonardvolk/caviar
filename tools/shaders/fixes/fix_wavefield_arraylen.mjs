#!/usr/bin/env node
/**
 * Fix the broken wavefieldEncoder files
 * The arrays are part of a struct, not standalone
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

function fixFile(fileName) {
    const filePath = path.join(shadersDir, fileName);
    console.log(`\n🔧 Fixing ${fileName}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // These arrays are fixed-size arrays in the struct (MAX_OSCILLATORS = 32)
    // Fix the incorrect arrayLength(&spatial_freqs) references
    content = content.replace(/arrayLength\(&spatial_freqs\)/g, 'MAX_OSCILLATORS');
    content = content.replace(/arrayLength\(&phases\)/g, 'MAX_OSCILLATORS');
    content = content.replace(/arrayLength\(&amplitudes\)/g, 'MAX_OSCILLATORS');
    
    fs.writeFileSync(filePath, content);
    console.log(`  ✅ Fixed array length references`);
}

console.log('🔧 Fixing wavefieldEncoder arrayLength issues...');

fixFile('wavefieldEncoder.wgsl');
fixFile('wavefieldEncoder_optimized.wgsl');

console.log('\n✅ Fixed! Run: npm run shaders:sync ; npm run shaders:gate:iphone');
