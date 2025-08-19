#!/usr/bin/env node
/**
 * Auto-fix array bounds in wavefieldEncoder.wgsl and wavefieldEncoder_optimized.wgsl
 * These two files have 40 warnings between them!
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

function fixWavefieldEncoder(fileName) {
    const filePath = path.join(shadersDir, fileName);
    console.log(`\n🔧 Fixing ${fileName}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    let changeCount = 0;
    
    // Pattern replacements
    const replacements = [
        // Shared arrays (workgroup memory) - use constant 256u
        {
            pattern: /shared_spatial_freqs\[([^\]]+)\]/g,
            replacement: 'shared_spatial_freqs[clamp_index_dyn($1, 256u)]'
        },
        {
            pattern: /shared_phases\[([^\]]+)\]/g,
            replacement: 'shared_phases[clamp_index_dyn($1, 256u)]'
        },
        {
            pattern: /shared_amplitudes\[([^\]]+)\]/g,
            replacement: 'shared_amplitudes[clamp_index_dyn($1, 256u)]'
        },
        // Storage arrays - use arrayLength
        {
            pattern: /(?<!shared_)spatial_freqs\[([^\]]+)\]/g,
            replacement: 'spatial_freqs[clamp_index_dyn($1, arrayLength(&spatial_freqs))]'
        },
        {
            pattern: /(?<!shared_)phases\[([^\]]+)\]/g,
            replacement: 'phases[clamp_index_dyn($1, arrayLength(&phases))]'
        },
        {
            pattern: /(?<!shared_)amplitudes\[([^\]]+)\]/g,
            replacement: 'amplitudes[clamp_index_dyn($1, arrayLength(&amplitudes))]'
        },
        // Fixed-size array
        {
            pattern: /dispersion_factors\[([^\]]+)\]/g,
            replacement: 'dispersion_factors[clamp_index_dyn($1, 3u)]'
        }
    ];
    
    for (const {pattern, replacement} of replacements) {
        const matches = content.match(pattern);
        if (matches) {
            // Check if already clamped
            const unclampedMatches = matches.filter(m => !m.includes('clamp_index_dyn'));
            if (unclampedMatches.length > 0) {
                content = content.replace(pattern, (match, p1) => {
                    if (match.includes('clamp_index_dyn')) {
                        return match; // Already clamped
                    }
                    changeCount++;
                    return replacement.replace('$1', p1);
                });
            }
        }
    }
    
    fs.writeFileSync(filePath, content);
    console.log(`  ✅ Fixed ${changeCount} array accesses`);
    return changeCount;
}

console.log('🎯 Fixing wavefieldEncoder files (40 warnings total)');

const total = 
    fixWavefieldEncoder('wavefieldEncoder.wgsl') +
    fixWavefieldEncoder('wavefieldEncoder_optimized.wgsl');

console.log(`\n✅ Total: Fixed ${total} array accesses`);
console.log('\nRun: npm run shaders:sync ; npm run shaders:gate:iphone');
