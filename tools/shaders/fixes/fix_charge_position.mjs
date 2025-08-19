#!/usr/bin/env node
/**
 * Fix all remaining charge.position to charge.position.xyz in topologicalOverlay.wgsl
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const filePath = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders', 'topologicalOverlay.wgsl');

console.log('🔧 Fixing remaining charge.position references...\n');

let content = fs.readFileSync(filePath, 'utf8');

// Replace all remaining charge.position with charge.position.xyz and other.position with other.position.xyz
// But only in distance calculations
const fixes = [
    {
        // In calculateEnergyDensity
        old: /let distance = length\(worldPos - charge\.position\)/g,
        new: 'let distance = length(worldPos - charge.position.xyz)'
    },
    {
        // In calculateCoherence
        old: /let distance = length\(worldPos - charge\.position\)/g,
        new: 'let distance = length(worldPos - charge.position.xyz)'
    }
];

let changeCount = 0;
for (const fix of fixes) {
    const matches = content.match(fix.old);
    if (matches) {
        content = content.replace(fix.old, fix.new);
        changeCount += matches.length;
    }
}

fs.writeFileSync(filePath, content);
console.log(`✅ Fixed ${changeCount} remaining charge.position references`);
console.log('\nRun: npm run shaders:sync ; npm run shaders:gate:iphone');
