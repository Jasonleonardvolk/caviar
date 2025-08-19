#!/usr/bin/env node
/**
 * Fixes const vs let warnings
 * Changes immutable let declarations to const
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// Known immutable variables from the warnings
const CONST_CANDIDATES = {
    'lenticularInterlace.wgsl': ['subpixel_width'],
    'propagation.wgsl': ['view_angle'],
    'velocityField.wgsl': ['momentum', 'value']
};

function processFile(filePath) {
    const fileName = path.basename(filePath);
    
    // Only process files with known const candidates
    if (!CONST_CANDIDATES[fileName]) {
        return false;
    }
    
    console.log(`Processing ${fileName}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    const changes = [];
    const candidates = CONST_CANDIDATES[fileName];
    
    candidates.forEach(varName => {
        // Find let declarations of this variable
        const pattern = new RegExp(`\\blet\\s+${varName}\\s*=`, 'g');
        let match;
        
        while ((match = pattern.exec(content)) !== null) {
            // Check if this variable is ever reassigned
            const declarationLine = content.lastIndexOf('\n', match.index) + 1;
            const endOfFunction = content.indexOf('\n}', match.index);
            const scope = content.substring(match.index, endOfFunction);
            
            // Look for reassignments
            const reassignPattern = new RegExp(`\\b${varName}\\s*=`, 'g');
            const reassignments = [...scope.matchAll(reassignPattern)];
            
            // First match is the declaration, so if there's only 1, no reassignment
            if (reassignments.length <= 1) {
                // Replace let with const
                const replacement = match[0].replace('let', 'const');
                content = content.substring(0, match.index) + replacement + content.substring(match.index + match[0].length);
                modified = true;
                changes.push(`Changed 'let ${varName}' to 'const ${varName}'`);
            }
        }
    });
    
    // Also find other obvious const candidates (simple literals)
    const literalPattern = /\blet\s+(\w+)\s*=\s*([\d.]+|true|false)\s*;/g;
    let match;
    
    while ((match = literalPattern.exec(content)) !== null) {
        const varName = match[1];
        const value = match[2];
        
        // Check if reassigned
        const scope = content.substring(match.index, Math.min(match.index + 1000, content.length));
        if (!scope.includes(`${varName} =`) || scope.indexOf(`${varName} =`) === 0) {
            const replacement = match[0].replace('let', 'const');
            content = content.substring(0, match.index) + replacement + content.substring(match.index + match[0].length);
            modified = true;
            changes.push(`Changed 'let ${varName} = ${value}' to const`);
        }
    }
    
    if (modified) {
        // Backup original
        const backupPath = filePath + '.pre-const.bak';
        if (!fs.existsSync(backupPath)) {
            fs.copyFileSync(filePath, backupPath);
        }
        
        // Write fixed file
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`  ✅ Fixed: ${changes.length} changes`);
        changes.forEach(c => console.log(`    - ${c}`));
        return true;
    }
    
    console.log(`  ⏭️  No const/let issues`);
    return false;
}

// Main
function main() {
    console.log('🔧 Fixing const vs let warnings...\n');
    
    if (!fs.existsSync(shadersDir)) {
        console.error('Shader directory not found:', shadersDir);
        process.exit(1);
    }
    
    const files = fs.readdirSync(shadersDir)
        .filter(f => f.endsWith('.wgsl'))
        .map(f => path.join(shadersDir, f));
    
    console.log(`Found ${files.length} shader files\n`);
    
    let fixedCount = 0;
    for (const file of files) {
        if (processFile(file)) {
            fixedCount++;
        }
    }
    
    console.log(`\n✅ Complete! Modified ${fixedCount} files.`);
    
    if (fixedCount > 0) {
        console.log('\n📁 Backups created with .pre-const.bak extension');
        console.log('\n🧪 Next: Run validation to verify fixes:');
        console.log('  npm run shaders:gate:iphone');
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { processFile };