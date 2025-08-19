#!/usr/bin/env node
/**
 * Adds bounds checking to all dynamic array accesses in WGSL shaders
 * Inserts clamp_index helper function and wraps all array accesses
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

// The clamp_index helper function to add
const CLAMP_HELPER = `
// Bounds checking helper - prevents out-of-bounds array access
fn clamp_index(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}`;

function processFile(filePath) {
    const fileName = path.basename(filePath);
    console.log(`Processing ${fileName}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    const changes = [];
    
    // Check if clamp_index already exists
    if (!content.includes('fn clamp_index')) {
        // Find a good place to insert the helper
        const structEnd = content.lastIndexOf('}');
        const firstFn = content.indexOf('fn ');
        
        let insertPos = 0;
        if (firstFn > 0) {
            // Insert before first function
            insertPos = firstFn;
            content = content.slice(0, insertPos) + CLAMP_HELPER + '\n\n' + content.slice(insertPos);
        } else {
            // Insert after initial comments
            const lines = content.split('\n');
            for (let i = 0; i < lines.length; i++) {
                if (!lines[i].startsWith('//') && lines[i].trim() !== '') {
                    lines.splice(i, 0, '', CLAMP_HELPER, '');
                    content = lines.join('\n');
                    break;
                }
            }
        }
        modified = true;
        changes.push('Added clamp_index helper function');
    }
    
    // Find array accesses that need bounds checking
    const patterns = [
        // Match array[index] but not if it's already in clamp_index
        /(\b\w+)\[([^[\]]+)\]/g
    ];
    
    // Arrays to skip (built-in types, not actual arrays)
    const skipArrays = new Set(['vec2', 'vec3', 'vec4', 'mat2x2', 'mat3x3', 'mat4x4', 'array']);
    
    // Process each pattern
    for (const pattern of patterns) {
        let match;
        const replacements = [];
        
        while ((match = pattern.exec(content)) !== null) {
            const arrayName = match[1];
            const indexExpr = match[2];
            const fullMatch = match[0];
            
            // Skip if already clamped
            if (content.substring(match.index - 20, match.index).includes('clamp_index')) continue;
            
            // Skip built-in types
            if (skipArrays.has(arrayName)) continue;
            
            // Skip numeric literals
            if (/^\d+u?$/.test(indexExpr.trim())) continue;
            
            // Skip if it's a type declaration
            if (content.substring(match.index - 10, match.index).includes('array<')) continue;
            
            // This needs clamping
            replacements.push({
                start: match.index,
                end: match.index + fullMatch.length,
                original: fullMatch,
                arrayName,
                indexExpr
            });
        }
        
        // Apply replacements in reverse order
        replacements.reverse().forEach(r => {
            // Determine array size
            let sizeExpr = '256u'; // Default fallback
            
            // Look for array declaration
            const sizePattern = new RegExp(`${r.arrayName}\\s*:\\s*array<[^,]+,\\s*(\\w+)>`, 'g');
            const sizeMatch = sizePattern.exec(content);
            if (sizeMatch) {
                sizeExpr = sizeMatch[1];
            } else if (content.includes('const N ')) {
                sizeExpr = 'N';
            } else if (content.includes('WORKGROUP_SIZE')) {
                sizeExpr = 'WORKGROUP_SIZE';
            }
            
            const newExpr = `${r.arrayName}[clamp_index(${r.indexExpr}, ${sizeExpr})]`;
            content = content.substring(0, r.start) + newExpr + content.substring(r.end);
            modified = true;
            changes.push(`Clamped ${r.original}`);
        });
    }
    
    if (modified) {
        // Backup original
        const backupPath = filePath + '.pre-bounds.bak';
        if (!fs.existsSync(backupPath)) {
            fs.copyFileSync(filePath, backupPath);
        }
        
        // Write fixed file
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`  ✅ Fixed: ${changes.length} changes`);
        return true;
    }
    
    console.log(`  ⏭️  No changes needed`);
    return false;
}

// Main
function main() {
    console.log('🔧 Adding bounds checking to shaders...\n');
    
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
        console.log('\n📁 Backups created with .pre-bounds.bak extension');
        console.log('\n🧪 Next: Run validation to verify fixes:');
        console.log('  npm run shaders:gate:iphone');
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { processFile };