#!/usr/bin/env node
/**
 * Fixes vec3 storage buffer alignment issues
 * Converts vec3<f32> to vec4<f32> in storage buffer structs
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const shadersDir = path.join(__dirname, '..', '..', '..', 'frontend', 'lib', 'webgpu', 'shaders');

function processFile(filePath) {
    const fileName = path.basename(filePath);
    console.log(`Processing ${fileName}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    const changes = [];
    
    // Find all struct definitions
    const structPattern = /struct\s+(\w+)\s*\{([^}]+)\}/g;
    let match;
    const structs = [];
    
    while ((match = structPattern.exec(content)) !== null) {
        structs.push({
            name: match[1],
            body: match[2],
            start: match.index,
            end: match.index + match[0].length,
            fullMatch: match[0]
        });
    }
    
    // Check which structs are used in storage buffers
    const storageStructs = new Set();
    structs.forEach(s => {
        const storagePattern = new RegExp(`var<storage[^>]*>\\s+\\w+\\s*:\\s*(array<)?${s.name}`, 'g');
        if (storagePattern.test(content)) {
            storageStructs.add(s.name);
        }
    });
    
    // Process structs that need fixing
    structs.reverse(); // Process from bottom to top
    
    structs.forEach(struct => {
        if (!storageStructs.has(struct.name)) return;
        
        // Check if struct has vec3 fields
        if (!struct.body.includes('vec3<f32>')) return;
        
        console.log(`  Found vec3 in storage struct ${struct.name}`);
        
        // Parse fields
        const lines = struct.body.split('\n').map(line => line.trim()).filter(line => line);
        const newLines = [];
        const vec3Fields = [];
        
        lines.forEach(line => {
            if (line.includes('vec3<f32>')) {
                // Extract field name
                const fieldMatch = line.match(/(\w+)\s*:\s*vec3<f32>/);
                if (fieldMatch) {
                    const fieldName = fieldMatch[1];
                    vec3Fields.push(fieldName);
                    
                    // Convert to vec4
                    const newLine = line.replace('vec3<f32>', 'vec4<f32>');
                    newLines.push(newLine + '  // .xyz for position, .w unused (padding)');
                    changes.push(`Converted ${fieldName} from vec3 to vec4`);
                } else {
                    newLines.push(line);
                }
            } else {
                newLines.push(line);
            }
        });
        
        if (vec3Fields.length > 0) {
            // Rebuild struct
            const newBody = '\n    ' + newLines.join('\n    ') + '\n';
            const newStruct = `struct ${struct.name} {${newBody}}`;
            
            content = content.substring(0, struct.start) + newStruct + content.substring(struct.end);
            modified = true;
            
            // Now update all usages to use .xyz
            vec3Fields.forEach(field => {
                // Find usages like particle.pos that aren't already .xyz
                const usagePattern = new RegExp(`(\\w+)\\.${field}(?!\\.xyz)(?![\\w])`, 'g');
                let usageMatch;
                const replacements = [];
                
                while ((usageMatch = usagePattern.exec(content)) !== null) {
                    // Check context - if it's being assigned a vec3, we need .xyz
                    const lineStart = content.lastIndexOf('\n', usageMatch.index) + 1;
                    const lineEnd = content.indexOf('\n', usageMatch.index);
                    const line = content.substring(lineStart, lineEnd);
                    
                    // Check if this is in a vec3 context
                    if (line.includes('vec3<f32>') || line.includes('vec3(')) {
                        replacements.push({
                            start: usageMatch.index,
                            end: usageMatch.index + usageMatch[0].length,
                            replacement: `${usageMatch[1]}.${field}.xyz`
                        });
                    }
                }
                
                // Apply replacements in reverse
                replacements.reverse().forEach(r => {
                    content = content.substring(0, r.start) + r.replacement + content.substring(r.end);
                    changes.push(`Updated usage to ${r.replacement}`);
                });
            });
        }
    });
    
    if (modified) {
        // Backup original
        const backupPath = filePath + '.pre-vec4.bak';
        if (!fs.existsSync(backupPath)) {
            fs.copyFileSync(filePath, backupPath);
        }
        
        // Write fixed file
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`  ✅ Fixed: ${changes.length} changes`);
        return true;
    }
    
    console.log(`  ⏭️  No vec3 storage issues`);
    return false;
}

// Main
function main() {
    console.log('🔧 Fixing vec3 storage buffer alignment...\n');
    
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
        console.log('\n📁 Backups created with .pre-vec4.bak extension');
        console.log('\n🧪 Next: Run validation to verify fixes:');
        console.log('  npm run shaders:gate:iphone');
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export { processFile };