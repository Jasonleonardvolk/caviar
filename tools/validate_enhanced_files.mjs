#!/usr/bin/env node
/**
 * validate_enhanced_files.mjs
 * 
 * Validates that all "enhanced" files are documented in ENHANCED_FILES_MANIFEST.md
 * Helps prevent accidental creation of duplicate enhanced files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '../..');

async function findEnhancedFiles() {
    const patterns = [
        '**/*enhanced*.*',
        '**/*Enhanced*.*'
    ];
    
    const ignorePatterns = [
        'node_modules/**',
        '_TRASH_2025/**',
        '.git/**',
        'ENHANCED_FILES_MANIFEST.md',
        'validate_enhanced_files.mjs'
    ];
    
    const files = new Set();
    
    for (const pattern of patterns) {
        const matches = await glob(pattern, {
            cwd: projectRoot,
            ignore: ignorePatterns
        });
        matches.forEach(f => files.add(f));
    }
    
    return Array.from(files).sort();
}

function readManifest() {
    const manifestPath = path.join(projectRoot, 'ENHANCED_FILES_MANIFEST.md');
    if (!fs.existsSync(manifestPath)) {
        console.error('❌ ENHANCED_FILES_MANIFEST.md not found!');
        process.exit(1);
    }
    
    const content = fs.readFileSync(manifestPath, 'utf-8');
    const documented = new Set();
    
    // Extract file paths from markdown table
    const lines = content.split('\\n');
    for (const line of lines) {
        // Match file paths in backticks
        const matches = line.match(/`([^`]+enhanced[^`]+)`/gi);
        if (matches) {
            matches.forEach(m => {
                const filepath = m.slice(1, -1); // Remove backticks
                documented.add(filepath.replace(/\\\\/g, '/'));
            });
        }
    }
    
    return documented;
}

async function validate() {
    console.log('🔍 Validating enhanced files...\\n');
    
    const foundFiles = await findEnhancedFiles();
    const documentedFiles = readManifest();
    
    const undocumented = [];
    const documented = [];
    
    for (const file of foundFiles) {
        const normalizedPath = file.replace(/\\\\/g, '/');
        let isDocumented = false;
        
        // Check if file matches any documented path
        for (const docPath of documentedFiles) {
            if (normalizedPath.endsWith(docPath) || docPath.endsWith(normalizedPath)) {
                isDocumented = true;
                break;
            }
        }
        
        if (isDocumented) {
            documented.push(file);
        } else {
            undocumented.push(file);
        }
    }
    
    // Report results
    if (documented.length > 0) {
        console.log('✅ Documented enhanced files:');
        documented.forEach(f => console.log(`   ${f}`));
        console.log('');
    }
    
    if (undocumented.length > 0) {
        console.log('⚠️  UNDOCUMENTED enhanced files (may be duplicates):');
        undocumented.forEach(f => console.log(`   ${f}`));
        console.log('\\n📋 Action required:');
        console.log('   1. If legitimate: Add to ENHANCED_FILES_MANIFEST.md');
        console.log('   2. If duplicate: Merge into canonical version and delete');
        console.log('   3. If experimental: Move to experimental/ folder');
        return false;
    }
    
    console.log('✨ All enhanced files are properly documented!');
    return true;
}

// Run validation
validate().then(success => {
    process.exit(success ? 0 : 1);
}).catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
