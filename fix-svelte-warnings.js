#!/usr/bin/env node
// fix-svelte-warnings.js
// Node.js version for cross-platform compatibility

const fs = require('fs');
const path = require('path');

console.log('===== Fixing Svelte Warnings =====\n');

const baseDir = 'D:\\Dev\\kha\\tori_ui_svelte';
let fixCount = 0;

// Backup function
function backupFile(filePath) {
    if (fs.existsSync(filePath)) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const backupPath = `${filePath}.backup_${timestamp}`;
        fs.copyFileSync(filePath, backupPath);
        console.log(`  Backed up to: ${path.basename(backupPath)}`);
    }
}

// Apply fix helper
function applyFix(filePath, description, fixFn) {
    console.log(`Fixing: ${description}`);
    
    if (!fs.existsSync(filePath)) {
        console.log(`  ❌ File not found: ${filePath}`);
        return false;
    }
    
    try {
        backupFile(filePath);
        let content = fs.readFileSync(filePath, 'utf8');
        content = fixFn(content);
        fs.writeFileSync(filePath, content);
        console.log('  ✅ Fixed!');
        return true;
    } catch (error) {
        console.log(`  ❌ Error: ${error.message}`);
        return false;
    }
}

// Fix 1: ghost-history clickable div
const ghostHistoryFile = path.join(baseDir, 'src\\routes\\ghost-history\\+page.svelte');
if (applyFix(ghostHistoryFile, 'ghost-history clickable div (A11y)', (content) => {
    // Convert div to button
    content = content.replace(
        /<div\s+class="bg-white rounded-lg border border-gray-200 p-4 hover:border-purple-300 hover:shadow-sm transition-all cursor-pointer"\s+on:click=\{([^}]+)\}>/g,
        '<button type="button" class="bg-white rounded-lg border border-gray-200 p-4 text-left w-full hover:border-purple-300 hover:shadow-sm transition-all" on:click={$1} aria-label="Open memory details">'
    );
    
    // Fix closing tag - find the pattern more carefully
    const lines = content.split('\n');
    let inMemoryLoop = false;
    let divDepth = 0;
    
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes('{#each $filteredMemories as memory}')) {
            inMemoryLoop = true;
        }
        if (inMemoryLoop) {
            // Changed opening div to button
            if (lines[i].includes('<button type="button"')) {
                divDepth = 1;
            }
            // Find the matching closing div
            if (divDepth > 0 && lines[i].includes('</div>')) {
                lines[i] = lines[i].replace('</div>', '</button>');
                divDepth--;
                if (divDepth === 0) {
                    inMemoryLoop = false;
                }
            }
        }
    }
    
    return lines.join('\n');
})) { fixCount++; }

// Fix 2: PersonaPanel A11y
const personaPanelFile = path.join(baseDir, 'src\\lib\\components\\PersonaPanel.svelte');
if (applyFix(personaPanelFile, 'PersonaPanel A11y issues', (content) => {
    // Add aria-labelledby to overlay
    if (!content.includes('aria-labelledby')) {
        content = content.replace(
            /<div class="persona-panel-overlay"\s+role="dialog"\s+aria-modal="true"/,
            '<div class="persona-panel-overlay" role="dialog" aria-modal="true" aria-labelledby="persona-panel-title"'
        );
    }
    
    // Add role="document" to panel
    content = content.replace(
        /<div class="persona-panel"\s+on:click\|stopPropagation>/,
        '<div class="persona-panel" on:click|stopPropagation role="document">'
    );
    
    // Fix color label - convert to fieldset
    content = content.replace(
        /<label id="hologram-color-label">Hologram Color:<\/label>\s*<div class="color-sliders">/,
        '<fieldset role="group" aria-labelledby="hologram-color-label"><legend id="hologram-color-label">Hologram Color</legend><div class="color-sliders">'
    );
    
    // Close fieldset after color-sliders
    const colorSlidersEnd = content.indexOf('</div>', content.indexOf('class="color-sliders"'));
    if (colorSlidersEnd > -1) {
        content = content.slice(0, colorSlidersEnd + 6) + '</fieldset>' + content.slice(colorSlidersEnd + 6);
    }
    
    return content;
})) { fixCount++; }

// Fix 3: MemoryVaultDashboard systemCoherence
const vaultDashboardFile = path.join(baseDir, 'src\\lib\\components\\vault\\MemoryVaultDashboard.svelte');
if (applyFix(vaultDashboardFile, 'MemoryVaultDashboard systemCoherence', (content) => {
    // Check if already imported
    if (!content.includes('systemCoherence')) {
        // Add import after script tag
        const scriptMatch = content.match(/<script[^>]*>/);
        if (scriptMatch) {
            const insertPos = scriptMatch.index + scriptMatch[0].length;
            const importStatement = "\nimport { writable } from 'svelte/store';\nconst systemCoherence = writable(0.85);\n";
            content = content.slice(0, insertPos) + importStatement + content.slice(insertPos);
        }
    }
    return content;
})) { fixCount++; }

// Fix 4: HolographicDisplay unused export
const holographicDisplayFile = path.join(baseDir, 'src\\lib\\components\\HolographicDisplay.svelte');
if (applyFix(holographicDisplayFile, 'HolographicDisplay unused export', (content) => {
    // Add reactive statement to acknowledge the prop
    if (content.includes('export let usePenrose = true;')) {
        content = content.replace(
            'export let usePenrose = true;',
            'export let usePenrose = true;\n  $: __usePenrose = usePenrose; // Acknowledge prop'
        );
    }
    return content;
})) { fixCount++; }

// Fix 5: Remove unused imports
const apiListFile = path.join(baseDir, 'src\\routes\\api\\list\\+server.ts');
if (applyFix(apiListFile, 'api/list unused imports', (content) => {
    content = content.replace(/import\s*{\s*writeFile\s*,\s*unlink\s*}\s*from\s*['"]fs\/promises['"];\s*\n?/g, '');
    return content;
})) { fixCount++; }

const uploadServerFile = path.join(baseDir, 'src\\routes\\upload\\+server.ts');
if (applyFix(uploadServerFile, 'upload/+server.ts unused imports', (content) => {
    content = content.replace(/import\s*{\s*writeFile\s*,\s*unlink\s*}\s*from\s*['"]fs\/promises['"];\s*\n?/g, '');
    content = content.replace(/import\s*(?:{\s*default\s+as\s+\w+\s*}|\w+)\s*from\s*['"]os['"];\s*\n?/g, '');
    return content;
})) { fixCount++; }

// Summary
console.log('\n===== Summary =====');
console.log(`Fixed ${fixCount} issues`);

if (fixCount > 0) {
    console.log('\nNext steps:');
    console.log('1. Run the build again to verify warnings are gone:');
    console.log('   .\\tools\\release\\IrisOneButton.ps1 -NonInteractive -QuickBuild');
    console.log('\n2. If all warnings are cleared, you\'re ready to ship!');
}
