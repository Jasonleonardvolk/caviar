#!/usr/bin/env node

/**
 * Fix Cognitive System Imports
 * Updates all cognitive system files to use centralized types
 */

const fs = require('fs');
const path = require('path');

const cognitiveDir = path.join(__dirname, 'tori_ui_svelte', 'src', 'lib', 'cognitive');

console.log('🔧 Fixing Cognitive System Imports\n');
console.log('===================================\n');

// Files to update
const filesToFix = [
  'cognitiveState.ts',
  'cognitiveEngine.ts',
  'braidMemory.ts',
  'memoryMetrics.ts',
  'paradoxAnalyzer.ts',
  'contradictionMonitor.ts',
  'phaseController.ts',
  'closureGuard.ts'
];

// Import replacements
const importFixes = {
  "from './loopRecord'": "from './types'",
  "from './cognitiveState'": "from './types'",
  "type LoopRecord": "type { LoopRecord }",
  "type ConceptDiffState": "type { ConceptDiffState }",
  "type BraidMemory": "type { BraidMemory }",
  "// TEMPORARILY DISABLED - CACHE ISSUES": "",
  "// TEMPORARY LOCAL TYPE DEFINITIONS TO BYPASS CACHE ISSUES": ""
};

let totalFixed = 0;

filesToFix.forEach(file => {
  const filePath = path.join(cognitiveDir, file);
  
  try {
    if (fs.existsSync(filePath)) {
      let content = fs.readFileSync(filePath, 'utf8');
      let modified = false;
      
      // Add import from types if not present
      if (!content.includes("from './types'")) {
        // Find the first import statement
        const firstImportIndex = content.indexOf('import ');
        if (firstImportIndex !== -1) {
          // Add our import at the top
          const importStatement = "import type { LoopRecord, ConceptDiffState, BraidMemory, MemoryMetrics, ParadoxEvent, AssociatorResult, ClosureResult, FeedbackOptions, CognitiveEngineConfig } from './types';\n";
          content = importStatement + content;
          modified = true;
        }
      }
      
      // Apply fixes
      for (const [search, replace] of Object.entries(importFixes)) {
        if (content.includes(search)) {
          content = content.replace(new RegExp(search, 'g'), replace);
          modified = true;
        }
      }
      
      // Remove duplicate type definitions
      const typeDefPattern = /interface (LoopRecord|ConceptDiffState|BraidMemory) \{[\s\S]*?\n\}/g;
      if (typeDefPattern.test(content)) {
        content = content.replace(typeDefPattern, '');
        modified = true;
      }
      
      if (modified) {
        fs.writeFileSync(filePath, content);
        console.log(`✅ Fixed: ${file}`);
        totalFixed++;
      } else {
        console.log(`⏭️  Skipped: ${file} (no changes needed)`);
      }
    } else {
      console.log(`❌ Not found: ${file}`);
    }
  } catch (error) {
    console.error(`❌ Error fixing ${file}:`, error.message);
  }
});

console.log(`\n✨ Fixed ${totalFixed} files`);

// Now check if index_phase3.ts needs fixing
const indexPath = path.join(cognitiveDir, 'index_phase3.ts');
if (fs.existsSync(indexPath)) {
  const content = fs.readFileSync(indexPath, 'utf8');
  
  // Check if the file is malformed (contains write_file syntax)
  if (content.includes('write_file({')) {
    console.log('\n⚠️  index_phase3.ts appears to be malformed');
    console.log('   Creating a clean version...');
    
    // Extract the actual content between the backticks
    const match = content.match(/content: `([^`]+)`/s);
    if (match) {
      const cleanContent = match[1].replace(/\\n/g, '\n').replace(/\\'/g, "'");
      fs.writeFileSync(indexPath, cleanContent);
      console.log('✅ Fixed index_phase3.ts');
    }
  }
}

console.log('\n🎯 Next step: Run "npx tsc --noEmit" to check remaining errors');
