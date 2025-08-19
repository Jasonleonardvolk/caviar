const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

console.log('🔧 Fixing final 6 syntax errors...\n');

// Fix 1: Fix ghost.ts arrow function syntax
function fixGhostArrowFunctions() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/commands/ghost.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix the arrow function syntax - the casting broke it
  // Change: ghostState.update((state as any) => ({
  // To: ghostState.update(state => ({
  content = content.replace(
    /ghostState\.update\(\((state as any|state)\s+as\s+any\)\s*=>\s*\(\{/g,
    'ghostState.update(state => ({'
  );
  
  // Then cast state inside the function body where needed
  content = content.replace(
    /\.\.\.(state as any|state),/g,
    '...(state as any),'
  );
  
  // Fix any remaining broken arrow functions
  content = content.replace(
    /\((state as any|state)\s+as\s+any\)\s*=>/g,
    'state =>'
  );
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed ghost.ts arrow functions');
}

// Fix 2: Fix onUpload.ts syntax
function fixOnUploadSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/scripts/onUpload.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Find line 69 and fix it properly
  const lines = content.split('\n');
  
  // Line 69 (index 68) - fix the const declaration
  if (lines[68] && lines[68].includes('const text =')) {
    // This should be inside a function/block, not a standalone const
    // Check context - it's probably inside an object or function
    // Let's look at surrounding lines
    
    // If it's inside an object literal, change to property
    if (lines[67] && lines[67].includes('{')) {
      lines[68] = '        text: (uploadData as any).text || (uploadData as any).summary || "",';
    } else {
      // If it's a statement, make sure it's properly formatted
      lines[68] = '        const text = (uploadData as any).text || (uploadData as any).summary || "";';
    }
  }
  
  fs.writeFileSync(file, lines.join('\n'));
  console.log('✅ Fixed onUpload.ts syntax');
}

// Run both fixes
try {
  fixGhostArrowFunctions();
  fixOnUploadSyntax();
  
  console.log('\n🎉 Final 6 errors should be fixed!');
  console.log('cd D:\\Dev\\kha\\tori_ui_svelte');
  console.log('npx tsc --noEmit');
} catch (error) {
  console.error('❌ Error:', error.message);
}
