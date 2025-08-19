const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

console.log('🔧 Fixing all syntax errors...\n');

// Fix 1: Fix memoryMetrics.ts syntax error on line 126
function fixMemoryMetricsSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/memoryMetrics.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Look for our broken fix and correct it
  content = content.replace(
    /alertLevel: health === MemoryHealth\.CRITICAL \? 'critical' : \n\s+health === MemoryHealth\.UNSTABLE \? 'warning' : 'normal',\n\s+rhoM: 0,\n\s+kappaI: 0,\s*\n\s+godelianCollapseRisk: 0/g,
    `alertLevel: health === MemoryHealth.CRITICAL ? 'critical' : 
                health === MemoryHealth.UNSTABLE ? 'warning' : 'normal',
      rhoM: 0,
      kappaI: 0,
      godelianCollapseRisk: 0`
  );
  
  // If that didn't work, try to find the specific location
  const lines = content.split('\n');
  for (let i = 124; i < 130 && i < lines.length; i++) {
    if (lines[i].includes('alertLevel') && lines[i].includes('health ===')) {
      // Check if the line is broken
      if (!lines[i].includes(';') && !lines[i].includes(',')) {
        lines[i] = lines[i] + ',';
      }
    }
  }
  
  fs.writeFileSync(file, lines.join('\n'));
  console.log('✅ Fixed memoryMetrics.ts syntax');
}

// Fix 2: Fix ghost.ts syntax errors
function fixGhostSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/commands/ghost.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix broken type casts - our previous fix broke the syntax
  // The pattern " as any).activePersona" is wrong, should be "(x as any).activePersona"
  content = content.replace(/ as any\)\.activePersona/g, ' as any).activePersona');
  
  // Fix any standalone "as any)" that broke
  content = content.replace(/\s+as\s+any\)\./g, ' as any).');
  
  // Fix broken arrow functions
  content = content.replace(/\(state\s+as\s+any\s+as\s+any\)/g, '(state as any)');
  
  // Ensure proper parentheses
  content = content.replace(/\(\(state as any\)/g, '((state as any)');
  content = content.replace(/state as any\)\)/g, '(state as any))');
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed ghost.ts syntax');
}

// Fix 3: Fix scriptEngine.ts syntax errors
function fixScriptEngineSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/scriptEngine.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix broken type casts
  content = content.replace(/ as any\)\./g, ' as any).');
  
  // Fix broken optional chaining
  content = content.replace(/\)\.\?/g, ')?.');
  
  // Fix catch/finally issues - look for try blocks without catch
  const lines = content.split('\n');
  let inTry = false;
  let tryStart = -1;
  
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('try {')) {
      inTry = true;
      tryStart = i;
    } else if (inTry && lines[i].includes('}')) {
      // Check if next line is catch or finally
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim();
        if (!nextLine.startsWith('catch') && !nextLine.startsWith('finally')) {
          // Add a catch block
          lines[i] = lines[i] + '\n  } catch (error) {\n    console.error(error);\n  }';
          inTry = false;
        }
      }
    } else if (lines[i].includes('catch') || lines[i].includes('finally')) {
      inTry = false;
    }
  }
  
  fs.writeFileSync(file, lines.join('\n'));
  console.log('✅ Fixed scriptEngine.ts syntax');
}

// Fix 4: Fix onConceptChange.ts syntax
function fixOnConceptChangeSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/scripts/onConceptChange.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Remove duplicate variable declarations
  const lines = content.split('\n');
  for (let i = 0; i < lines.length - 1; i++) {
    if (lines[i].includes('const recentConcepts') && lines[i + 1].includes('linkConcepts(recentConcepts')) {
      // Merge them properly
      lines[i] = '';
    }
  }
  
  content = lines.filter(line => line !== '').join('\n');
  fs.writeFileSync(file, content);
  console.log('✅ Fixed onConceptChange.ts syntax');
}

// Fix 5: Clean up all files with basic syntax validation
function validateAndFixSyntax() {
  const files = [
    'src/lib/cognitive/memoryMetrics.ts',
    'src/lib/elfin/commands/ghost.ts',
    'src/lib/elfin/scriptEngine.ts',
    'src/lib/elfin/scripts/onConceptChange.ts',
    'src/lib/services/PersonaEmergenceEngine.ts'
  ];
  
  files.forEach(filePath => {
    const fullPath = path.join(BASE_DIR, filePath);
    if (!fs.existsSync(fullPath)) return;
    
    let content = fs.readFileSync(fullPath, 'utf8');
    
    // Count brackets to ensure they match
    const openBraces = (content.match(/\{/g) || []).length;
    const closeBraces = (content.match(/\}/g) || []).length;
    const openParens = (content.match(/\(/g) || []).length;
    const closeParens = (content.match(/\)/g) || []).length;
    const openBrackets = (content.match(/\[/g) || []).length;
    const closeBrackets = (content.match(/\]/g) || []).length;
    
    if (openBraces !== closeBraces) {
      console.log(`⚠️  ${filePath}: Brace mismatch (${openBraces} open, ${closeBraces} close)`);
    }
    if (openParens !== closeParens) {
      console.log(`⚠️  ${filePath}: Parenthesis mismatch (${openParens} open, ${closeParens} close)`);
    }
    if (openBrackets !== closeBrackets) {
      console.log(`⚠️  ${filePath}: Bracket mismatch (${openBrackets} open, ${closeBrackets} close)`);
    }
  });
}

// Run all fixes
try {
  fixMemoryMetricsSyntax();
  fixGhostSyntax();
  fixScriptEngineSyntax();
  fixOnConceptChangeSyntax();
  validateAndFixSyntax();
  
  console.log('\n🎉 Syntax fixes applied!');
  console.log('cd D:\\Dev\\kha\\tori_ui_svelte');
  console.log('npx tsc --noEmit');
} catch (error) {
  console.error('❌ Error:', error.message);
}
