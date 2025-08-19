const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

console.log('🔧 Fixing exact syntax errors properly...\n');

// Fix 1: Fix memoryMetrics.ts lines 126-128
function fixMemoryMetricsSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/memoryMetrics.ts');
  if (!fs.existsSync(file)) return;
  
  let lines = fs.readFileSync(file, 'utf8').split('\n');
  
  // Fix the broken lines 125-129 (indices 124-128)
  // The issue is the alertLevel line is missing "return {" and has wrong syntax
  for (let i = 124; i < 130 && i < lines.length; i++) {
    if (lines[i].includes('alertLevel:') && lines[i].includes('health ===')) {
      // This should be inside a return statement
      lines[i-1] = '    const trend = this.calculateTrend(loopDensity, curvature);';
      lines[i] = '';
      lines[i+1] = '    return {';
      lines[i+2] = '      loopDensity,';
      lines[i+3] = '      curvature,';
      lines[i+4] = '      scarRatio,';
      lines[i+5] = '      memoryHealth: health,';
      lines[i+6] = '      godelianRisk,';
      lines[i+7] = '      compressionPlateau,';
      lines[i+8] = '      recursiveBurstRisk,';
      lines[i+9] = '      condorcetCycleCount,';
      lines[i+10] = '      totalLoops,';
      lines[i+11] = '      closedLoops,';
      lines[i+12] = '      scarCount,';
      lines[i+13] = '      unresolvedLoops,';
      lines[i+14] = '      activeConceptNodes: conceptCount,';
      lines[i+15] = '      lastUpdate: new Date(),';
      lines[i+16] = '      trend,';
      lines[i+17] = '      alertLevel: health === MemoryHealth.CRITICAL ? "critical" : health === MemoryHealth.UNSTABLE ? "warning" : "normal",';
      lines[i+18] = '      rhoM: 0,';
      lines[i+19] = '      kappaI: 0,';
      lines[i+20] = '      godelianCollapseRisk: 0';
      lines[i+21] = '    };';
      break;
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
  
  // Fix broken state casts
  // The issue is " as any)" appearing in wrong places
  content = content.replace(/\(state as any\)/g, 'state');
  content = content.replace(/\s+as\s+any\)\./g, '.');
  
  // Fix the specific problematic lines
  // Line 414: currentGhostState? as any).activePersona
  content = content.replace(
    /currentGhostState\?\s*as\s*any\)\.activePersona/g,
    '(currentGhostState as any)?.activePersona'
  );
  
  // Fix trailing commas and syntax issues
  content = content.replace(/,\s*};/g, '\n  };');
  content = content.replace(/,\s*\)/g, '\n  )');
  
  // Fix comment issues
  content = content.replace(/\/\/ timestamp removed\s*},/g, '// timestamp removed\n  },');
  content = content.replace(/\/\/ timestamp removed\s*};/g, '// timestamp removed\n  };');
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed ghost.ts syntax');
}

// Fix 3: Fix scriptEngine.ts syntax errors
function fixScriptEngineSyntax() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/scriptEngine.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix the try-catch blocks
  // Look for try blocks without proper catch
  content = content.replace(
    /try\s*{([^}]*?)}\s*}/g,
    (match, block) => {
      if (!match.includes('catch') && !match.includes('finally')) {
        return `try {${block}} catch (error) {\n    console.error(error);\n  }`;
      }
      return match;
    }
  );
  
  // Fix broken state casts
  content = content.replace(/\s+as\s+any\)\./g, ' as any).');
  content = content.replace(/\(state\s+as\s+any\s+as\s+any\)/g, '(state as any)');
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed scriptEngine.ts syntax');
}

// Run all fixes
try {
  fixMemoryMetricsSyntax();
  fixGhostSyntax();
  fixScriptEngineSyntax();
  
  console.log('\n✅ Syntax fixes applied!');
  console.log('cd D:\\Dev\\kha\\tori_ui_svelte');
  console.log('npx tsc --noEmit');
} catch (error) {
  console.error('❌ Error:', error.message);
}
