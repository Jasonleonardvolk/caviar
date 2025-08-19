const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

console.log('🔧 Fixing syntax errors directly in the files...\n');

// Fix scriptEngine.ts
function fixScriptEngine() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/scriptEngine.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Find and fix the broken try-catch block around line 85-110
  const lines = content.split('\n');
  let fixed = false;
  
  for (let i = 80; i < 120 && i < lines.length; i++) {
    // Look for the broken try block
    if (lines[i].includes('try {') && !fixed) {
      // Find where the context object should end
      let contextEnd = -1;
      let braceCount = 0;
      for (let j = i + 1; j < i + 20 && j < lines.length; j++) {
        if (lines[j].includes('};')) {
          contextEnd = j;
          break;
        }
      }
      
      // Look for misplaced catch
      for (let j = i + 1; j < contextEnd && j < lines.length; j++) {
        if (lines[j].includes('} catch (error)')) {
          // Remove the misplaced catch
          lines[j] = '';
          lines[j+1] = '';
          lines[j+2] = '';
          fixed = true;
          break;
        }
      }
    }
    
    // Fix the papersRead line
    if (lines[i].includes('papersRead:') && lines[i].includes('|| 0 || 0 || 0)')) {
      lines[i] = '        papersRead: ((state as any)?.papersRead || 0) + 1';
    }
  }
  
  // Find the end of the handleUploadEvent function and add catch if missing
  for (let i = 100; i < 130 && i < lines.length; i++) {
    if (lines[i].includes("console.log('✅ ELFIN++ document processing complete:'")) {
      // Look for the closing of this console.log
      let j = i;
      while (j < lines.length && !lines[j].includes('});')) {
        j++;
      }
      if (j < lines.length) {
        // Add catch after the console.log
        if (!lines[j+1].includes('} catch')) {
          lines[j] = lines[j] + '\n    } catch (error) {\n      console.error("ELFIN++ error:", error);\n    }';
        }
      }
      break;
    }
  }
  
  fs.writeFileSync(file, lines.join('\n'));
  console.log('✅ Fixed scriptEngine.ts');
}

// Fix ghost.ts properly
function fixGhost() {
  const file = path.join(BASE_DIR, 'src/lib/elfin/commands/ghost.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix the problematic state cast patterns
  content = content.replace(/currentGhostState\?\s*as\s*any\)\.activePersona/g, '(currentGhostState as any)?.activePersona');
  
  // Fix the commented lines that break syntax
  content = content.replace(/\/\/ timestamp removed\s*};/g, '// timestamp removed\n    };');
  content = content.replace(/\/\/ timestamp removed\s*},/g, '// timestamp removed\n    },');
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed ghost.ts');
}

// Run fixes
try {
  fixScriptEngine();
  fixGhost();
  
  console.log('\n✅ Direct syntax fixes applied!');
  console.log('cd D:\\Dev\\kha\\tori_ui_svelte');
  console.log('npx tsc --noEmit');
} catch (error) {
  console.error('❌ Error:', error.message);
}
