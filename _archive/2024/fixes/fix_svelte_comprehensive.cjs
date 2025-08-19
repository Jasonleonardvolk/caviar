const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

// Count errors by type
function analyzeErrors() {
  console.log('\n📊 Analyzing error patterns...\n');
  
  // Run tsc and capture output
  const { execSync } = require('child_process');
  let errors;
  try {
    execSync('npx tsc --noEmit', { cwd: BASE_DIR, encoding: 'utf8' });
    console.log('No errors found!');
    return;
  } catch (e) {
    errors = e.stdout || e.output?.join('\n') || '';
  }
  
  // Parse error types
  const errorTypes = {};
  const errorFiles = {};
  
  errors.split('\n').forEach(line => {
    const match = line.match(/error (TS\d+):/);
    if (match) {
      const code = match[1];
      errorTypes[code] = (errorTypes[code] || 0) + 1;
      
      const fileMatch = line.match(/(.+\.ts):(\d+):(\d+) - error/);
      if (fileMatch) {
        const file = fileMatch[1];
        errorFiles[file] = (errorFiles[file] || 0) + 1;
      }
    }
  });
  
  console.log('Error types:');
  Object.entries(errorTypes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([code, count]) => {
      console.log(`  ${code}: ${count} errors`);
    });
  
  console.log('\nFiles with most errors:');
  Object.entries(errorFiles)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([file, count]) => {
      const shortFile = file.replace('src/lib/', '').replace('src/routes/', 'routes/');
      console.log(`  ${shortFile}: ${count} errors`);
    });
}

// Fix all common patterns
function fixAllPatterns() {
  console.log('\n🔧 Applying comprehensive fixes...\n');
  
  // Fix 1: Fix all missing imports in stores/index.ts
  const storeIndexPath = path.join(BASE_DIR, 'src/lib/stores/index.ts');
  if (fs.existsSync(storeIndexPath)) {
    let content = fs.readFileSync(storeIndexPath, 'utf8');
    
    // Add all missing exports
    const missingExports = `
// Export everything from all stores
export * from './conceptMesh';
export * from './ghostPersona';
export * from './persistence';
export * from './session';
export * from './toriStorage';
export * from './types';

// Re-export specific items that might be imported by name
export { 
  ghostState, 
  activeAgents, 
  conversationLog,
  vaultEntries,
  sealedArcs
} from './ghostPersona';

export {
  activateConcept,
  focusConcept,
  linkConcepts,
  addConcept,
  setLastTriggeredGhost
} from './conceptMesh';

// Additional exports for compatibility
export { conceptMesh, systemCoherence } from './conceptMesh';
`;
    
    if (!content.includes('// Export everything from all stores')) {
      content += missingExports;
      fs.writeFileSync(storeIndexPath, content);
      console.log('✅ Fixed stores/index.ts exports');
    }
  }
  
  // Fix 2: Create missing type declarations for routes
  const routesDirs = [
    'src/routes',
    'src/routes/api/chat',
    'src/routes/api/chat/export-all',
    'src/routes/api/chat/history',
    'src/routes/api/ghost-memory/all',
    'src/routes/api/list',
    'src/routes/api/memory/state',
    'src/routes/api/pdf/stats',
    'src/routes/api/soliton/[...path]',
    'src/routes/health',
    'src/routes/login',
    'src/routes/logout',
    'src/routes/upload'
  ];
  
  routesDirs.forEach(dir => {
    const typesPath = path.join(BASE_DIR, dir, '$types.d.ts');
    const dirPath = path.join(BASE_DIR, dir);
    
    if (fs.existsSync(dirPath) && !fs.existsSync(typesPath)) {
      const content = `import type * as Kit from '@sveltejs/kit';

export type PageServerLoad = Kit.ServerLoad<{}>;
export type PageLoad = Kit.Load<{}>;
export type LayoutServerLoad = Kit.ServerLoad<{}>;
export type LayoutLoad = Kit.Load<{}>;
export type RequestHandler = Kit.RequestHandler;
export type Actions = Kit.Actions;
`;
      fs.writeFileSync(typesPath, content);
      console.log(`✅ Created $types.d.ts for ${dir}`);
    }
  });
  
  // Fix 3: Fix command duplicate exports
  const commandFiles = [
    'src/lib/elfin/commands/ghost.ts',
    'src/lib/elfin/commands/project.ts',
    'src/lib/elfin/commands/vault.ts'
  ];
  
  commandFiles.forEach(file => {
    const filePath = path.join(BASE_DIR, file);
    if (fs.existsSync(filePath)) {
      let content = fs.readFileSync(filePath, 'utf8');
      
      // Remove duplicate exports at end of file
      content = content.replace(/export \{ \w+ \};\s*$/gm, '');
      
      // Fix redeclared exports
      content = content.replace(/export async function (\w+).*?\n.*?export \{ \1 \};/gs, 
        (match, name) => match.replace(`export { ${name} };`, ''));
      
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed duplicate exports in ${file}`);
    }
  });
  
  // Fix 4: Fix service type conflicts
  const serviceFiles = [
    'src/lib/services/intentTracking.ts',
    'src/lib/services/typingProsody.ts'
  ];
  
  serviceFiles.forEach(file => {
    const filePath = path.join(BASE_DIR, file);
    if (fs.existsSync(filePath)) {
      let content = fs.readFileSync(filePath, 'utf8');
      
      // Change conflicting export type to export interface
      content = content.replace(/export type \{ ([^}]+) \};/g, 
        '// Types exported individually above');
      
      fs.writeFileSync(filePath, content);
      console.log(`✅ Fixed type exports in ${file}`);
    }
  });
  
  // Fix 5: Fix SharedArrayBuffer issue in photoMorphPipeline
  const morphPath = path.join(BASE_DIR, 'src/lib/webgpu/photoMorphPipeline.fixed.ts');
  if (fs.existsSync(morphPath)) {
    let content = fs.readFileSync(morphPath, 'utf8');
    
    content = content.replace(
      /this\.device\.queue\.writeBuffer\(buffer, 0, data\);/g,
      `// Ensure ArrayBuffer not SharedArrayBuffer
    const safeData = data.buffer instanceof SharedArrayBuffer 
      ? (() => { const arr = new Float32Array(data.length); arr.set(data); return arr; })()
      : data;
    this.device.queue.writeBuffer(buffer, 0, safeData as ArrayBufferView & { buffer: ArrayBuffer });`
    );
    
    fs.writeFileSync(morphPath, content);
    console.log('✅ Fixed photoMorphPipeline SharedArrayBuffer');
  }
}

// Main execution
console.log('🚀 Starting comprehensive TypeScript fix for tori_ui_svelte...\n');

try {
  // First apply all fixes
  fixAllPatterns();
  
  // Then analyze remaining errors
  analyzeErrors();
  
} catch (error) {
  console.error('❌ Error:', error.message);
}
