import * as fs from 'fs';
import * as path from 'path';

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

// Fix 1: Comment out conflicting global declarations
function fixGlobalDTS() {
  const globalPath = path.join(BASE_DIR, 'src/lib/types/global.d.ts');
  if (!fs.existsSync(globalPath)) return;
  
  let content = fs.readFileSync(globalPath, 'utf8');
  
  // Comment out conflicting exports
  const conflictingExports = [
    'bridgeConfig',
    'cognitiveState', 
    'contradictionMonitor',
    'phaseController',
    'closureGuard',
    'cognitiveEngine',
    'braidMemory',
    'memoryMetrics',
    'paradoxAnalyzer',
    'conceptMesh',
    'ghostPersona'
  ];
  
  conflictingExports.forEach(exp => {
    const regex = new RegExp(`export const ${exp}:.*?;`, 'g');
    content = content.replace(regex, `// Commented to fix duplicate: export const ${exp}: ...;`);
  });
  
  fs.writeFileSync(globalPath, content);
  console.log('✅ Fixed global.d.ts conflicts');
}

// Fix 2: Fix BridgeConfig interface
function fixBridgeConfig() {
  const bridgePath = path.join(BASE_DIR, 'src/lib/bridgeConfig.ts');
  if (!fs.existsSync(bridgePath)) return;
  
  let content = fs.readFileSync(bridgePath, 'utf8');
  
  // Add missing properties to bridge configs
  content = content.replace(
    /status: "unknown"\s*}/g,
    `status: "unknown",
        api: { endpoint: '/api' },
        websocket: { endpoint: '/ws' },
        features: []
    }`
  );
  
  fs.writeFileSync(bridgePath, content);
  console.log('✅ Fixed bridgeConfig.ts');
}

// Fix 3: Remove accidental write_file call
function fixCognitiveEngine() {
  const enginePath = path.join(BASE_DIR, 'src/lib/cognitive/cognitiveEngine_phase3.ts');
  if (!fs.existsSync(enginePath)) return;
  
  let content = fs.readFileSync(enginePath, 'utf8');
  
  // Remove the write_file call at the beginning
  if (content.startsWith('write_file')) {
    content = '// Fixed: Removed accidental write_file call\n' + 
              content.replace(/^write_file\({[\s\S]*?\n}\);?\n?/, '');
  }
  
  fs.writeFileSync(enginePath, content);
  console.log('✅ Fixed cognitiveEngine_phase3.ts');
}

// Fix 4: Fix type exports vs value exports
function fixTypeExports() {
  const typesPath = path.join(BASE_DIR, 'src/lib/stores/types.ts');
  if (!fs.existsSync(typesPath)) return;
  
  let content = fs.readFileSync(typesPath, 'utf8');
  
  // Change export { } to export type { } for type-only exports
  content = content.replace(
    /export \{\s*(ConceptDiff|SystemState|ConceptNode|EnhancedConcept|CoherenceMetrics|EntropyMetrics|ConceptMeshEvent|ConceptMeshStorage|NetworkStats|GhostInteraction|ThoughtspaceState|ConceptMeshConfig)[\s\S]*?\}/g,
    (match) => {
      const types = match.match(/\w+/g)?.filter(w => w !== 'export') || [];
      return `export type { ${types.join(', ')} }`;
    }
  );
  
  fs.writeFileSync(typesPath, content);
  console.log('✅ Fixed stores/types.ts exports');
}

// Fix 5: Add missing store exports
function fixStoreExports() {
  const indexPath = path.join(BASE_DIR, 'src/lib/stores/index.ts');
  if (!fs.existsSync(indexPath)) return;
  
  let content = fs.readFileSync(indexPath, 'utf8');
  
  // Add missing exports if not present
  if (!content.includes('export { ghostState')) {
    content += `
// Added missing exports
export { ghostState, activeAgents, conversationLog } from './ghostPersona';
export { vaultEntries, sealedArcs } from './ghostPersona';
export { activateConcept, focusConcept, linkConcepts, addConcept } from './conceptMesh';
`;
  }
  
  fs.writeFileSync(indexPath, content);
  console.log('✅ Fixed stores/index.ts exports');
}

// Run all fixes
console.log('Starting TypeScript fixes for tori_ui_svelte...\n');

fixGlobalDTS();
fixBridgeConfig();
fixCognitiveEngine();
fixTypeExports();
fixStoreExports();

console.log('\n🎉 Main fixes applied!');
console.log('\nNext steps:');
console.log('1. cd D:\\Dev\\kha\\tori_ui_svelte');
console.log('2. npx tsc --noEmit');
console.log('3. See how many errors remain');
