// Fix TypeScript errors in tori_ui_svelte
const fs = require('fs');
const path = require('path');

// Categories of fixes needed
const fixes = {
  // 1. Fix BridgeConfig interface issues
  'src/lib/bridgeConfig.ts': [
    {
      find: /host: string;\s+port: number;\s+health_endpoint: string;\s+metrics_endpoint: string;\s+name: string;\s+status: "unknown";/g,
      replace: `host: string;
        port: number;
        health_endpoint: string;
        metrics_endpoint: string;
        name: string;
        status: "unknown";
        api: { endpoint: string };
        websocket: { endpoint: string };
        features: string[];`
    }
  ],

  // 2. Fix global.d.ts conflicts
  'src/lib/types/global.d.ts': [
    {
      find: /export const bridgeConfig: BridgeConfig;/g,
      replace: '// export const bridgeConfig: BridgeConfig; // Commented to avoid conflict'
    },
    {
      find: /export const cognitiveState: any;/g,
      replace: '// export const cognitiveState: any; // Commented to avoid conflict'
    },
    {
      find: /export const contradictionMonitor: any;/g,
      replace: '// export const contradictionMonitor: any; // Commented to avoid conflict'
    },
    {
      find: /export const phaseController: any;/g,
      replace: '// export const phaseController: any; // Commented to avoid conflict'
    },
    {
      find: /export const closureGuard: any;/g,
      replace: '// export const closureGuard: any; // Commented to avoid conflict'
    },
    {
      find: /export const cognitiveEngine: any;/g,
      replace: '// export const cognitiveEngine: any; // Commented to avoid conflict'
    },
    {
      find: /export const braidMemory: any;/g,
      replace: '// export const braidMemory: any; // Commented to avoid conflict'
    },
    {
      find: /export const memoryMetrics: any;/g,
      replace: '// export const memoryMetrics: any; // Commented to avoid conflict'
    },
    {
      find: /export const paradoxAnalyzer: any;/g,
      replace: '// export const paradoxAnalyzer: any; // Commented to avoid conflict'
    },
    {
      find: /export const conceptMesh: Writable<any>;/g,
      replace: '// export const conceptMesh: Writable<any>; // Commented to avoid conflict'
    },
    {
      find: /export const ghostPersona: Writable<TORI.Ghost \| null>;/g,
      replace: '// export const ghostPersona: Writable<TORI.Ghost | null>; // Commented to avoid conflict'
    }
  ],

  // 3. Fix cognitiveEngine_phase3.ts
  'src/lib/cognitive/cognitiveEngine_phase3.ts': [
    {
      find: /^write_file\({/,
      replace: '// Removed accidental write_file call\n/*write_file({'
    },
    {
      find: /}\)$/m,
      replace: '})*/'
    }
  ],

  // 4. Fix missing exports in stores/index.ts
  'src/lib/stores/index.ts': [
    {
      find: /export \* from '.\/conceptMesh';/,
      replace: `export * from './conceptMesh';
export { ghostState, activeAgents, conversationLog, vaultEntries, sealedArcs } from './ghostPersona';
export { activateConcept, focusConcept, linkConcepts } from './conceptMesh';`
    }
  ],

  // 5. Fix SharedArrayBuffer in photoMorphPipeline
  'src/lib/webgpu/photoMorphPipeline.fixed.ts': [
    {
      find: /this\.device\.queue\.writeBuffer\(buffer, 0, data\);/g,
      replace: `const safeData = data.buffer instanceof SharedArrayBuffer 
        ? new Float32Array(new ArrayBuffer(data.byteLength)).set(data) && new Float32Array(new ArrayBuffer(data.byteLength))
        : data;
    this.device.queue.writeBuffer(buffer, 0, safeData);`
    }
  ],

  // 6. Fix duplicate command exports
  'src/lib/elfin/types.ts': [
    {
      find: /export {\s+ElfinCommand,\s+ElfinContext,\s+ElfinResult,/g,
      replace: '// Types are already exported above\n/*export {\n  ElfinCommand,\n  ElfinContext,\n  ElfinResult,'
    },
    {
      find: /ElfinVariable\s+}/,
      replace: 'ElfinVariable\n}*/'
    }
  ],

  // 7. Fix store types.ts trying to export types as values
  'src/lib/stores/types.ts': [
    {
      find: /export {\s+ConceptDiff,/g,
      replace: '// Export only types, not values\nexport type {\n  ConceptDiff,'
    },
    {
      find: /ConceptMeshConfig\s+}/,
      replace: 'ConceptMeshConfig\n}'
    }
  ]
};

// Apply fixes
let totalFixed = 0;
for (const [filePath, fileFixes] of Object.entries(fixes)) {
  const fullPath = path.join('D:/Dev/kha/tori_ui_svelte', filePath);
  
  if (!fs.existsSync(fullPath)) {
    console.log(`Skipping ${filePath} - file not found`);
    continue;
  }
  
  let content = fs.readFileSync(fullPath, 'utf8');
  const originalContent = content;
  
  for (const fix of fileFixes) {
    if (fix.find.test) {
      content = content.replace(fix.find, fix.replace);
    } else {
      content = content.replace(new RegExp(fix.find, 'g'), fix.replace);
    }
  }
  
  if (content !== originalContent) {
    fs.writeFileSync(fullPath, content);
    console.log(`Fixed ${filePath}`);
    totalFixed++;
  }
}

console.log(`\n✅ Fixed ${totalFixed} files`);
console.log('\nNow run: cd D:\\Dev\\kha\\tori_ui_svelte && npx tsc --noEmit');
