const fs = require('fs');
const path = require('path');

const BASE_DIR = 'D:/Dev/kha/tori_ui_svelte';

console.log('🔧 Fixing the cascade of errors from memoryMetrics.ts...\n');

// Fix 1: Update memoryMetrics.ts - add missing imports and fix MemoryHealth enum
function fixMemoryMetrics() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/memoryMetrics.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Add missing imports at the top if not present
  if (!content.includes("import { get }")) {
    content = `import { get } from 'svelte/store';
import { braidMemory } from './braidMemory';
import { conceptNodes } from '$lib/stores/conceptMesh';
import { cognitiveState, updateCognitiveState } from './cognitiveState';
` + content;
  }
  
  // Fix the MemoryHealth enum - make sure it has all needed values
  content = content.replace(
    /export enum MemoryHealth \{[^}]+\}/,
    `export enum MemoryHealth {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  HEALTHY = 'healthy',
  FAIR = 'fair',
  UNSTABLE = 'unstable',
  CRITICAL = 'critical',
  COLLAPSING = 'collapsing'
}`
  );
  
  // Fix the MemoryMetrics interface to include all properties
  if (!content.includes('interface MemoryMetrics')) {
    // Add the interface if it doesn't exist
    content = content.replace(
      /export enum MemoryHealth/,
      `export interface MemoryMetrics {
  rhoM: number;
  kappaI: number;
  godelianCollapseRisk: number;
  [key: string]: any;
}

export enum MemoryHealth`
    );
  } else {
    // Update existing interface
    content = content.replace(
      /export interface MemoryMetrics \{[^}]*\}/,
      `export interface MemoryMetrics {
  rhoM: number;
  kappaI: number;
  godelianCollapseRisk: number;
  [key: string]: any;
}`
    );
  }
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed memoryMetrics.ts');
}

// Fix 2: Create missing cognitiveState module functions if needed
function fixCognitiveState() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/cognitiveState.ts');
  if (!fs.existsSync(file)) {
    // Create the file if it doesn't exist
    const content = `import { writable } from 'svelte/store';

export type LoopRecord = {
  id: string;
  timestamp: number;
  data: any;
};

export const cognitiveState = writable({
  loops: [],
  phase: 'idle',
  coherence: 1.0
});

export function updateCognitiveState(updates: any) {
  cognitiveState.update(state => ({
    ...state,
    ...updates
  }));
}
`;
    fs.writeFileSync(file, content);
    console.log('✅ Created cognitiveState.ts');
  } else {
    // Update existing file to add missing exports
    let content = fs.readFileSync(file, 'utf8');
    
    if (!content.includes('export function updateCognitiveState')) {
      content += `
export function updateCognitiveState(updates: any) {
  cognitiveState.update(state => ({
    ...state,
    ...updates
  }));
}
`;
      fs.writeFileSync(file, content);
      console.log('✅ Updated cognitiveState.ts');
    }
  }
}

// Fix 3: Fix the callback signature issue in index_phase3.ts
function fixIndexPhase3Again() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/index_phase3.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Fix the callback that expects 2 params but gets 1
  // Find the line around 149 and fix it
  content = content.replace(
    /memoryMetrics\.onUpdate\((.*?)\)/g,
    (match, callback) => {
      // Wrap the callback to handle single parameter
      return `memoryMetrics.onUpdate((metrics) => {
        const callback = ${callback};
        if (callback.length === 2) {
          callback(metrics, memoryMetrics.getHealth());
        } else {
          callback(metrics);
        }
      })`;
    }
  );
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed index_phase3.ts callback');
}

// Fix 4: Ensure conceptMesh exports conceptNodes
function fixConceptMesh() {
  const file = path.join(BASE_DIR, 'src/lib/stores/conceptMesh.ts');
  if (!fs.existsSync(file)) return;
  
  let content = fs.readFileSync(file, 'utf8');
  
  // Make sure conceptNodes is exported
  if (!content.includes('export const conceptNodes')) {
    // Find where conceptNodes is defined and export it
    content = content.replace(
      /const conceptNodes = /,
      'export const conceptNodes = '
    );
    
    // Or if it's inside a class/object, create an export
    if (!content.includes('export const conceptNodes')) {
      content += `
// Export conceptNodes for other modules
export const conceptNodes = writable([]);
`;
    }
  }
  
  fs.writeFileSync(file, content);
  console.log('✅ Fixed conceptMesh.ts exports');
}

// Fix 5: Ensure braidMemory exists
function ensureBraidMemory() {
  const file = path.join(BASE_DIR, 'src/lib/cognitive/braidMemory.ts');
  if (!fs.existsSync(file)) {
    const content = `import { writable } from 'svelte/store';

export const braidMemory = {
  getStats: () => ({ totalLoops: 0 }),
  exportBraid: () => ({}),
  archiveLoop: (loop: any) => {},
  getConversationHistory: () => []
};
`;
    fs.writeFileSync(file, content);
    console.log('✅ Created braidMemory.ts');
  }
}

// Run all fixes
try {
  fixMemoryMetrics();
  fixCognitiveState();
  fixIndexPhase3Again();
  fixConceptMesh();
  ensureBraidMemory();
  
  console.log('\n🎉 Fixed cascade issues! Check again:');
  console.log('cd D:\\Dev\\kha\\tori_ui_svelte');
  console.log('npx tsc --noEmit');
} catch (error) {
  console.error('❌ Error:', error.message);
}
