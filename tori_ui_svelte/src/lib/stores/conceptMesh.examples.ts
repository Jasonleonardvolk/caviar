// Example usage and test cases for the concept diff system
import { 
  conceptMeshStore, 
  onConceptDiff, 
  addConcept, 
  addEdge,
  getLastDiff,
  DiffType,
  type DiffEvent 
} from './conceptMesh';

// ========== Example 1: Subscribe to concept diffs ==========
export function subscribeToDiffs() {
  const unsubscribe = onConceptDiff((events: DiffEvent[]) => {
    console.log('📊 Concept mesh changed:', events.length, 'events');
    
    events.forEach(event => {
      switch (event.type) {
        case DiffType.NODE_ADD:
          console.log(`✅ New concept: ${event.node?.label} (${event.id})`);
          break;
          
        case DiffType.NODE_UPDATE:
          console.log(`📝 Updated concept ${event.id}:`);
          if (event.before && event.after) {
            const before = event.before as any;
            const after = event.after as any;
            if (before.phaseTag !== after.phaseTag) {
              console.log(`  Phase: ${before.phaseTag} → ${after.phaseTag}`);
            }
            if (before.weight !== after.weight) {
              console.log(`  Weight: ${before.weight} → ${after.weight}`);
            }
          }
          break;
          
        case DiffType.EDGE_ADD:
          console.log(`🔗 New relation: ${event.edge?.from} → ${event.edge?.to} (${event.edge?.relationType})`);
          break;
          
        case DiffType.EDGE_WEIGHT_SHIFT:
          console.log(`⚖️ Edge weight changed: ${event.from} → ${event.to}: ${event.before} → ${event.after}`);
          break;
          
        case DiffType.NODE_DELETE:
          console.log(`❌ Concept deleted: ${event.id}`);
          break;
          
        case DiffType.EDGE_DELETE:
          console.log(`✂️ Relation removed: ${event.key}`);
          break;
      }
    });
  });
  
  return unsubscribe;
}

// ========== Example 2: Add concepts and watch diffs ==========
export async function testConceptAddition() {
  console.log('🧪 Testing concept addition...');
  
  // Subscribe to diffs first
  const unsubscribe = subscribeToDiffs();
  
  // Add some concepts
  const id1 = await addConcept('quantum mechanics', 'Study of matter at atomic scale', 'physics');
  const id2 = await addConcept('consciousness', 'Subjective experience and awareness', 'philosophy');
  const id3 = await addConcept('quantum consciousness', 'Theory linking quantum mechanics to consciousness', 'theory');
  
  // Add relationships
  await addEdge(id1, id3, 0.8, 'influences');
  await addEdge(id2, id3, 0.9, 'influences');
  await addEdge(id3, id1, 0.3, 'requires');
  
  // Check last diff
  const lastDiff = await getLastDiff();
  console.log('📋 Last diff had', lastDiff.length, 'events');
  
  // Clean up
  setTimeout(() => {
    unsubscribe();
    console.log('✅ Test completed');
  }, 100);
}

// ========== Example 3: UI component integration ==========
export function createDiffVisualizerComponent() {
  return {
    // Svelte component script
    script: `
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { onConceptDiff, DiffType, type DiffEvent } from '$lib/stores/conceptMesh';
  
  let diffs: DiffEvent[] = [];
  let unsubscribe: (() => void) | null = null;
  
  onMount(() => {
    unsubscribe = onConceptDiff((events) => {
      // Add new events to the beginning
      diffs = [...events, ...diffs].slice(0, 50); // Keep last 50
    });
  });
  
  onDestroy(() => {
    if (unsubscribe) unsubscribe();
  });
  
  function getDiffIcon(type: DiffType): string {
    const icons = {
      [DiffType.NODE_ADD]: '➕',
      [DiffType.NODE_DELETE]: '➖',
      [DiffType.NODE_UPDATE]: '📝',
      [DiffType.EDGE_ADD]: '🔗',
      [DiffType.EDGE_DELETE]: '✂️',
      [DiffType.EDGE_WEIGHT_SHIFT]: '⚖️'
    };
    return icons[type] || '❓';
  }
  
  function getDiffColor(type: DiffType): string {
    const colors = {
      [DiffType.NODE_ADD]: 'text-green-600',
      [DiffType.NODE_DELETE]: 'text-red-600',
      [DiffType.NODE_UPDATE]: 'text-blue-600',
      [DiffType.EDGE_ADD]: 'text-purple-600',
      [DiffType.EDGE_DELETE]: 'text-orange-600',
      [DiffType.EDGE_WEIGHT_SHIFT]: 'text-yellow-600'
    };
    return colors[type] || 'text-gray-600';
  }
</script>`,
    
    // Svelte component template
    template: `
<div class="diff-visualizer p-4 bg-gray-50 rounded-lg">
  <h3 class="text-lg font-semibold mb-3">Concept Mesh Activity</h3>
  
  {#if diffs.length === 0}
    <p class="text-gray-500 text-sm">No changes detected yet...</p>
  {:else}
    <div class="space-y-2 max-h-96 overflow-y-auto">
      {#each diffs as diff}
        <div class="flex items-center gap-2 p-2 bg-white rounded border border-gray-200">
          <span class="text-xl" title="{diff.type}">
            {getDiffIcon(diff.type)}
          </span>
          <div class="flex-1 text-sm {getDiffColor(diff.type)}">
            {#if diff.type === DiffType.NODE_ADD}
              Added: {diff.node?.label || diff.id}
            {:else if diff.type === DiffType.NODE_UPDATE}
              Updated: {diff.id}
            {:else if diff.type === DiffType.EDGE_ADD}
              Linked: {diff.edge?.from} → {diff.edge?.to}
            {:else if diff.type === DiffType.EDGE_WEIGHT_SHIFT}
              Weight: {diff.from} → {diff.to} ({diff.before} → {diff.after})
            {:else}
              {diff.type}: {diff.id || diff.key}
            {/if}
          </div>
          <span class="text-xs text-gray-400">
            {new Date(diff.timestamp || Date.now()).toLocaleTimeString()}
          </span>
        </div>
      {/each}
    </div>
  {/if}
</div>`
  };
}

// ========== Example 4: Batch updates and performance test ==========
export async function testBatchPerformance() {
  console.log('🚀 Testing batch performance...');
  
  const startTime = Date.now();
  let eventCount = 0;
  
  const unsubscribe = onConceptDiff((events) => {
    eventCount += events.length;
  });
  
  // Add 100 concepts
  const conceptIds: string[] = [];
  for (let i = 0; i < 100; i++) {
    const id = await addConcept(
      `concept_${i}`, 
      `Description for concept ${i}`, 
      i % 2 === 0 ? 'even' : 'odd'
    );
    conceptIds.push(id);
  }
  
  // Add 200 edges (random connections)
  for (let i = 0; i < 200; i++) {
    const from = conceptIds[Math.floor(Math.random() * conceptIds.length)];
    const to = conceptIds[Math.floor(Math.random() * conceptIds.length)];
    if (from !== to) {
      await addEdge(from, to, Math.random(), 'random');
    }
  }
  
  const duration = Date.now() - startTime;
  console.log(`✅ Batch test completed in ${duration}ms`);
  console.log(`📊 Total diff events: ${eventCount}`);
  
  unsubscribe();
}

// ========== Example 5: Ghost memory integration ==========
export function integrateWithGhostMemory() {
  // When a ghost event occurs, update the concept mesh
  if (typeof window !== 'undefined') {
    window.addEventListener('ghost:new', async (event: any) => {
      const ghost = event.detail;
      
      // Add ghost persona as concept if not exists
      const personaId = await addConcept(
        `ghost.${ghost.persona}`,
        `Ghost persona: ${ghost.persona}`,
        'ghost-persona'
      );
      
      // Add phase state as concept
      const phaseId = await addConcept(
        `phase.${ghost.phaseTag?.toFixed(2)}`,
        `Phase state at ${ghost.phaseTag}`,
        'phase-state'
      );
      
      // Link them
      await addEdge(personaId, phaseId, ghost.coherence || 0.5, 'emerged-at');
      
      console.log(`👻 Ghost ${ghost.persona} linked to concept mesh`);
    });
  }
}

// Export test suite
export const conceptDiffTests = {
  subscribeToDiffs,
  testConceptAddition,
  testBatchPerformance,
  integrateWithGhostMemory
};
