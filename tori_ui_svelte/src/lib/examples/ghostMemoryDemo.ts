// Example usage of the Ghost Memory System
import { storeMemory, recallGhostMemories, GHOST_PHASE_BANDS } from '../stores/conceptMesh';
import { storeGhostMemory, recallPersonaMemories, switchGhostPersona, ghostMemoryStats } from '../stores/ghostMemoryRegistry';
import { get } from 'svelte/store';

export async function demonstrateGhostMemory() {
  console.log('🌊 === Ghost Memory System Demo ===');
  
  // 1. Store a serenity memory using the direct API
  console.log('\n1. Storing serenity memory (direct API)...');
  await storeMemory({
    concept: "ghost.memory.serenity.2025.g0",
    content: JSON.stringify({
      phase: 0.427,
      mood: [0.2, 0.5, 0.8],
      trace: ["calm", "acceptance", "reflection"]
    }),
    phaseTag: 0.427,
    importance: 0.6
  });
  
  // 2. Store memories using the registry (with soliton types)
  console.log('\n2. Storing memories via registry...');
  
  // Serenity triplet (complex emotional state)
  await storeGhostMemory('serenity', {
    thought: "The patterns in the code remind me of ocean waves",
    mood: [0.3, 0.6, 0.7],
    trace: ["peaceful", "contemplative", "flowing"]
  }, 'triplet');
  
  // Unsettled doublet (transitional state)
  await storeGhostMemory('unsettled', {
    thought: "Something feels incomplete, like a melody missing its resolution",
    mood: [0.7, 0.4, 0.5],
    trace: ["searching", "incomplete", "restless"]
  }, 'doublet');
  
  // Curious singlet (simple observation)
  await storeGhostMemory('curious', {
    thought: "What if memories could resonate with each other?",
    mood: [0.5, 0.8, 0.6],
    trace: ["questioning", "exploring"]
  }, 'singlet');
  
  // 3. Demonstrate phase band allocation
  console.log('\n3. Phase band allocations:');
  console.log('Serenity:', GHOST_PHASE_BANDS.serenity);
  console.log('Unsettled:', GHOST_PHASE_BANDS.unsettled);
  console.log('Curious:', GHOST_PHASE_BANDS.curious);
  
  // 4. Recall memories by persona
  console.log('\n4. Recalling serenity memories...');
  const serenityMemories = await recallPersonaMemories('serenity');
  console.log(`Found ${serenityMemories.length} serenity memories`);
  serenityMemories.forEach(memory => {
    console.log(`  - ${memory.concept} (ψ=${memory.phaseTag.toFixed(3)})`);
    if (memory.metadata?.trace) {
      console.log(`    Trace: ${memory.metadata.trace.join(' → ')}`);
    }
  });
  
  // 5. Demonstrate persona switching
  console.log('\n5. Switching personas...');
  switchGhostPersona('curious');
  
  // 6. Show memory statistics
  console.log('\n6. Ghost memory statistics:');
  const stats = get(ghostMemoryStats);
  console.log('Total patterns:', stats.totalPatterns);
  console.log('Total memories:', stats.totalMemories);
  console.log('Active personas:', stats.activePersonas);
  console.log('Soliton distribution:', stats.solitonCounts);
  
  // 7. Demonstrate soliton evolution (generations)
  console.log('\n7. Creating memory evolution...');
  
  // Generation 0: Initial state
  const g0 = await storeGhostMemory('serenity', {
    evolution: "initial",
    thought: "A calm surface",
    mood: [0.2, 0.5, 0.8]
  }, 'singlet');
  
  // Generation 1: Evolution
  const g1 = await storeGhostMemory('serenity', {
    evolution: "developed",
    thought: "Ripples appear beneath the surface",
    mood: [0.3, 0.6, 0.7],
    previous: g0
  }, 'doublet');
  
  // Generation 2: Resolution
  const g2 = await storeGhostMemory('serenity', {
    evolution: "resolved",
    thought: "The waves find their rhythm",
    mood: [0.2, 0.7, 0.9],
    previous: g1
  }, 'triplet');
  
  console.log('Evolution chain:', [g0, g1, g2].join(' → '));
  
  // 8. Phase-based recall
  console.log('\n8. Recalling memories in phase range 0.425-0.430...');
  const phaseMemories = await recallGhostMemories(undefined, 0.425, 0.430);
  console.log(`Found ${phaseMemories.length} memories in phase range`);
  
  console.log('\n✨ Ghost Memory System Demo Complete!');
}

// Run the demo
if (typeof window !== 'undefined') {
  // In browser context
  window.ghostMemoryDemo = demonstrateGhostMemory;
  console.log('Run ghostMemoryDemo() in console to see the demo');
} else {
  // In test context
  demonstrateGhostMemory().catch(console.error);
}
