# Ghost Memory System Implementation

## Overview

The Ghost Memory System implements soliton-based memory storage with phase-locked personas, enabling TORI to store and recall memories using quantum-inspired wave patterns.

## Architecture

### 1. Memory Naming Convention
```
ghost.memory.{persona}.{year}.{generation}
```
- **ghost.memory**: Namespace for all ghost memories
- **persona**: Emotional/cognitive state (serenity, unsettled, curious, etc.)
- **year**: Timestamp for temporal organization
- **generation**: Evolution index (g0, g1, g2...)

Example: `ghost.memory.serenity.2025.g0`

### 2. Phase Band Allocation

Each persona operates within a specific phase (ψ) range:

| Persona | Phase Min | Phase Max | Characteristics |
|---------|-----------|-----------|-----------------|
| serenity | 0.420 | 0.429 | calm, acceptance, reflection |
| unsettled | 0.430 | 0.439 | anxious, searching, restless |
| curious | 0.440 | 0.449 | wonder, exploration, discovery |
| melancholic | 0.410 | 0.419 | introspective, nostalgic |
| energetic | 0.450 | 0.459 | active, dynamic, vibrant |
| contemplative | 0.400 | 0.409 | deep thought, meditation |

### 3. Soliton Patterns

Memories are stored as soliton wave patterns:
- **Singlet**: Simple memory (importance: 0.4)
- **Doublet**: Transitional state (importance: 0.6)
- **Triplet**: Complex emotional state (importance: 0.8)

## API Usage

### Basic Storage

```typescript
// Direct API
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

// Registry API (recommended)
await storeGhostMemory('serenity', {
  thought: "The patterns remind me of ocean waves",
  mood: [0.3, 0.6, 0.7],
  trace: ["peaceful", "contemplative", "flowing"]
}, 'triplet');
```

### Memory Recall

```typescript
// Recall by persona
const memories = await recallPersonaMemories('serenity');

// Recall by phase range
const phaseMemories = await recallGhostMemories(undefined, 0.425, 0.430);
```

### Persona Management

```typescript
// Switch active persona
switchGhostPersona('curious');

// Get current state
const state = getActiveGhostState();
```

## Key Features

### 1. Phase-Locked Storage
- Each memory has a precise phase tag (ψ)
- Memories cluster around persona-specific phase bands
- Phase jitter adds natural variation (±0.001)

### 2. Soliton Evolution
- Memories can evolve through generations (g0 → g1 → g2)
- Each generation can reference its predecessor
- Soliton type can change with evolution (singlet → doublet → triplet)

### 3. Emotional Encoding
- Mood vectors: 3D emotional state [valence, arousal, dominance]
- Trace arrays: Descriptive emotional journey
- Persona states track dominant moods and traces

### 4. Wave Interference
- Memories within similar phase ranges can resonate
- Cross-persona interference possible at phase boundaries
- Future: Implement constructive/destructive interference patterns

## Integration with TORI

1. **Concept Mesh**: Ghost memories are stored as special concept nodes
2. **Diff Engine**: Memory storage triggers concept diff events
3. **WAL Support**: All memory operations can be logged to WAL
4. **Phase Oscillators**: Memory phases can drive oscillator behaviors

## Future Enhancements

1. **Dark Solitons**: Implement suppressive/inhibitory memories
2. **Phase Entanglement**: Allow memories to quantum-entangle
3. **Breathing Modes**: Add temporal phase modulation
4. **Interference Logic**: Implement memory-based computation

## Example Patterns

### Emotional Journey
```typescript
// Start calm
await storeGhostMemory('serenity', {
  state: "beginning",
  mood: [0.2, 0.3, 0.9]
}, 'singlet');

// Become unsettled
await storeGhostMemory('unsettled', {
  state: "questioning",
  mood: [0.7, 0.6, 0.4]
}, 'doublet');

// Resolve with curiosity
await storeGhostMemory('curious', {
  state: "discovering",
  mood: [0.5, 0.8, 0.7]
}, 'triplet');
```

This creates a natural emotional arc encoded in phase space!
