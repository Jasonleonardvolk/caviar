# Phase 3: Saigon Mesh-to-Text Bridge + Hologram Voice

## Overview
Implement the Saigon LSTM ↔ neural mesh-to-text bridge to enable natural language generation from concept meshes.

## Deliverables

### 1. Saigon Generator Package
- [ ] Create `saigon_generator/` package
- [ ] Implement lazy model loading (2-4GB weights)
- [ ] Add `TORI_SAIGON_PATH` environment variable support
- [ ] Create download script for model weights

### 2. Mesh-to-Text API
- [ ] Implement `/api/mesh2text` POST endpoint
- [ ] Add `/api/mesh2text/stream` SSE endpoint for real-time generation
- [ ] Support persona-based text generation
- [ ] Add temperature and max_tokens controls

### 3. Frontend Integration
- [ ] Update Svelte components to use SSE stream
- [ ] Connect to HolographicMemory for voice synthesis
- [ ] Add WebAudio or ElevenLabs TTS fallback
- [ ] Create visual feedback during generation

### 4. Testing & Documentation
- [ ] Unit tests for mesh2text API
- [ ] Integration tests with mock Saigon
- [ ] Update API documentation
- [ ] Add usage examples

## Technical Specification

### API Interface
```python
POST /api/mesh2text
{
    "concept_ids": ["concept_123", "concept_456"],
    "max_tokens": 100,
    "temperature": 0.7,
    "persona": {
        "name": "Scholar",
        "style": "academic"
    }
}

Response:
{
    "text": "Generated natural language text...",
    "tokens_generated": 42,
    "concept_ids_used": ["concept_123", "concept_456"]
}
```

### SSE Stream Format
```javascript
GET /api/mesh2text/stream?concept_ids=id1,id2&max_tokens=100

event: token
data: "The"

event: token
data: "quantum"

event: done
data: [DONE]
```

## Implementation Plan

1. **Week 1**: Core Saigon generator with lazy loading
2. **Week 2**: API endpoints and SSE streaming
3. **Week 3**: Frontend integration and voice synthesis
4. **Week 4**: Testing, optimization, and documentation

## Success Criteria
- [ ] Text generation from concept IDs works
- [ ] SSE streaming provides real-time feedback
- [ ] Hologram voice speaks generated text
- [ ] Performance: <2s for 100 tokens
- [ ] Model weights load on-demand (not at startup)
