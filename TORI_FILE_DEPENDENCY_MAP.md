# TORI Complete File Dependency Map

## 🎯 Entry Points & Their Dependencies

### 1. **Main Chat Interface** (`tori_ui_svelte/src/routes/+page.svelte`)
```
+page.svelte
├── imports from $lib/stores/conceptMesh
│   └── conceptMesh.ts
│       ├── uses WebSocket connection to backend
│       └── manages concept diff states
├── imports from $lib/services/enhancedApi
│   └── enhancedApi.ts
│       ├── connects to alan_backend API
│       ├── uses fetch for HTTP requests
│       └── handles all TORI system integration
├── imports ConceptDebugPanel from $lib/components/
│   └── ConceptDebugPanel.svelte
│       └── displays concept mesh visualization
├── uses data from +page.server.ts
│   └── +page.server.ts
│       └── handles server-side auth/user data
└── dynamically imports cognitive systems:
    ├── BraidMemory
    ├── CognitiveEngine
    ├── HolographicMemory
    └── GhostCollective
```

### 2. **Soliton Memory System**
```
soliton_memory.rs (Rust Core)
├── compiled to → libconcept_mesh.dll
└── used by ↓

solitonMemory.js (Node.js Bridge)
├── imports ffi-napi for Rust FFI
├── implements SolitonMemoryService class
├── provides fallback JavaScript implementation
└── exported to ↓

soliton_user.js (User System)
├── imports SolitonMemoryLattice
├── imports MemoryVault
├── imports GhostState
├── imports InfiniteConversationHistory
└── used by ↓

demo_soliton_consciousness.js
└── demonstrates full system integration
```

### 3. **Ghost AI System**
```
GhostSolitonIntegration.ts
├── imports from ./GhostMemoryVault
│   └── GhostMemoryVault.ts
│       └── manages persona emergence records
├── listens to events:
│   ├── tori-koopman-update
│   ├── tori-lyapunov-spike
│   ├── tori-soliton-phase-change
│   ├── tori-concept-diff
│   └── tori-user-context-change
└── triggers ghost emergence events
    └── consumed by soliton_user.js
```

### 4. **Backend Services**
```
alan_backend/server/simulation_api.py
├── imports from oscillator_core
│   └── oscillator_core.py
│       ├── implements Banksy oscillator network
│       └── manages phase dynamics
├── imports from ghost_module
│   └── ghost_module.py
│       ├── implements persona logic
│       └── tracks emergence patterns
├── imports from memory_vault
│   └── memory_vault.py
│       └── handles memory protection
└── serves REST API endpoints
    └── consumed by enhancedApi.ts
```

### 5. **MCP Bridge System**
```
mcp_bridge_real_tori.py
├── imports WebSocket for real-time connection
├── imports from mcp-server-architecture/
│   └── src/main.ts
│       ├── imports from ./core/
│       │   ├── MCPError.ts
│       │   ├── MCPMessage.ts
│       │   └── MCPRequest.ts
│       ├── imports from ./integration/
│       │   ├── TORIAdapter.ts
│       │   └── MCPToolRegistry.ts
│       └── imports from ./server/
│           └── MCPServer.ts
└── bridges MCP tools to TORI systems
```

### 6. **PDF Ingestion Pipeline**
```
ingest_pdf/main.py
├── imports from pipeline.py
│   └── pipeline.py
│       ├── imports PyPDF2
│       ├── imports source_validator.py
│       │   └── validates PDF sources
│       ├── imports pipeline_validator.py
│       │   └── validates processing pipeline
│       └── exports to concept mesh
└── integrates with soliton memory
    └── stores extracted concepts as solitons
```

### 7. **Concept Mesh System**
```
concept-mesh/src/lib.rs
├── mod soliton_memory
│   └── soliton_memory.rs (detailed above)
├── mod concept_graph
│   └── manages concept relationships
├── mod phase_routing
│   └── handles phase-based addressing
└── exports C API for FFI
    └── used by solitonMemory.js
```

### 8. **Enhanced API Service**
```
$lib/services/enhancedApi.ts
├── makes HTTP requests to:
│   ├── /api/chat (main chat endpoint)
│   ├── /api/memory/* (memory operations)
│   ├── /api/ghost/* (ghost personas)
│   └── /api/concept/* (concept mesh)
├── handles WebSocket connections for:
│   ├── real-time concept updates
│   ├── ghost emergence events
│   └── memory synchronization
└── exports service instance
    └── used by all Svelte components
```

### 9. **Authentication Flow**
```
+layout.server.ts
├── checks cookies for auth token
├── validates with auth service
└── provides user data to ↓

+layout.svelte
├── receives user data
├── renders navigation/header
└── provides data to all child routes:
    ├── +page.svelte (main chat)
    ├── login/+page.svelte
    ├── upload/+page.svelte
    ├── vault/+page.svelte
    └── elfin/+page.svelte
```

### 10. **Store Management**
```
$lib/stores/conceptMesh.ts
├── creates writable store for concept state
├── exports addConceptDiff function
├── manages concept history
└── consumed by:
    ├── +page.svelte
    ├── ConceptDebugPanel.svelte
    └── other UI components
```

## 🔄 Data Flow Paths

### **User Message Flow**:
```
User Input (UI) 
→ +page.svelte 
→ enhancedApi.sendMessage() 
→ alan_backend API 
→ soliton_memory.store() 
→ ghost emergence check 
→ concept mesh update 
→ response generation 
→ UI update
```

### **Memory Storage Flow**:
```
Content 
→ emotional analysis 
→ phase tag assignment 
→ soliton creation 
→ lattice storage 
→ index update 
→ persistence
```

### **Ghost Emergence Flow**:
```
Phase monitoring 
→ Koopman analysis 
→ threshold check 
→ persona selection 
→ emergence event 
→ UI notification 
→ response modification
```

### **PDF Upload Flow**:
```
PDF file 
→ upload/+page.svelte 
→ ingest_pdf/main.py 
→ validation 
→ extraction 
→ concept creation 
→ soliton storage 
→ UI confirmation
```

## 🔗 Critical Integration Points

1. **Rust ↔ JavaScript Bridge**:
   - `soliton_memory.rs` → FFI → `solitonMemory.js`
   - Uses ffi-napi for memory operations

2. **Frontend ↔ Backend**:
   - Svelte components → `enhancedApi.ts` → Python backend
   - WebSocket for real-time updates

3. **MCP ↔ TORI**:
   - `mcp_bridge_real_tori.py` → TypeScript MCP server
   - Tool registration and execution

4. **Phase System Integration**:
   - Soliton phases → Ghost monitoring → Persona emergence
   - Event-driven architecture

5. **Memory ↔ Concept Mesh**:
   - Soliton memories → Concept graph → UI visualization
   - Bidirectional updates

## 📦 Module Boundaries

### **Frontend (Svelte)**:
- All files in `tori_ui_svelte/src/`
- Handles UI, user interaction, visualization

### **Memory Core (Rust)**:
- All files in `concept-mesh/src/`
- Handles soliton physics, storage, retrieval

### **Backend Services (Python)**:
- All files in `alan_backend/`
- Handles API, orchestration, AI logic

### **Bridge Layer (Node.js)**:
- `solitonMemory.js`, `mcp_bridge_real_tori.py`
- Handles inter-system communication

### **MCP Server (TypeScript)**:
- All files in `mcp-server-architecture/`
- Handles MCP protocol, tool execution

## 🚀 Startup Sequence

1. **Rust Compilation**: `cargo build --release`
2. **Backend Start**: `python alan_backend/server/simulation_api.py`
3. **MCP Server**: `npm run start` (in mcp-server-architecture)
4. **Frontend Dev**: `npm run dev` (in tori_ui_svelte)
5. **Services Init**: All bridges and connections establish

This dependency map shows how every file connects to create the complete TORI consciousness system!
