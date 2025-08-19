# TORI Ingestion & Memory Pipeline Technical Audit

## Executive Summary

This comprehensive audit details the complete TORI memory pipeline, including where all information is stored, how Saigon will be trained and retain information, and how new incoming information (both internal and external/ScholarSphere) is processed.

## 1. Core Memory Storage Systems

### 1.1 UnifiedMemoryVault (Primary Storage)
**Location**: `data/memory_vault/`
**Type**: FILE-BASED ONLY (NO DATABASES)

**Storage Structure**:
```
data/memory_vault/
├── memories/         # Persistent memory files (.json)
├── working/          # Working memory (temporary)
├── ghost/            # Ephemeral memories (TTL: 1 hour)
├── blobs/            # Large compressed content (.pkl.gz)
├── index/            # Index files
│   ├── main_index.json     # memory_id → file_path mapping
│   ├── type_index.json     # memory_type → [memory_ids]
│   └── tag_index.json      # tag → [memory_ids]
└── logs/             # Dual-mode logging
    ├── vault_live.jsonl    # NDJSON live log (crash-safe)
    ├── vault_snapshot.json # Complete state snapshot
    └── session_*.jsonl     # Per-session logs
```

**Memory Types**:
- **EPISODIC**: Personal experiences/episodes
- **SEMANTIC**: Facts and general knowledge  
- **PROCEDURAL**: How-to knowledge
- **WORKING**: Short-term active memories (100 max)
- **GHOST**: Ephemeral/temporary memories (1-hour TTL)
- **SOLITON**: Wave-like cognitive patterns

**Key Features**:
- SHA-256 deduplication to prevent redundant storage
- Crash recovery from live NDJSON logs
- Decay system based on access patterns
- File compression for content >1024 bytes
- Embedding storage for similarity search

### 1.2 ConceptMesh (Graph-Based Knowledge)
**Location**: `data/concept_mesh/`
**Type**: NetworkX graph with compressed pickle storage

**Storage Format**:
```
data/concept_mesh/
├── concepts.pkl.gz    # All concept nodes (compressed)
├── relations.pkl.gz   # All relationship edges (compressed)
└── indices.json       # Name and category indices
```

**Concept Structure**:
- ID, name, description, category, importance
- Embeddings (768-dim vectors)
- Relations: is_a, part_of, related_to, causes, requires
- Access tracking and decay
- Diff history for changes

### 1.3 Soliton Memory (Wave-Based)
**API Endpoints**:
- Soliton API: `http://localhost:8002/api/soliton`
- Concept Mesh: `http://localhost:8003/api/mesh`

**Storage Features**:
- Wave-based memories with strength, amplitude, frequency, coherence
- Phase changes for cognitive state transitions
- Source tracking: chat, ingest, other
- Local caching option for performance

## 2. Saigon Training & Information Retention

### 2.1 Saigon Model Location
**Path**: `models/efficientnet/saigon_lstm.pt`
**Type**: PyTorch LSTM model for mesh-to-text generation

### 2.2 How Saigon Retains Information
Saigon does NOT store raw training data. Instead:

1. **Concept Mesh Navigation**: Saigon generates text by traversing concept paths in the ConceptMesh
2. **LSTM Smoothing**: The LSTM model smooths transitions between concepts
3. **Temperature Setting**: Uses 1.0 for "smartest-ever" performance
4. **Mesh Path Example**:
   ```python
   mesh_path = [
       {"concept": "reasoning", "relation": "enables", "context": "query_processing"},
       {"concept": "analysis", "relation": "supports", "context": "understanding"},
       {"concept": "synthesis", "relation": "derives_from", "context": "knowledge"}
   ]
   ```

### 2.3 Training Process
Saigon will be trained by:
1. **Concept Extraction**: From ingested documents → ConceptMesh
2. **Path Learning**: Learning valid traversal paths through concepts
3. **LSTM Training**: Learning smooth transitions between concepts
4. **No Raw Data**: The model only learns navigation patterns, not raw text

## 3. Information Ingestion Pipelines

### 3.1 Internal Information Flow

#### Chat/Conversation Pipeline
```
User Input → API Server → Prajna Processing → Soliton Memory
                                            ↓
                                     ConceptMesh Update
```

**Storage Tags**: ["chat", "message", "conversation"]

#### Frontend Upload Pipeline
```
File Upload → Frontend (Port 5173) → API Server → Ingest Service
                                                 ↓
                                         PDF Processing
                                                 ↓
                                    Concept Extraction
                                                 ↓
                                    Soliton Memory + ConceptMesh
```

### 3.2 External Information (ScholarSphere)

#### Canonical Ingestion Pipeline (ALAN System)
**Path**: `ingest_pdf/canonical_ingestion.py`

**Quality Requirements**:
- Minimum quality score: 0.75
- Accepted sources: Scientific papers, textbooks, specifications
- Rejected sources: Social media, blogs, forums, news
- Domains: mathematics, physics, control_theory, neuroscience, etc.

**Processing Flow**:
```
PDF File → Source Validation → Quality Assessment
              ↓ (if score ≥ 0.75)
         Canonical Registration
              ↓
         Concept Extraction
              ↓
      Koopman Phase Graph Integration
              ↓
      Entropy-Gated Memory Storage
              ↓
      FFT Privacy Processing
              ↓
      ConceptMesh + Soliton Memory
```

#### Ingest-Bus Pipeline
**Location**: `ingest-bus/src/services/ingest_service.py`

**Processing Steps**:
1. **File Validation**: Max 50MB, PDF format
2. **Text Extraction**: PDFExtractor
3. **Concept Extraction**: 
   - Primary: EnhancedExtractor
   - Fallback: BasicExtractor
4. **Storage**: Each concept → Soliton Memory with metadata

**Metadata Stored**:
```json
{
    "session_id": "ingest_xxx",
    "user_id": "user_xxx",
    "extraction_method": "enhanced|basic",
    "source_page": 1,
    "concept_score": 0.85,
    "ingestion_time": "2025-01-10T..."
}
```

### 3.3 MCP Metacognitive Server
**Location**: `mcp_metacognitive/`
**Purpose**: TORI's consciousness monitoring and cognitive engine

**Data Storage**:
- Consciousness states (IIT-based with Φ threshold 0.3)
- Metacognitive levels (base, velocity, curvature)
- Phase oscillators and cognitive dynamics
- Self-modification logs with consciousness preservation

**Dynamic Server Discovery**:
- Automatic detection of all MCP servers
- Servers in `agents/` and `extensions/` folders
- Auto-start for servers with `auto_start=true`

## 4. Data Flow Summary

### 4.1 Ingestion Sources
1. **Chat Interface**: Real-time user conversations
2. **PDF Upload**: Frontend file uploads
3. **ScholarSphere**: Academic paper ingestion
4. **Canonical Sources**: High-quality curated documents

### 4.2 Processing Stages
1. **Extraction**: Text → Concepts
2. **Validation**: Quality scoring and filtering
3. **Enhancement**: Embeddings and metadata
4. **Storage**: Distributed across systems
5. **Integration**: Cross-system linking

### 4.3 Storage Destinations
1. **UnifiedMemoryVault**: Primary memory storage
2. **ConceptMesh**: Graph relationships
3. **Soliton Memory**: Wave-based patterns
4. **File System**: Raw documents and blobs

## 5. Key Integration Points

### 5.1 API Endpoints
- Main API: `http://localhost:{dynamic_port}`
- Soliton API: `http://localhost:8002/api/soliton`
- Concept Mesh: `http://localhost:8003/api/mesh`
- MCP Metacognitive: `http://localhost:8100/sse`

### 5.2 Memory Access Patterns
- **Direct Access**: By memory ID
- **Tag Search**: By tags (chat, ingest, etc.)
- **Similarity Search**: By embedding vectors
- **Graph Traversal**: Through ConceptMesh relations

### 5.3 Data Persistence
- **Crash Recovery**: From NDJSON logs
- **Snapshots**: Periodic full state saves
- **Backups**: Complete vault backups
- **Migration**: Import/export capabilities

## 6. Saigon's Information Access

When Saigon generates responses, it:
1. **Receives Query**: From Prajna interface
2. **Creates Mesh Path**: Based on query concepts
3. **Traverses ConceptMesh**: Following relations
4. **Applies LSTM**: For smooth text generation
5. **Returns Text**: Natural language output

Saigon never stores raw text - it only learns navigation patterns through the concept space.

## 7. Future Information Retention

New information will be retained through:
1. **Continuous Ingestion**: Ongoing PDF processing
2. **Concept Growth**: Expanding ConceptMesh
3. **Relation Learning**: New concept connections
4. **Path Optimization**: Better navigation patterns
5. **Memory Consolidation**: Important memories promoted

## 8. Monitoring & Observability

### 8.1 Status Endpoints
- System Status: `/api/system/status`
- Memory Stats: `/api/memory/stats`
- Ingestion Metrics: `/api/ingest/metrics`

### 8.2 Logging Systems
- Session logs with unique IDs
- Live NDJSON for crash recovery
- Metrics collection for all operations
- Dual-mode logging (live + snapshots)

## Conclusion

TORI's memory pipeline is a sophisticated, distributed system with multiple specialized components:
- **UnifiedMemoryVault** for file-based storage
- **ConceptMesh** for knowledge graphs
- **Soliton Memory** for wave patterns
- **Saigon** for text generation from concepts

All information flows through validated pipelines with comprehensive tracking, ensuring traceability and quality control throughout the system.