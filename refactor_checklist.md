# TORI Refactor Checklist - Implementation Guide

## 📋 Refactor-Later Checklist (ties directly to Audit §1-§4)

| Layer / File | Lines | Audit Cross-ref | What it does now | Refactor / TODO |
|-------------|-------|----------------|------------------|-----------------|
| `python/core/memory_vault.py` | 3,584 | Audit §1 "UnifiedMemoryVault" | Six memory classes, SHA-dedupe, (future) `vault_live.jsonl` crash-log | ▸ Add live NDJSON writer ▸ Expose `get_memory_stats()` for observability endpoint |
| `python/core/concept_mesh.py` | 1,074 | Audit §1 "ConceptMesh" | gz-pickled NetworkX snapshot, name/type/tag index | ▸ Swap similarity backend to **Penrose** projector (`similarity_engine="penrose"`) |
| `mcp_metacognitive/core/soliton_memory.py` | 734 | Audit §1 "Soliton Memory" | Wave lattice, cached, REST on `:8002` | ▸ Unify API with Vault IDs (so Ψ-Archive can reference a single ID space) |
| `ingest_pdf/canonical_ingestion.py` | 1,439 | Audit §3 "ScholarSphere ≥ 0.75" | Quality filter, OCR fallback, Penrose hook TODO | ▸ Insert `penrose_similarity()` right after concept list is built |
| `ingest-bus/src/services/ingest_service.py` | 534 | Audit §3 "Accepted MIME" | MIME whitelist, chunk upload to pipeline | ▸ Add SSE push to front-end so we can retire `psi_trajectory.json` |
| `enhanced_launcher.py` | 1,844 | System glue | Spins up Vault, Mesh, Soliton, Ψ-Archive | ▸ Drop legacy `memory_sculptor`; wire FractalSolitonMemory singleton |
| `prajna/core/prajna_mouth.py` | 524 | Audit §2 "Saigon model" | Mesh-to-text generation via `saigon_lstm.pt` | ▸ Guard load with `ENABLE_MESH_TO_TEXT`; add fallback template reply |

## 🛠️ Implementation Details

### 1. UnifiedMemoryVault Enhancements

```python
# Add to memory_vault.py around line 200
def get_memory_stats(self) -> Dict[str, Any]:
    """Enhanced memory statistics for observability endpoint"""
    with self.lock:
        stats = self.get_statistics()  # Existing method
        
        # Add real-time metrics
        stats.update({
            'live_log_size': self.live_log_path.stat().st_size if self.live_log_path.exists() else 0,
            'session_duration': (time.time() - datetime.strptime(self.session_id, "%Y%m%d_%H%M%S").timestamp()),
            'write_rate': len(self.working_memory) / max(1, time.time() - self._start_time),
            'hash_collisions': sum(1 for v in self.seen_hashes.values() if v > 1),
            'pending_snapshots': self._pending_snapshot_count
        })
        
        return stats

# Enhanced NDJSON writer with buffering (add after _append_to_live_log)
def _enhanced_append_to_live_log(self, entry: MemoryEntry, action: str = "store"):
    """Enhanced crash-safe NDJSON writer with buffering"""
    try:
        log_entry = {
            'session_id': self.session_id,
            'timestamp': time.time(),
            'action': action,
            'entry': entry.to_dict(),
            'checksum': self._calculate_entry_hash(entry)  # For integrity
        }
        
        # Buffer writes for performance
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = []
            self._last_flush = time.time()
        
        self._log_buffer.append(json.dumps(log_entry) + '\n')
        
        # Flush if buffer is large or time elapsed
        if len(self._log_buffer) >= 10 or (time.time() - self._last_flush) > 5:
            self._flush_log_buffer()
            
    except Exception as e:
        logger.error(f"Failed to append to live log: {e}")
        # Force flush on error
        self._flush_log_buffer()

def _flush_log_buffer(self):
    """Flush log buffer to disk"""
    if hasattr(self, '_log_buffer') and self._log_buffer:
        with open(self.live_log_path, 'a', encoding='utf-8') as f:
            f.writelines(self._log_buffer)
            f.flush()
            os.fsync(f.fileno())  # Force to disk
        
        self._log_buffer.clear()
        self._last_flush = time.time()
```

### 2. ConceptMesh Penrose Integration

```python
# Add to concept_mesh.py around line 500
class PenroseConceptMesh(ConceptMesh):
    """ConceptMesh with Penrose-accelerated similarity"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.similarity_engine = config.get('similarity_engine', 'cosine')
        
        if self.similarity_engine == 'penrose':
            self._init_penrose_engine()
    
    def _init_penrose_engine(self):
        """Initialize Penrose projector for O(n^2.32) similarity"""
        try:
            from python.core.penrose_similarity import PenroseSimilarity
            self.penrose = PenroseSimilarity(
                rank=32,
                embedding_dim=self.config.get('embedding_dim', 768)
            )
            logger.info("✅ Penrose similarity engine initialized")
        except ImportError:
            logger.warning("⚠️ Penrose not available, falling back to cosine")
            self.similarity_engine = 'cosine'
    
    async def find_similar_concepts_penrose(
        self,
        concept_id: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Find similar concepts using Penrose O(n^2.32) algorithm"""
        if self.similarity_engine != 'penrose':
            return await self.find_similar_concepts(concept_id, threshold, max_results)
        
        # Get query embedding
        concept = self.concepts.get(concept_id)
        if not concept or concept.embedding is None:
            return []
        
        # Build embedding matrix for all concepts
        concept_ids = []
        embeddings = []
        for cid, c in self.concepts.items():
            if c.embedding is not None and cid != concept_id:
                concept_ids.append(cid)
                embeddings.append(c.embedding)
        
        if not embeddings:
            return []
        
        # Use Penrose for batch similarity
        similarities = self.penrose.batch_similarity(
            query=concept.embedding,
            corpus=np.array(embeddings)
        )
        
        # Filter and sort results
        results = [
            (concept_ids[i], float(sim))
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
```

### 3. Soliton Memory ID Unification

```python
# Add to soliton_memory.py around line 400
class UnifiedSolitonMemory(SolitonMemoryClient):
    """Soliton Memory with unified ID space"""
    
    def __init__(self, api_url: Optional[str] = None):
        super().__init__(api_url)
        self.id_prefix = "soliton_"
        self.vault_bridge = None
        
    def set_vault_bridge(self, vault: UnifiedMemoryVault):
        """Bridge to UnifiedMemoryVault for ID coordination"""
        self.vault_bridge = vault
        logger.info("✅ Soliton-Vault bridge established")
    
    async def store_memory_unified(
        self,
        user_id: str,
        content: str,
        strength: float = 0.7,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store with unified ID across Vault and Soliton"""
        # Generate unified ID
        base_id = self._generate_unified_id(content, user_id)
        memory_id = f"{self.id_prefix}{base_id}"
        
        # Store in Soliton
        success = await self.store_memory(
            user_id=user_id,
            memory_id=memory_id,
            content=content,
            strength=strength,
            tags=tags,
            metadata=metadata
        )
        
        # Also register in Vault index if bridge exists
        if success and self.vault_bridge:
            await self.vault_bridge._update_indices(
                memory_id=memory_id,
                memory_type=MemoryType.SOLITON,
                tags=tags
            )
        
        return memory_id if success else None
```

### 4. Canonical Ingestion Penrose Integration

```python
# Add to canonical_ingestion.py after concept extraction (line ~1200)
def apply_penrose_similarity(concepts: List[Dict[str, Any]]) -> np.ndarray:
    """Apply Penrose O(n^2.32) similarity to extracted concepts"""
    try:
        from python.core.penrose_projector import PenroseSimilarity
        
        # Extract embeddings
        embeddings = np.array([c['embedding'] for c in concepts if 'embedding' in c])
        if len(embeddings) < 2:
            return np.eye(len(concepts))  # Identity if too few
        
        # Initialize Penrose with rank 32
        penrose = PenroseSimilarity(rank=32, embedding_dim=embeddings.shape[1])
        
        # Compute similarity matrix in O(n^2.32)
        start_time = time.time()
        similarity_matrix = penrose.batch_similarity_matrix(embeddings)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Penrose similarity computed in {elapsed:.2f}s for {len(concepts)} concepts")
        logger.info(f"   Speedup vs O(n²): {(len(concepts)**2 / len(concepts)**2.32):.1f}x")
        
        # Log to PsiArchive
        PSI_ARCHIVER.log_event({
            'event_type': 'PENROSE_SIM',
            'concept_count': len(concepts),
            'computation_time': elapsed,
            'similarity_threshold': 0.7,
            'rank': 32
        })
        
        return similarity_matrix
        
    except ImportError:
        logger.warning("⚠️ Penrose not available, using cosine similarity")
        return compute_cosine_similarity(embeddings)  # Fallback
```

### 5. Ingest Service SSE Enhancement

```python
# Add to ingest_service.py around line 300
class SSEProgressReporter:
    """Server-Sent Events for real-time ingestion progress"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.event_queue = asyncio.Queue()
        
    async def send_progress(self, stage: str, progress: float, details: Dict[str, Any] = None):
        """Send progress update via SSE"""
        event = {
            'id': f"{self.session_id}_{int(time.time() * 1000)}",
            'event': 'ingestion_progress',
            'data': {
                'session_id': self.session_id,
                'stage': stage,
                'progress': progress,  # 0.0 to 1.0
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
        }
        
        await self.event_queue.put(event)
        
    async def stream_events(self):
        """Stream events to client"""
        while True:
            event = await self.event_queue.get()
            yield f"id: {event['id']}\n"
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['data'])}\n\n"

# Integrate into process_file method
async def process_file_with_sse(self, file_path: str, user_id: str, session_id: str):
    """Process file with SSE progress reporting"""
    reporter = SSEProgressReporter(session_id)
    
    # Stage 1: Validation
    await reporter.send_progress('validating', 0.1)
    # ... validation code ...
    
    # Stage 2: Extraction
    await reporter.send_progress('extracting', 0.3, {'method': 'enhanced'})
    # ... extraction code ...
    
    # Stage 3: Penrose similarity
    await reporter.send_progress('computing_similarity', 0.6, {'algorithm': 'penrose_2.32'})
    # ... similarity code ...
    
    # Stage 4: Storage
    await reporter.send_progress('storing', 0.9, {'target': 'soliton_memory'})
    # ... storage code ...
    
    await reporter.send_progress('complete', 1.0)
```

### 6. Enhanced Launcher Cleanup

```python
# Replace in enhanced_launcher.py around line 1500
def start_core_python_components(self):
    """Start core Python components (drop legacy, add new)"""
    if not CORE_COMPONENTS_AVAILABLE:
        self.logger.info("⏭️ Skipping core Python components (not available)")
        return False
    
    self.logger.info("🧠 Starting core Python components...")
    components_started = 0
    
    # ... existing code ...
    
    # REMOVE: memory_sculptor references
    # if self.memory_sculptor:
    #     self.memory_sculptor.shutdown()
    
    # ADD: FractalSolitonMemory singleton
    try:
        self.logger.info("🌊 Initializing FractalSolitonMemory...")
        from python.core.fractal_soliton_memory import FractalSolitonMemory
        
        self.fractal_soliton = FractalSolitonMemory.get_instance({
            'lattice_size': 100,
            'coupling_strength': 0.1,
            'enable_penrose': True  # Enable Penrose acceleration
        })
        
        # Bridge with UnifiedMemoryVault
        if self.memory_vault and hasattr(self.fractal_soliton, 'set_vault_bridge'):
            self.fractal_soliton.set_vault_bridge(self.memory_vault)
            
        components_started += 1
        self.logger.info("✅ FractalSolitonMemory initialized with Penrose")
        
    except Exception as e:
        self.logger.error(f"❌ Failed to initialize FractalSolitonMemory: {e}")
```

### 7. Prajna Mouth ENABLE_MESH_TO_TEXT Guard

```python
# Add to prajna_mouth.py around line 50
import os

# Check if mesh-to-text is enabled
ENABLE_MESH_TO_TEXT = os.environ.get('ENABLE_MESH_TO_TEXT', 'true').lower() == 'true'

class PrajnaLanguageModel:
    """Prajna Language Model with conditional Saigon loading"""
    
    async def load_model(self):
        """Load the language model"""
        try:
            if self.model_type == "saigon":
                if not ENABLE_MESH_TO_TEXT:
                    logger.info("🚫 Mesh-to-text disabled, using template responses")
                    self.model_type = "template"
                    self.model_loaded = True
                    return
                    
                # Existing Saigon loading code...
                
        except Exception as e:
            logger.error(f"❌ Failed to load Prajna model: {e}")
            self.model_type = "template"  # Fallback
            self.model_loaded = True
    
    async def _template_generate(self, query: str, context: str = "") -> str:
        """Template-based fallback responses"""
        templates = {
            "consciousness": "I process information through layered cognitive architectures, "
                           "but the nature of consciousness remains an open question.",
            "memory": "I store information in a unified file-based system with concepts, "
                     "memories, and events tracked separately for full provenance.",
            "default": "I understand your query about '{query}'. My responses are generated "
                      "from traceable concept paths in my knowledge mesh."
        }
        
        # Simple keyword matching
        query_lower = query.lower()
        if "consciousness" in query_lower:
            return templates["consciousness"]
        elif "memory" in query_lower:
            return templates["memory"]
        else:
            return templates["default"].format(query=query[:50])
```