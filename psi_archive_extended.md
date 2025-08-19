# PsiArchive Extended Implementation - Rich Parent/Child Links & Spectral Replay

## 🔗 Enhanced PsiArchive Schema

### Base Event Structure (Extended)
```python
# core/psi_archive_extended.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json

@dataclass
class PsiEvent:
    """Extended PsiArchive event with rich provenance"""
    # Core fields (existing)
    event_id: str
    event_type: str  # LEARNING_UPDATE, RESPONSE_EVENT, PLAN_CREATION, etc.
    timestamp: datetime
    
    # NEW: Provenance fields
    parent_id: Optional[str] = None  # Event that caused this event
    source_doc_sha: Optional[str] = None  # SHA-256 of source document
    concept_ids: List[str] = field(default_factory=list)  # Concepts touched
    
    # NEW: Threading
    session_id: Optional[str] = None  # Chat/ingest session
    thread_id: Optional[str] = None  # Plan/debate chain
    
    # NEW: Delta tracking
    mesh_delta: Optional[Dict[str, Any]] = None  # Added nodes/edges
    
    # Existing fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_ndjson_line(self) -> str:
        """Serialize to NDJSON line"""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'parent_id': self.parent_id,
            'source_doc_sha': self.source_doc_sha,
            'concept_ids': self.concept_ids,
            'session_id': self.session_id,
            'thread_id': self.thread_id,
            'mesh_delta': self.mesh_delta,
            'metadata': self.metadata
        }
        return json.dumps(data, ensure_ascii=False)
```

## 📝 Implementation Sketches

### 1. Parent/Child & Provenance Fields

```python
# Extend existing log_learning_update method
class EnhancedPsiArchiver:
    """PsiArchive with rich provenance tracking"""
    
    def log_concept_ingestion(
        self,
        concept_ids: List[str],
        source_doc_path: str,
        parent_event_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log concept ingestion with full provenance"""
        # Calculate SHA-256 of source document
        with open(source_doc_path, 'rb') as f:
            source_doc_sha = hashlib.sha256(f.read()).hexdigest()
        
        event = PsiEvent(
            event_id=self._generate_event_id(),
            event_type='LEARNING_UPDATE',
            timestamp=datetime.now(),
            parent_id=parent_event_id or session_id,  # Link to ingest session
            source_doc_sha=source_doc_sha,
            concept_ids=concept_ids,
            session_id=session_id,
            metadata={
                'source_path': source_doc_path,
                'concept_count': len(concept_ids)
            }
        )
        
        self._append_event(event)
        
    def find_concept_origin(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Find when and where a concept was first learned"""
        # Stream through NDJSON files
        for archive_file in sorted(self.archive_dir.glob('*.ndjson*')):
            with self._open_archive(archive_file) as f:
                for line in f:
                    event = json.loads(line)
                    if concept_id in event.get('concept_ids', []):
                        return {
                            'first_seen': event['timestamp'],
                            'source_doc_sha': event['source_doc_sha'],
                            'source_path': event['metadata'].get('source_path'),
                            'parent_event': event['parent_id']
                        }
        return None
```

### 2. Session/Thread IDs for Debugging

```python
# Integration with Saigon response generation
class SaigonDebugLogger:
    """Track Saigon's decision path for debugging"""
    
    def log_response_generation(
        self,
        query: str,
        concept_path: List[str],
        response: str,
        session_id: str,
        parent_plan_id: Optional[str] = None
    ):
        """Log Saigon response with full context"""
        event = PsiEvent(
            event_id=self._generate_event_id(),
            event_type='RESPONSE_EVENT',
            timestamp=datetime.now(),
            parent_id=parent_plan_id,  # Link to planning phase
            concept_ids=concept_path,  # Concepts used in generation
            session_id=session_id,
            metadata={
                'query': query,
                'response_preview': response[:200],
                'path_length': len(concept_path),
                'model': 'saigon_lstm'
            }
        )
        
        PSI_ARCHIVER.append_event(event)
    
    def debug_hallucination(self, session_id: str, date: str):
        """Replay Saigon's path for a specific session"""
        events = []
        
        # Find all events for this session
        archive_file = self.archive_dir / f"{date}.ndjson.gz"
        with gzip.open(archive_file, 'rt') as f:
            for line in f:
                event = json.loads(line)
                if event.get('session_id') == session_id:
                    events.append(event)
        
        # Reconstruct the decision chain
        for event in sorted(events, key=lambda e: e['timestamp']):
            if event['event_type'] == 'RESPONSE_EVENT':
                print(f"[{event['timestamp']}] Saigon used concepts: {event['concept_ids']}")
                print(f"  Parent plan: {event['parent_id']}")
                print(f"  Query: {event['metadata']['query']}")
```

### 3. Incremental Δ-Dumps (Mesh Deltas)

```python
# Track mesh changes as deltas
class MeshDeltaTracker:
    """Track ConceptMesh changes for incremental sync"""
    
    def capture_mesh_delta(
        self,
        added_concepts: List[Concept],
        added_relations: List[ConceptRelation],
        modified_concepts: List[Concept] = None
    ) -> Dict[str, Any]:
        """Capture mesh changes as a delta object"""
        delta = {
            'timestamp': datetime.now().isoformat(),
            'added_nodes': [
                {
                    'id': c.id,
                    'name': c.name,
                    'embedding': c.embedding.tolist() if c.embedding is not None else None,
                    'metadata': c.metadata
                }
                for c in added_concepts
            ],
            'added_edges': [
                {
                    'source': r.source_id,
                    'target': r.target_id,
                    'relation_type': r.relation_type,
                    'strength': r.strength
                }
                for r in added_relations
            ],
            'modified_nodes': [
                {'id': c.id, 'changes': self._diff_concept(c)}
                for c in (modified_concepts or [])
            ]
        }
        
        return delta
    
    def apply_delta_to_mesh(self, mesh: ConceptMesh, delta: Dict[str, Any]):
        """Apply a delta to update a ConceptMesh"""
        # Add new concepts
        for node in delta['added_nodes']:
            mesh.add_concept(
                name=node['name'],
                embedding=np.array(node['embedding']) if node['embedding'] else None,
                metadata=node['metadata']
            )
        
        # Add new relations
        for edge in delta['added_edges']:
            mesh.add_relation(
                source_id=edge['source'],
                target_id=edge['target'],
                relation_type=edge['relation_type'],
                strength=edge['strength']
            )
```

### 4. Spectral Replay Utility

```python
#!/usr/bin/env python3
# tools/psi_replay.py

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

class PsiReplay:
    """Reconstruct system state at any point in time"""
    
    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        
    def replay_until(self, cutoff_time: datetime, output_dir: Path):
        """Replay all events up to cutoff time"""
        print(f"🔄 Replaying PsiArchive until {cutoff_time}")
        
        # Create temporary workspace
        temp_mesh = ConceptMesh({'storage_path': output_dir / 'mesh'})
        temp_vault = UnifiedMemoryVault({'storage_path': output_dir / 'vault'})
        
        events_processed = 0
        
        # Process all archive files in order
        for archive_file in sorted(self.archive_dir.glob('*.ndjson*')):
            # Extract date from filename
            file_date = self._parse_archive_date(archive_file)
            if file_date > cutoff_time.date():
                break
                
            print(f"📁 Processing {archive_file.name}")
            
            with self._open_archive(archive_file) as f:
                for line in f:
                    event = json.loads(line)
                    event_time = datetime.fromisoformat(event['timestamp'])
                    
                    if event_time > cutoff_time:
                        break
                    
                    # Apply event based on type
                    if event['event_type'] == 'LEARNING_UPDATE':
                        self._apply_learning_update(event, temp_mesh, temp_vault)
                    elif event['event_type'] == 'CONCEPT_REVISION':
                        self._apply_revision(event, temp_mesh)
                    
                    events_processed += 1
                    
                    if events_processed % 1000 == 0:
                        print(f"  Processed {events_processed} events...")
        
        # Save reconstructed state
        temp_mesh.save_mesh()
        temp_vault.save_all()
        
        print(f"✅ Replay complete: {events_processed} events processed")
        print(f"📊 Reconstructed state saved to {output_dir}")
        
        return {
            'events_processed': events_processed,
            'final_concept_count': len(temp_mesh.concepts),
            'final_memory_count': temp_vault.get_statistics()['total_memories']
        }
    
    def _apply_learning_update(self, event: Dict, mesh: ConceptMesh, vault: UnifiedMemoryVault):
        """Apply a learning update event"""
        if 'mesh_delta' in event and event['mesh_delta']:
            delta = event['mesh_delta']
            
            # Apply mesh delta
            for node in delta.get('added_nodes', []):
                mesh.concepts[node['id']] = Concept(**node)
            
            for edge in delta.get('added_edges', []):
                mesh.add_relation(
                    source_id=edge['source'],
                    target_id=edge['target'],
                    relation_type=edge['relation_type']
                )

def main():
    parser = argparse.ArgumentParser(description='PsiArchive Time Travel')
    parser.add_argument('--until', required=True, help='Replay until this timestamp (ISO format)')
    parser.add_argument('--archive-dir', default='data/archive', help='PsiArchive directory')
    parser.add_argument('--output-dir', default='data/replay_output', help='Output directory')
    
    args = parser.parse_args()
    
    cutoff = datetime.fromisoformat(args.until)
    replay = PsiReplay(Path(args.archive_dir))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = replay.replay_until(cutoff, output_dir)
    print(f"\n📈 Final statistics: {json.dumps(stats, indent=2)}")

if __name__ == '__main__':
    main()
```

### 5. Optional Hash-Chain for Tamper Evidence

```python
# tools/psi_chain.py
class PsiHashChain:
    """Add hash-chain headers to PsiArchive files"""
    
    def seal_daily_archive(self, archive_file: Path):
        """Add hash of previous day's archive as header"""
        # Get previous day's file
        prev_file = self._get_previous_archive(archive_file)
        
        if prev_file and prev_file.exists():
            # Calculate SHA-256 of previous file
            with open(prev_file, 'rb') as f:
                prev_sha = hashlib.sha256(f.read()).hexdigest()
            
            # Prepend hash header to today's file
            header = {
                'archive_version': '2.0',
                'prev_archive_sha': prev_sha,
                'prev_archive_name': prev_file.name,
                'sealed_at': datetime.now().isoformat()
            }
            
            # Read existing content
            with open(archive_file, 'r') as f:
                content = f.read()
            
            # Write header + content
            with open(archive_file, 'w') as f:
                f.write(json.dumps(header) + '\n')
                f.write('---ARCHIVE_START---\n')
                f.write(content)
    
    def verify_chain(self, start_date: str, end_date: str) -> bool:
        """Verify hash chain integrity"""
        current_date = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        while current_date <= end:
            archive_file = self.archive_dir / f"{current_date.strftime('%Y-%m-%d')}.ndjson"
            
            if not archive_file.exists():
                current_date += timedelta(days=1)
                continue
            
            # Read header
            with open(archive_file, 'r') as f:
                header_line = f.readline()
                if not header_line.startswith('{'):
                    print(f"⚠️ No header in {archive_file.name}")
                    return False
                
                header = json.loads(header_line)
                
                # Verify previous file hash
                if 'prev_archive_sha' in header:
                    prev_file = self.archive_dir / header['prev_archive_name']
                    if prev_file.exists():
                        with open(prev_file, 'rb') as pf:
                            actual_sha = hashlib.sha256(pf.read()).hexdigest()
                        
                        if actual_sha != header['prev_archive_sha']:
                            print(f"❌ Hash mismatch for {prev_file.name}")
                            return False
            
            current_date += timedelta(days=1)
        
        print("✅ Hash chain verified")
        return True
```

## 🎯 Integration Points

### Update `add_concept_diff` in ingestion pipeline:
```python
def add_concept_diff_extended(
    concept: Concept,
    source_doc_path: str,
    session_id: str,
    parent_event_id: Optional[str] = None
):
    """Enhanced concept diff with full provenance"""
    # Existing mesh update
    added_concepts, added_relations = mesh.add_concept_with_relations(concept)
    
    # Capture delta
    delta = MeshDeltaTracker().capture_mesh_delta(
        added_concepts=[concept],
        added_relations=added_relations
    )
    
    # Log with provenance
    PSI_ARCHIVER.log_concept_ingestion(
        concept_ids=[concept.id],
        source_doc_path=source_doc_path,
        parent_event_id=parent_event_id,
        session_id=session_id,
        mesh_delta=delta
    )
```

### Query Examples:
```bash
# Find origin of "graphene" concept
jq 'select(.concept_ids[]? | contains("graphene"))' data/archive/*.ndjson | head -1

# Debug yesterday's hallucination
python tools/psi_replay.py --until "2025-01-09T23:59:59" --output-dir debug_state
python debug_saigon.py --state-dir debug_state --session-id "chat_123"

# Sync laptop with minimal data
rsync -av --include="*.ndjson" --include="*delta*" server:/archive/ laptop:/archive/
```

## 🚀 Benefits Achieved

1. **Full Provenance**: Every concept traced to source PDF via SHA-256
2. **Deterministic Debugging**: Replay exact state at any timestamp
3. **Lightweight Sync**: Delta-only transfers (KB not GB)
4. **Tamper Evidence**: Optional hash-chain without blockchain
5. **Thread Tracking**: Debug conversation flows and planning chains

All implemented with **zero databases** - just enhanced NDJSON and file operations!
