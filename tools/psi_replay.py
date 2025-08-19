#!/usr/bin/env python3
"""
PsiArchive Spectral Replay - Reconstruct system state at any point in time
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.psi_archive_extended import PsiEvent, get_psi_archiver
from python.core.concept_mesh import ConceptMesh, Concept, ConceptRelation
from python.core.memory_vault import UnifiedMemoryVault, MemoryEntry, MemoryType
from penrose_projector.core import PenroseProjector


class PsiReplay:
    """Reconstruct system state at any point in time"""
    
    def __init__(self, archive_dir: Path, snapshots_dir: Optional[Path] = None):
        self.archive_dir = archive_dir
        self.snapshots_dir = snapshots_dir or Path("data/snapshots")
        self.archiver = get_psi_archiver(str(archive_dir))
        
    def find_latest_snapshot(self, before_time: datetime) -> Optional[Path]:
        """Find the most recent snapshot before the given time"""
        if not self.snapshots_dir.exists():
            return None
            
        snapshots = []
        for snapshot_dir in self.snapshots_dir.glob("????-??-??"):
            try:
                snapshot_date = datetime.strptime(snapshot_dir.name, "%Y-%m-%d")
                if snapshot_date < before_time:
                    snapshots.append((snapshot_date, snapshot_dir))
            except ValueError:
                continue
        
        if not snapshots:
            return None
            
        # Return the most recent snapshot
        snapshots.sort(key=lambda x: x[0], reverse=True)
        return snapshots[0][1]
    
    def replay_fast(self, cutoff_time: datetime, output_dir: Path) -> Dict[str, Any]:
        """Fast replay using snapshots + forward deltas"""
        print(f"⚡ Fast replay mode enabled")
        
        # Find latest snapshot before cutoff
        snapshot_dir = self.find_latest_snapshot(cutoff_time)
        
        if snapshot_dir:
            snapshot_time = datetime.strptime(snapshot_dir.name, "%Y-%m-%d")
            print(f"📸 Found snapshot from {snapshot_dir.name}")
            print(f"🔄 Replaying {(cutoff_time - snapshot_time).days} days of deltas")
            
            # Copy snapshot as starting point
            print(f"📂 Copying snapshot to {output_dir}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(snapshot_dir, output_dir)
            
            # Load state from snapshot
            mesh = ConceptMesh({'storage_path': str(output_dir / 'concept_mesh')})
            vault = UnifiedMemoryVault({'storage_path': str(output_dir / 'memory_vault')})
            
            # Get deltas since snapshot
            deltas = self.archiver.get_mesh_deltas(snapshot_time)
            
            events_processed = 0
            for delta_info in deltas:
                if datetime.fromisoformat(delta_info['timestamp']) > cutoff_time:
                    break
                    
                # Apply delta
                if delta_info['delta']:
                    self._apply_mesh_delta(delta_info['delta'], mesh)
                    events_processed += 1
                    
                    if events_processed % 100 == 0:
                        print(f"  ⚡ Applied {events_processed} deltas...")
            
            print(f"✅ Fast replay complete: {events_processed} deltas applied")
            
            # Save updated state
            mesh._save_mesh()
            vault.save_all()
            
            # Generate summary
            mesh_stats = mesh.get_statistics()
            vault_stats = vault.get_statistics()
            
            return {
                'replay_mode': 'fast',
                'snapshot_used': str(snapshot_dir),
                'snapshot_date': snapshot_time.isoformat(),
                'deltas_applied': events_processed,
                'replay_cutoff': cutoff_time.isoformat(),
                'final_concept_count': mesh_stats['total_concepts'],
                'final_memory_count': vault_stats['total_memories'],
                'output_directory': str(output_dir)
            }
        else:
            print(f"⚠️ No snapshot found, falling back to full replay")
            return self.replay_until(cutoff_time, output_dir)
        
    def replay_until(self, cutoff_time: datetime, output_dir: Path, fast_mode: bool = False) -> Dict[str, Any]:
        """Replay all events up to cutoff time"""
        if fast_mode:
            return self.replay_fast(cutoff_time, output_dir)
            
        print(f"🔄 Replaying PsiArchive until {cutoff_time}")
        print(f"📁 Archive directory: {self.archive_dir}")
        print(f"📂 Output directory: {output_dir}")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir = output_dir / 'concept_mesh'
        vault_dir = output_dir / 'memory_vault'
        
        # Initialize fresh components
        mesh = ConceptMesh({'storage_path': str(mesh_dir)})
        vault = UnifiedMemoryVault({'storage_path': str(vault_dir)})
        
        events_processed = 0
        concepts_added = 0
        memories_added = 0
        
        # Process all events up to cutoff
        for archive_file in self.archiver._iter_archive_files():
            print(f"\n📄 Processing {archive_file.name}")
            
            file_events = 0
            for event in self.archiver._read_archive_file(archive_file):
                if event.timestamp > cutoff_time:
                    print(f"  ⏸️  Reached cutoff time at event {event.event_id}")
                    break
                
                # Apply event based on type
                if event.event_type == 'LEARNING_UPDATE':
                    added = self._apply_learning_update(event, mesh, vault)
                    concepts_added += added
                    
                elif event.event_type == 'CONCEPT_REVISION':
                    self._apply_revision(event, mesh)
                    
                elif event.event_type == 'MEMORY_STORE':
                    if self._apply_memory_store(event, vault):
                        memories_added += 1
                
                events_processed += 1
                file_events += 1
                
                if file_events % 100 == 0:
                    print(f"  ⚡ Processed {file_events} events from this file...")
            
            print(f"  ✅ Processed {file_events} events from {archive_file.name}")
            
            # Stop if we've passed cutoff
            if event.timestamp > cutoff_time:
                break
        
        # Save reconstructed state
        print(f"\n💾 Saving reconstructed state...")
        mesh._save_mesh()
        vault.save_all()
        
        # Generate summary
        mesh_stats = mesh.get_statistics()
        vault_stats = vault.get_statistics()
        
        summary = {
            'replay_mode': 'full',
            'replay_cutoff': cutoff_time.isoformat(),
            'events_processed': events_processed,
            'concepts_added': concepts_added,
            'memories_added': memories_added,
            'final_concept_count': mesh_stats['total_concepts'],
            'final_memory_count': vault_stats['total_memories'],
            'concept_categories': mesh_stats['category_distribution'],
            'memory_types': vault_stats['type_distribution'],
            'output_directory': str(output_dir)
        }
        
        # Save summary
        with open(output_dir / 'replay_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Replay complete!")
        print(f"📊 Events processed: {events_processed}")
        print(f"🧠 Concepts reconstructed: {mesh_stats['total_concepts']}")
        print(f"💾 Memories reconstructed: {vault_stats['total_memories']}")
        
        return summary
    
    def _apply_mesh_delta(self, delta: Dict[str, Any], mesh: ConceptMesh):
        """Apply a mesh delta directly"""
        # Add concepts
        for node in delta.get('added_nodes', []):
            try:
                mesh.add_concept(
                    name=node['name'],
                    description=node.get('description', ''),
                    category=node.get('category', 'general'),
                    importance=node.get('importance', 1.0),
                    metadata=node.get('metadata', {})
                )
            except Exception as e:
                print(f"  ⚠️  Failed to add concept from delta: {e}")
        
        # Add relations
        for edge in delta.get('added_edges', []):
            try:
                mesh.add_relation(
                    source_id=edge['source'],
                    target_id=edge['target'],
                    relation_type=edge['relation_type'],
                    strength=edge.get('strength', 1.0),
                    metadata=edge.get('metadata', {})
                )
            except Exception as e:
                print(f"  ⚠️  Failed to add relation from delta: {e}")
    
    def _apply_learning_update(self, event: PsiEvent, mesh: ConceptMesh, vault: UnifiedMemoryVault) -> int:
        """Apply a learning update event"""
        concepts_added = 0
        
        # Apply mesh delta if present
        if event.mesh_delta:
            delta = event.mesh_delta
            
            # Add concepts
            for node in delta.get('added_nodes', []):
                try:
                    # Create concept from delta
                    concept_id = mesh.add_concept(
                        name=node['name'],
                        description=node.get('description', ''),
                        category=node.get('category', 'general'),
                        importance=node.get('importance', 1.0),
                        metadata=node.get('metadata', {})
                    )
                    
                    # Store ID mapping
                    if node['id'] != concept_id:
                        # Map old ID to new ID if different
                        mesh.name_index[node['id']] = concept_id
                    
                    concepts_added += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to add concept {node.get('name', 'unknown')}: {e}")
            
            # Add relations
            for edge in delta.get('added_edges', []):
                try:
                    mesh.add_relation(
                        source_id=edge['source'],
                        target_id=edge['target'],
                        relation_type=edge['relation_type'],
                        strength=edge.get('strength', 1.0),
                        metadata=edge.get('metadata', {})
                    )
                except Exception as e:
                    print(f"  ⚠️  Failed to add relation: {e}")
        
        # Apply Penrose relations if present
        if event.penrose_stats and 'csr_file' in event.penrose_stats:
            try:
                csr_path = Path(event.penrose_stats['csr_file'])
                if csr_path.exists():
                    print(f"  🎯 Applying Penrose relations from {csr_path.name}")
                    
                    # Load the sparse matrix
                    projector = PenroseProjector()
                    sparse_matrix = projector.load_sparse_compressed(csr_path)
                    
                    # Apply relations if mesh supports it
                    if hasattr(mesh, 'add_relations_from_penrose'):
                        relation_stats = mesh.add_relations_from_penrose(
                            sparse_matrix=sparse_matrix,
                            concept_ids=event.concept_ids
                        )
                        print(f"     Added {relation_stats['relations_added']} Penrose relations")
                    else:
                        # Fallback: manually add relations
                        rows, cols = sparse_matrix.nonzero()
                        added = 0
                        for idx in range(len(rows)):
                            i, j = rows[idx], cols[idx]
                            if i >= j:  # Skip lower triangle
                                continue
                            
                            weight = sparse_matrix[i, j]
                            if weight >= 0.7:  # Default threshold
                                source_id = event.concept_ids[i]
                                target_id = event.concept_ids[j]
                                
                                if source_id in mesh.concepts and target_id in mesh.concepts:
                                    mesh.add_relation(
                                        source_id=source_id,
                                        target_id=target_id,
                                        relation_type='similar_penrose',
                                        strength=float(weight),
                                        metadata={'penrose_similarity': float(weight)}
                                    )
                                    added += 1
                        
                        if added > 0:
                            print(f"     Added {added} Penrose relations (fallback mode)")
                else:
                    print(f"  ⚠️  Penrose CSR file not found: {csr_path}")
                    
            except Exception as e:
                print(f"  ⚠️  Failed to apply Penrose relations: {e}")
        
        # Store concept IDs in metadata for reference
        if event.concept_ids and event.source_doc_sha:
            for concept_id in event.concept_ids:
                if concept_id in mesh.concepts:
                    concept = mesh.concepts[concept_id]
                    concept.metadata['source_doc_sha'] = event.source_doc_sha
                    concept.metadata['origin_event'] = event.event_id
        
        return concepts_added
    
    def _apply_revision(self, event: PsiEvent, mesh: ConceptMesh):
        """Apply a concept revision event"""
        if not event.concept_ids:
            return
        
        concept_id = event.concept_ids[0]  # Usually revisions affect one concept
        
        if concept_id in mesh.concepts:
            concept = mesh.concepts[concept_id]
            
            # Apply revision from metadata
            if 'new_name' in event.metadata:
                concept.name = event.metadata['new_name']
            
            if 'new_description' in event.metadata:
                concept.description = event.metadata['new_description']
            
            if 'new_importance' in event.metadata:
                concept.importance = event.metadata['new_importance']
            
            # Update revision history
            if 'revision_history' not in concept.metadata:
                concept.metadata['revision_history'] = []
            
            concept.metadata['revision_history'].append({
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'changes': event.metadata
            })
    
    def _apply_memory_store(self, event: PsiEvent, vault: UnifiedMemoryVault) -> bool:
        """Apply a memory store event"""
        if 'memory_data' not in event.metadata:
            return False
        
        memory_data = event.metadata['memory_data']
        
        try:
            # Store memory
            memory_id = vault.store(
                content=memory_data.get('content', ''),
                memory_type=memory_data.get('type', 'semantic'),
                metadata=memory_data.get('metadata', {}),
                importance=memory_data.get('importance', 1.0),
                tags=memory_data.get('tags', [])
            )
            
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to store memory: {e}")
            return False
    
    def find_divergence(self, state_dir1: Path, state_dir2: Path) -> Dict[str, Any]:
        """Compare two replay states to find where they diverge"""
        print(f"\n🔍 Comparing states:")
        print(f"  State 1: {state_dir1}")
        print(f"  State 2: {state_dir2}")
        
        # Load both meshes
        mesh1 = ConceptMesh({'storage_path': str(state_dir1 / 'concept_mesh')})
        mesh2 = ConceptMesh({'storage_path': str(state_dir2 / 'concept_mesh')})
        
        # Compare concepts
        concepts1 = set(mesh1.concepts.keys())
        concepts2 = set(mesh2.concepts.keys())
        
        only_in_1 = concepts1 - concepts2
        only_in_2 = concepts2 - concepts1
        
        divergence = {
            'concepts_only_in_state1': list(only_in_1),
            'concepts_only_in_state2': list(only_in_2),
            'concept_count_diff': len(concepts1) - len(concepts2)
        }
        
        # Compare edges
        edges1 = set()
        edges2 = set()
        
        for relation in mesh1.relations:
            edges1.add((relation.source_id, relation.target_id, relation.relation_type))
        
        for relation in mesh2.relations:
            edges2.add((relation.source_id, relation.target_id, relation.relation_type))
        
        edges_only_in_1 = edges1 - edges2
        edges_only_in_2 = edges2 - edges1
        
        divergence['edges_only_in_state1'] = len(edges_only_in_1)
        divergence['edges_only_in_state2'] = len(edges_only_in_2)
        
        # Find first diverging concept
        for concept_id in sorted(concepts1 & concepts2):
            c1 = mesh1.concepts[concept_id]
            c2 = mesh2.concepts[concept_id]
            
            if c1.name != c2.name or c1.description != c2.description:
                divergence['first_diverging_concept'] = {
                    'id': concept_id,
                    'name_in_state1': c1.name,
                    'name_in_state2': c2.name,
                    'origin_event_1': c1.metadata.get('origin_event'),
                    'origin_event_2': c2.metadata.get('origin_event')
                }
                break
        
        return divergence


def main():
    parser = argparse.ArgumentParser(
        description='PsiArchive Time Travel - Reconstruct system state at any point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay until specific timestamp
  psi_replay --until "2025-01-09T12:00:00"
  
  # Fast replay using snapshots
  psi_replay --until "2025-01-09T12:00:00" --fast
  
  # Replay until midnight yesterday
  psi_replay --until "2025-01-09T00:00:00" --output-dir yesterday_state
  
  # Create snapshot for today
  psi_replay --until "$(date +%FT00:00:00)" --output-dir snapshots/$(date +%F)
  
  # Compare two states
  psi_replay --compare state1/ state2/
        """
    )
    
    parser.add_argument(
        '--until',
        help='Replay until this timestamp (ISO format)'
    )
    parser.add_argument(
        '--archive-dir',
        default='data/archive',
        help='PsiArchive directory (default: data/archive)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/replay_output',
        help='Output directory for reconstructed state'
    )
    parser.add_argument(
        '--snapshots-dir',
        default='data/snapshots',
        help='Snapshots directory for fast mode (default: data/snapshots)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: use latest snapshot + forward deltas'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('STATE1', 'STATE2'),
        help='Compare two replay states'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        replay = PsiReplay(Path(args.archive_dir), Path(args.snapshots_dir))
        divergence = replay.find_divergence(Path(args.compare[0]), Path(args.compare[1]))
        
        print("\n📊 Divergence Analysis:")
        print(json.dumps(divergence, indent=2))
        
    elif args.until:
        # Replay mode
        cutoff = datetime.fromisoformat(args.until)
        replay = PsiReplay(Path(args.archive_dir), Path(args.snapshots_dir))
        
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            print(f"⚠️  Output directory {output_dir} already exists!")
            response = input("Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            shutil.rmtree(output_dir)
        
        summary = replay.replay_until(cutoff, output_dir, fast_mode=args.fast)
        
        print(f"\n📁 State reconstructed in: {output_dir}")
        print(f"📄 Summary saved to: {output_dir}/replay_summary.json")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
