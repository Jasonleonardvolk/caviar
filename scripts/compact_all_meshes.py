"""
TORI Concept Mesh Compaction Script

PURPOSE:
    Implements Phase 5: Nightly Snapshot and Delta-Compaction Routines.
    Snapshots all concept meshes and compacts their WALs to optimize storage
    and maintain system performance.

WHAT IT DOES:
    1. Scans all user and group concept meshes
    2. Creates compressed snapshots with checksums
    3. Performs mesh compaction (dedupe, prune, optimize)
    4. Manages WAL checkpoints and cleanup
    5. Creates daily ricci24h checkpoints
    6. Removes old snapshots based on retention policy
    7. Provides disaster recovery restore functionality

USAGE:
    python compact_all_meshes.py [--force] [--config CONFIG_FILE]
    python compact_all_meshes.py --restore user:alice snapshot.json.gz

FEATURES:
    - Automatic compaction based on size and time thresholds
    - Compressed snapshots with integrity verification
    - Multi-tenant support (user and group scopes)
    - Configurable retention policies
    - Detailed progress reporting and metrics
    - Safe restore from snapshots

AUTHOR: TORI System Maintenance
LAST UPDATED: 2025-01-26
"""

import logging
import json
import gzip
import shutil
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import os
import sys

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our components with fallbacks
try:
    from python.core.concept_mesh import ConceptMesh
    MESH_AVAILABLE = True
    MESH_TYPE = "ConceptMesh"
except ImportError:
    try:
        # Try the scoped version if available
        from python.core.scoped_concept_mesh import ScopedConceptMesh
        # Use ScopedConceptMesh as ConceptMesh for compatibility
        ConceptMesh = ScopedConceptMesh
        MESH_AVAILABLE = True
        MESH_TYPE = "ScopedConceptMesh"
    except ImportError:
        logging.warning("⚠️ Neither ConceptMesh nor ScopedConceptMesh available - using mock")
        MESH_AVAILABLE = False
        MESH_TYPE = "Mock"
        
        # Create a mock class for testing
        class ConceptMesh:
            def __init__(self, *args, **kwargs):
                self.concepts = {}
                self.relations = []
                self.config = kwargs.get('config', {})
            
            def _save_mesh(self):
                pass
            
            @classmethod
            def get_instance(cls, *args, **kwargs):
                return cls(*args, **kwargs)

# Try to import WAL manager
try:
    from python.core.scoped_wal import WALManager
    WAL_AVAILABLE = True
except ImportError:
    WAL_AVAILABLE = False
    logging.warning("⚠️ WAL manager not available")
    
    # Mock WAL manager
    class WALManager:
        @staticmethod
        def get_wal(scope: str, scope_id: str):
            class MockWAL:
                def get_stats(self):
                    return {'wal_size_bytes': 0, 'checkpoint_size_bytes': 0}
                def _create_checkpoint(self):
                    pass
            return MockWAL()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/compaction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MeshCompactor:
    """
    Handles snapshot generation and delta compaction for all meshes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Paths
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.snapshot_dir = self.data_dir / 'snapshots'
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Compaction settings
        self.compress_snapshots = self.config.get('compress_snapshots', True)
        self.keep_snapshots_days = self.config.get('keep_snapshots_days', 30)
        self.max_wal_size_mb = self.config.get('max_wal_size_mb', 50)
        self.min_compaction_interval_hours = self.config.get('min_compaction_interval_hours', 12)
        
        # Checkpoint tagging
        self.ricci_tag = self.config.get('ricci_tag', 'ricci24h')
        
        # Create logs directory if needed
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"MeshCompactor initialized with {MESH_TYPE} backend")
        logger.info(f"Snapshot directory: {self.snapshot_dir}")
        logger.info(f"WAL support: {'enabled' if WAL_AVAILABLE else 'disabled'}")
    
    async def compact_all_meshes(self, force: bool = False) -> Dict[str, Any]:
        """
        Main entry point - compact all user and group meshes
        
        Args:
            force: Force compaction even if not needed
            
        Returns:
            Summary of compaction results
        """
        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'meshes_processed': 0,
            'meshes_compacted': 0,
            'snapshots_created': 0,
            'errors': [],
            'details': {}
        }
        
        try:
            # Process user meshes
            user_results = await self._compact_scope_meshes('user', force)
            results['details']['users'] = user_results
            
            # Process group meshes
            group_results = await self._compact_scope_meshes('group', force)
            results['details']['groups'] = group_results
            
            # Create daily checkpoint
            if self._should_create_daily_checkpoint():
                checkpoint_path = await self._create_daily_checkpoint()
                results['daily_checkpoint'] = str(checkpoint_path)
            
            # Clean old snapshots
            cleaned = self._clean_old_snapshots()
            results['snapshots_cleaned'] = cleaned
            
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            results['errors'].append(str(e))
        
        # Summary
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Count totals
        for scope_results in results['details'].values():
            results['meshes_processed'] += scope_results['total']
            results['meshes_compacted'] += scope_results['compacted']
            results['snapshots_created'] += scope_results['snapshots']
        
        logger.info(f"Compaction complete: {results['meshes_compacted']}/{results['meshes_processed']} meshes compacted")
        
        return results
    
    async def _compact_scope_meshes(self, scope: str, force: bool) -> Dict[str, Any]:
        """
        Compact all meshes for a given scope (user or group)
        """
        scope_dir = self.data_dir / 'concept_mesh' / scope
        results = {
            'scope': scope,
            'total': 0,
            'compacted': 0,
            'snapshots': 0,
            'skipped': 0,
            'errors': []
        }
        
        if not scope_dir.exists():
            logger.info(f"No {scope} meshes found")
            return results
        
        # Process each mesh
        for mesh_dir in scope_dir.iterdir():
            if not mesh_dir.is_dir():
                continue
                
            scope_id = mesh_dir.name
            results['total'] += 1
            
            try:
                # Check if compaction needed
                if not force and not self._needs_compaction(scope, scope_id):
                    logger.info(f"Skipping {scope}:{scope_id} - compaction not needed")
                    results['skipped'] += 1
                    continue
                
                # Perform compaction
                compacted = await self._compact_single_mesh(scope, scope_id)
                if compacted:
                    results['compacted'] += 1
                    results['snapshots'] += 1
                
            except Exception as e:
                logger.error(f"Failed to compact {scope}:{scope_id}: {e}")
                results['errors'].append(f"{scope_id}: {str(e)}")
        
        return results
    
    async def _compact_single_mesh(self, scope: str, scope_id: str) -> bool:
        """
        Compact a single mesh - snapshot and clean WAL
        """
        logger.info(f"Compacting {scope}:{scope_id}")
        
        try:
            # Get mesh instance - handle both ConceptMesh types
            if MESH_TYPE == "ScopedConceptMesh":
                mesh = ConceptMesh.get_instance(scope, scope_id)
            else:
                # Standard ConceptMesh
                mesh = ConceptMesh.instance()
            
            # Get WAL instance if available
            if WAL_AVAILABLE:
                wal = WALManager.get_wal(scope, scope_id)
            else:
                wal = None
            
            # Create snapshot
            snapshot_path = await self._create_snapshot(scope, scope_id, mesh)
            
            # Compact mesh data
            original_size = self._get_mesh_size(mesh)
            self._deduplicate_concepts(mesh)
            self._prune_old_diffs(mesh)
            self._optimize_relations(mesh)
            
            # Save compacted mesh
            mesh._save_mesh()
            new_size = self._get_mesh_size(mesh)
            
            # Create WAL checkpoint if available
            if wal:
                wal._create_checkpoint()
            
            # Log results
            reduction_pct = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
            logger.info(f"Compacted {scope}:{scope_id} - Size reduced by {reduction_pct:.1f}% "
                       f"({original_size} -> {new_size} bytes)")
            
            # Update metrics
            self._update_compaction_metrics(scope, scope_id, original_size, new_size)
            
            return True
            
        except Exception as e:
            logger.error(f"Compaction failed for {scope}:{scope_id}: {e}")
            raise
    
    async def _create_snapshot(self, scope: str, scope_id: str, mesh) -> Path:
        """
        Create a snapshot of the mesh state
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mesh_{scope}_{scope_id}_{timestamp}.json"
        
        if self.compress_snapshots:
            filename += '.gz'
        
        snapshot_path = self.snapshot_dir / scope / scope_id / filename
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare snapshot data
        snapshot_data = {
            'version': '1.0',
            'scope': scope,
            'scope_id': scope_id,
            'timestamp': datetime.now().isoformat(),
            'mesh_data': {
                'concepts': {
                    cid: self._serialize_concept(c) 
                    for cid, c in mesh.concepts.items()
                },
                'relations': [
                    self._serialize_relation(r)
                    for r in mesh.relations
                ],
                'metadata': {
                    'total_concepts': len(mesh.concepts),
                    'total_relations': len(mesh.relations),
                    'categories': self._get_category_counts(mesh)
                }
            },
            'checksum': None
        }
        
        # Calculate checksum
        content_str = json.dumps(snapshot_data['mesh_data'], sort_keys=True)
        snapshot_data['checksum'] = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Write snapshot
        if self.compress_snapshots:
            with gzip.open(snapshot_path, 'wt', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2)
        else:
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Created snapshot: {snapshot_path}")
        return snapshot_path
    
    def _serialize_concept(self, concept) -> Dict[str, Any]:
        """Serialize a concept for snapshot"""
        return {
            'id': concept.id,
            'name': concept.name,
            'description': concept.description,
            'category': concept.category,
            'importance': concept.importance,
            'metadata': concept.metadata,
            'created_at': concept.created_at.isoformat(),
            'last_accessed': concept.last_accessed.isoformat(),
            'access_count': concept.access_count,
            'embedding_shape': concept.embedding.shape if concept.embedding is not None else None
        }
    
    def _serialize_relation(self, relation) -> Dict[str, Any]:
        """Serialize a relation for snapshot"""
        return {
            'source_id': relation.source_id,
            'target_id': relation.target_id,
            'relation_type': relation.relation_type,
            'strength': relation.strength,
            'bidirectional': relation.bidirectional,
            'metadata': relation.metadata,
            'created_at': relation.created_at.isoformat()
        }
    
    def _deduplicate_concepts(self, mesh) -> int:
        """Remove duplicate concepts from mesh"""
        seen_names = {}
        duplicates = []
        
        for concept_id, concept in mesh.concepts.items():
            name_key = concept.name.lower().strip()
            
            if name_key in seen_names:
                # Found duplicate
                existing_id = seen_names[name_key]
                existing = mesh.concepts[existing_id]
                
                # Keep the one with higher importance or more access
                if (concept.importance > existing.importance or 
                    concept.access_count > existing.access_count):
                    # Replace existing
                    duplicates.append(existing_id)
                    seen_names[name_key] = concept_id
                else:
                    # Remove this one
                    duplicates.append(concept_id)
            else:
                seen_names[name_key] = concept_id
        
        # Remove duplicates
        for dup_id in duplicates:
            if dup_id in mesh.concepts:
                del mesh.concepts[dup_id]
                # Also remove from indices
                concept = mesh.concepts.get(dup_id)
                if concept and concept.name in mesh.name_index:
                    del mesh.name_index[concept.name]
        
        if duplicates:
            logger.info(f"Removed {len(duplicates)} duplicate concepts")
        
        return len(duplicates)
    
    def _prune_old_diffs(self, mesh) -> int:
        """Prune old diffs beyond max_diff_history"""
        if not hasattr(mesh, 'diff_history'):
            return 0
            
        original_count = len(mesh.diff_history)
        max_diffs = mesh.config.get('max_diff_history', 1000)
        
        if original_count > max_diffs:
            # Keep only the most recent diffs
            mesh.diff_history = list(mesh.diff_history)[-max_diffs:]
            pruned = original_count - len(mesh.diff_history)
            logger.info(f"Pruned {pruned} old diffs")
            return pruned
        
        return 0
    
    def _optimize_relations(self, mesh) -> int:
        """Optimize relations - remove orphaned ones"""
        valid_concepts = set(mesh.concepts.keys())
        original_count = len(mesh.relations)
        
        # Filter valid relations
        mesh.relations = [
            r for r in mesh.relations
            if r.source_id in valid_concepts and r.target_id in valid_concepts
        ]
        
        removed = original_count - len(mesh.relations)
        if removed > 0:
            logger.info(f"Removed {removed} orphaned relations")
        
        return removed
    
    def _get_mesh_size(self, mesh) -> int:
        """Estimate mesh size in bytes"""
        # Simple estimation based on concept and relation count
        concept_size = len(mesh.concepts) * 500  # Rough estimate per concept
        relation_size = len(mesh.relations) * 100  # Rough estimate per relation
        return concept_size + relation_size
    
    def _get_category_counts(self, mesh) -> Dict[str, int]:
        """Get concept counts by category"""
        counts = {}
        for concept in mesh.concepts.values():
            category = concept.category or 'uncategorized'
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _needs_compaction(self, scope: str, scope_id: str) -> bool:
        """Check if mesh needs compaction"""
        try:
            # Check WAL size if available
            if WAL_AVAILABLE:
                wal = WALManager.get_wal(scope, scope_id)
                wal_stats = wal.get_stats()
                wal_size_mb = (wal_stats['wal_size_bytes'] + wal_stats['checkpoint_size_bytes']) / (1024 * 1024)
                
                if wal_size_mb > self.max_wal_size_mb:
                    logger.info(f"{scope}:{scope_id} needs compaction - WAL size {wal_size_mb:.1f}MB")
                    return True
            
            # Check last compaction time
            last_compact_file = self.snapshot_dir / scope / scope_id / '.last_compact'
            if last_compact_file.exists():
                last_compact_time = datetime.fromtimestamp(last_compact_file.stat().st_mtime)
                hours_since = (datetime.now() - last_compact_time).total_seconds() / 3600
                
                if hours_since < self.min_compaction_interval_hours:
                    return False
            
            # Check mesh file modification time
            mesh_path = self.data_dir / 'concept_mesh' / scope / scope_id
            if mesh_path.exists():
                oldest_file_time = min(
                    f.stat().st_mtime for f in mesh_path.iterdir() 
                    if f.is_file()
                )
                hours_since_modify = (datetime.now().timestamp() - oldest_file_time) / 3600
                
                if hours_since_modify > self.min_compaction_interval_hours:
                    logger.info(f"{scope}:{scope_id} needs compaction - {hours_since_modify:.1f}h since last modify")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking compaction need for {scope}:{scope_id}: {e}")
            return True  # Compact on error to be safe
    
    def _should_create_daily_checkpoint(self) -> bool:
        """Check if we should create the daily ricci24h checkpoint"""
        checkpoint_link = self.snapshot_dir / f"{self.ricci_tag}.latest"
        
        if not checkpoint_link.exists():
            return True
        
        # Check age of existing checkpoint
        checkpoint_age_hours = (datetime.now().timestamp() - checkpoint_link.stat().st_mtime) / 3600
        return checkpoint_age_hours >= 23  # Create new one after 23 hours
    
    async def _create_daily_checkpoint(self) -> Path:
        """Create the daily checkpoint with ricci24h tag"""
        timestamp = datetime.now().strftime('%Y%m%d')
        checkpoint_dir = self.snapshot_dir / self.ricci_tag / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifest
        manifest = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'tag': self.ricci_tag,
            'snapshots': []
        }
        
        # Copy latest snapshots to checkpoint
        for scope in ['user', 'group']:
            scope_dir = self.snapshot_dir / scope
            if not scope_dir.exists():
                continue
                
            for scope_id_dir in scope_dir.iterdir():
                if not scope_id_dir.is_dir():
                    continue
                    
                # Find latest snapshot
                snapshots = sorted(scope_id_dir.glob('mesh_*.json*'), 
                                 key=lambda p: p.stat().st_mtime, 
                                 reverse=True)
                
                if snapshots:
                    latest = snapshots[0]
                    dest = checkpoint_dir / scope / scope_id_dir.name / latest.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(latest, dest)
                    manifest['snapshots'].append({
                        'scope': scope,
                        'scope_id': scope_id_dir.name,
                        'file': latest.name,
                        'size': latest.stat().st_size
                    })
        
        # Write manifest
        manifest_path = checkpoint_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Update latest link
        latest_link = self.snapshot_dir / f"{self.ricci_tag}.latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_dir)
        
        logger.info(f"Created daily checkpoint: {checkpoint_dir}")
        return checkpoint_dir
    
    def _clean_old_snapshots(self) -> int:
        """Remove snapshots older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.keep_snapshots_days)
        removed = 0
        
        for snapshot_file in self.snapshot_dir.rglob('mesh_*.json*'):
            if snapshot_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    snapshot_file.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old snapshot {snapshot_file}: {e}")
        
        if removed > 0:
            logger.info(f"Cleaned {removed} old snapshots")
        
        return removed
    
    def _update_compaction_metrics(self, scope: str, scope_id: str, 
                                  original_size: int, new_size: int):
        """Update compaction metrics/timestamp"""
        # Update last compact timestamp
        last_compact_file = self.snapshot_dir / scope / scope_id / '.last_compact'
        last_compact_file.parent.mkdir(parents=True, exist_ok=True)
        last_compact_file.touch()
        
        # Could also update Prometheus metrics here if integrated
        pass
    
    async def restore_from_snapshot(self, scope: str, scope_id: str, 
                                  snapshot_path: Optional[Path] = None) -> bool:
        """
        Restore a mesh from snapshot (for disaster recovery)
        
        Args:
            scope: "user" or "group"
            scope_id: ID of the mesh
            snapshot_path: Specific snapshot to restore, or None for latest
            
        Returns:
            True if successful
        """
        try:
            # Find snapshot to restore
            if snapshot_path is None:
                # Find latest snapshot
                scope_dir = self.snapshot_dir / scope / scope_id
                if not scope_dir.exists():
                    logger.error(f"No snapshots found for {scope}:{scope_id}")
                    return False
                
                snapshots = sorted(scope_dir.glob('mesh_*.json*'),
                                 key=lambda p: p.stat().st_mtime,
                                 reverse=True)
                
                if not snapshots:
                    logger.error(f"No snapshots found for {scope}:{scope_id}")
                    return False
                
                snapshot_path = snapshots[0]
            
            logger.info(f"Restoring {scope}:{scope_id} from {snapshot_path}")
            
            # Load snapshot
            if snapshot_path.suffix == '.gz':
                with gzip.open(snapshot_path, 'rt', encoding='utf-8') as f:
                    snapshot_data = json.load(f)
            else:
                with open(snapshot_path, 'r', encoding='utf-8') as f:
                    snapshot_data = json.load(f)
            
            # Verify checksum
            mesh_data = snapshot_data['mesh_data']
            content_str = json.dumps(mesh_data, sort_keys=True)
            calculated_checksum = hashlib.sha256(content_str.encode()).hexdigest()
            
            if calculated_checksum != snapshot_data['checksum']:
                logger.error("Snapshot checksum mismatch - data may be corrupted")
                return False
            
            # Clear existing mesh data
            if MESH_TYPE == "ScopedConceptMesh":
                mesh = ConceptMesh.get_instance(scope, scope_id)
            else:
                mesh = ConceptMesh.instance()
                
            mesh.concepts.clear()
            mesh.relations.clear()
            
            # Restore concepts
            for concept_data in mesh_data['concepts'].values():
                # Recreate concept (simplified - would need full reconstruction in production)
                mesh.add_concept(
                    name=concept_data['name'],
                    description=concept_data['description'],
                    category=concept_data['category'],
                    importance=concept_data['importance'],
                    metadata=concept_data['metadata']
                )
            
            # Restore relations
            for relation_data in mesh_data['relations']:
                mesh.add_relation(
                    source_id=relation_data['source_id'],
                    target_id=relation_data['target_id'],
                    relation_type=relation_data['relation_type'],
                    strength=relation_data['strength'],
                    bidirectional=relation_data['bidirectional'],
                    metadata=relation_data['metadata']
                )
            
            # Save restored mesh
            mesh._save_mesh()
            
            logger.info(f"Successfully restored {scope}:{scope_id} with "
                       f"{len(mesh.concepts)} concepts and {len(mesh.relations)} relations")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore {scope}:{scope_id}: {e}")
            return False


async def main():
    """Main entry point for script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compact TORI concept meshes')
    parser.add_argument('--force', action='store_true', help='Force compaction even if not needed')
    parser.add_argument('--restore', nargs=2, metavar=('SCOPE:ID', 'SNAPSHOT'),
                       help='Restore a mesh from snapshot')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    compactor = MeshCompactor(config)
    
    if args.restore:
        # Restore mode
        scope_id_str, snapshot_path = args.restore
        scope, scope_id = scope_id_str.split(':')
        success = await compactor.restore_from_snapshot(scope, scope_id, Path(snapshot_path))
        return 0 if success else 1
    else:
        # Compaction mode
        results = await compactor.compact_all_meshes(force=args.force)
        
        # Save results
        results_file = Path('logs') / f"compaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nCompaction Summary:")
        print(f"  Meshes processed: {results['meshes_processed']}")
        print(f"  Meshes compacted: {results['meshes_compacted']}")
        print(f"  Snapshots created: {results['snapshots_created']}")
        print(f"  Duration: {results['duration_seconds']:.1f}s")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            return 1
        
        return 0


if __name__ == '__main__':
    # Create logs directory if needed
    Path('logs').mkdir(exist_ok=True)
    
    # Run async main
    exit_code = asyncio.run(main())
    exit(exit_code)
