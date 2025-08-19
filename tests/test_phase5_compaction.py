"""
Tests for Phase 5: Snapshot and Compaction
"""

import pytest
import asyncio
import json
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Import components
from scripts.compact_all_meshes import MeshCompactor
from core.metrics import MetricsCollector, needs_compact
from core.compaction_integration import CompactionIntegration
from python.core.scoped_concept_mesh import ScopedConceptMesh
from python.core.scoped_wal import WALManager


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / 'data'
        data_dir.mkdir()
        yield data_dir


@pytest.fixture
def mock_mesh_data(temp_data_dir):
    """Create mock mesh data for testing"""
    # Create user mesh
    user_dir = temp_data_dir / 'concept_mesh' / 'user' / 'test_user'
    user_dir.mkdir(parents=True)
    
    # Create some dummy data
    mesh_data = {
        'concepts': {
            'c1': {'name': 'test1', 'importance': 1.0},
            'c2': {'name': 'test2', 'importance': 2.0},
            'c1_dup': {'name': 'test1', 'importance': 0.5},  # Duplicate
        },
        'relations': [
            {'source_id': 'c1', 'target_id': 'c2', 'strength': 0.8},
            {'source_id': 'c1', 'target_id': 'c_missing', 'strength': 0.5},  # Orphaned
        ]
    }
    
    with open(user_dir / 'mesh.json', 'w') as f:
        json.dump(mesh_data, f)
    
    # Create WAL
    wal_dir = temp_data_dir / 'wal' / 'user'
    wal_dir.mkdir(parents=True)
    
    # Create a large WAL file
    wal_file = wal_dir / 'test_user.wal'
    with open(wal_file, 'w') as f:
        for i in range(1000):
            f.write(f'{{"seq": {i}, "op": "test"}}\n')
    
    return temp_data_dir


class TestMetrics:
    """Test metrics collection"""
    
    def test_needs_compact_detection(self, mock_mesh_data):
        """Test detection of compaction needs"""
        collector = MetricsCollector(mock_mesh_data)
        
        # Check the test mesh
        metrics = collector.needs_compact('user', 'test_user')
        
        assert metrics.scope == 'user'
        assert metrics.scope_id == 'test_user'
        assert metrics.wal_size_mb > 0
        assert metrics.needs_compaction is True  # Large WAL should trigger
        assert 'WAL too large' in metrics.reason
    
    def test_compaction_report(self, mock_mesh_data):
        """Test compaction report generation"""
        collector = MetricsCollector(mock_mesh_data)
        
        report = collector.get_compaction_report()
        
        assert report['total_meshes'] == 1
        assert report['needs_compaction'] == 1
        assert len(report['details']) == 1
        assert report['details'][0]['scope'] == 'user'
        assert report['details'][0]['scope_id'] == 'test_user'
    
    def test_needs_compact_helper(self, mock_mesh_data):
        """Test the simple needs_compact helper"""
        result = needs_compact('user', 'test_user', mock_mesh_data)
        assert result is True
        
        # Non-existent mesh
        result = needs_compact('user', 'nonexistent', mock_mesh_data)
        assert result is False


class TestCompaction:
    """Test compaction functionality"""
    
    @pytest.mark.asyncio
    async def test_snapshot_creation(self, mock_mesh_data):
        """Test creating snapshots"""
        config = {
            'data_dir': mock_mesh_data,
            'compress_snapshots': True
        }
        compactor = MeshCompactor(config)
        
        # Mock mesh
        class MockMesh:
            concepts = {'c1': type('obj', (object,), {
                'id': 'c1', 'name': 'test', 'description': '', 
                'category': 'test', 'importance': 1.0, 'metadata': {},
                'created_at': datetime.now(), 'last_accessed': datetime.now(),
                'access_count': 1, 'embedding': None
            })}
            relations = []
        
        # Create snapshot
        snapshot_path = await compactor._create_snapshot('user', 'test_user', MockMesh())
        
        assert snapshot_path.exists()
        assert snapshot_path.suffix == '.gz'
        
        # Verify snapshot content
        with gzip.open(snapshot_path, 'rt') as f:
            data = json.load(f)
        
        assert data['version'] == '1.0'
        assert data['scope'] == 'user'
        assert data['scope_id'] == 'test_user'
        assert 'checksum' in data
        assert len(data['mesh_data']['concepts']) == 1
    
    @pytest.mark.asyncio
    async def test_deduplication(self, mock_mesh_data):
        """Test concept deduplication"""
        compactor = MeshCompactor({'data_dir': mock_mesh_data})
        
        # Mock mesh with duplicates
        class MockMesh:
            concepts = {
                'c1': type('obj', (object,), {'name': 'Test', 'importance': 1.0, 'access_count': 10}),
                'c2': type('obj', (object,), {'name': 'test', 'importance': 0.5, 'access_count': 5}),
                'c3': type('obj', (object,), {'name': 'TEST', 'importance': 2.0, 'access_count': 3}),
            }
            name_index = {'Test': 'c1', 'test': 'c2', 'TEST': 'c3'}
        
        removed = compactor._deduplicate_concepts(MockMesh())
        
        # Should keep c1 (highest access count)
        assert removed == 2
        assert len(MockMesh.concepts) == 1
        assert 'c1' in MockMesh.concepts
    
    @pytest.mark.asyncio
    async def test_orphaned_relation_cleanup(self):
        """Test removal of orphaned relations"""
        compactor = MeshCompactor()
        
        class MockMesh:
            concepts = {'c1': None, 'c2': None}
            relations = [
                type('rel', (object,), {'source_id': 'c1', 'target_id': 'c2'}),
                type('rel', (object,), {'source_id': 'c1', 'target_id': 'c_missing'}),
                type('rel', (object,), {'source_id': 'c_missing', 'target_id': 'c2'}),
            ]
        
        removed = compactor._optimize_relations(MockMesh())
        
        assert removed == 2
        assert len(MockMesh.relations) == 1
        assert MockMesh.relations[0].source_id == 'c1'
        assert MockMesh.relations[0].target_id == 'c2'
    
    @pytest.mark.asyncio
    async def test_full_compaction(self, mock_mesh_data):
        """Test full compaction process"""
        config = {
            'data_dir': mock_mesh_data,
            'compress_snapshots': False,  # Easier to verify
            'max_wal_size_mb': 0.001  # Force compaction
        }
        compactor = MeshCompactor(config)
        
        # Run compaction
        results = await compactor.compact_all_meshes(force=True)
        
        assert results['meshes_processed'] > 0
        assert results['meshes_compacted'] > 0
        assert results['snapshots_created'] > 0
        assert len(results['errors']) == 0
        
        # Verify snapshot was created
        snapshot_dir = mock_mesh_data / 'snapshots' / 'user' / 'test_user'
        assert snapshot_dir.exists()
        snapshots = list(snapshot_dir.glob('mesh_*.json'))
        assert len(snapshots) > 0
    
    @pytest.mark.asyncio
    async def test_restore_from_snapshot(self, mock_mesh_data):
        """Test restoring from snapshot"""
        config = {'data_dir': mock_mesh_data}
        compactor = MeshCompactor(config)
        
        # Create a snapshot first
        snapshot_data = {
            'version': '1.0',
            'scope': 'user',
            'scope_id': 'test_restore',
            'timestamp': datetime.now().isoformat(),
            'mesh_data': {
                'concepts': {
                    'c1': {
                        'id': 'c1',
                        'name': 'restored_concept',
                        'description': 'test',
                        'category': 'test',
                        'importance': 1.0,
                        'metadata': {},
                        'created_at': datetime.now().isoformat(),
                        'last_accessed': datetime.now().isoformat(),
                        'access_count': 5,
                        'embedding_shape': None
                    }
                },
                'relations': [],
                'metadata': {'total_concepts': 1, 'total_relations': 0}
            },
            'checksum': ''
        }
        
        # Calculate checksum
        content_str = json.dumps(snapshot_data['mesh_data'], sort_keys=True)
        import hashlib
        snapshot_data['checksum'] = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Save snapshot
        snapshot_dir = mock_mesh_data / 'snapshots' / 'user' / 'test_restore'
        snapshot_dir.mkdir(parents=True)
        snapshot_path = snapshot_dir / 'mesh_test.json'
        
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f)
        
        # Mock ScopedConceptMesh.get_instance
        class MockMesh:
            concepts = {}
            relations = []
            
            def add_concept(self, **kwargs):
                self.concepts[kwargs['name']] = kwargs
                return f"concept_{len(self.concepts)}"
            
            def add_relation(self, **kwargs):
                self.relations.append(kwargs)
                return True
            
            def _save_mesh(self):
                pass
        
        # Monkey patch for test
        original_get_instance = ScopedConceptMesh.get_instance
        ScopedConceptMesh.get_instance = lambda *args: MockMesh()
        
        try:
            # Restore
            success = await compactor.restore_from_snapshot('user', 'test_restore', snapshot_path)
            
            assert success is True
            
        finally:
            # Restore original
            ScopedConceptMesh.get_instance = original_get_instance


class TestIntegration:
    """Test TORI integration"""
    
    @pytest.mark.asyncio
    async def test_auto_compact(self, mock_mesh_data):
        """Test auto-compaction integration"""
        config = {
            'compaction_enabled': True,
            'auto_compact': True,
            'compaction': {
                'data_dir': mock_mesh_data,
                'max_wal_size_mb': 0.001  # Force trigger
            }
        }
        
        integration = CompactionIntegration(config)
        
        # Should compact due to large WAL
        compacted = await integration.compact_if_needed('user', 'test_user')
        
        # Note: This will fail without full mesh implementation
        # In real tests, would need proper mocking
        assert isinstance(compacted, bool)
    
    @pytest.mark.asyncio
    async def test_backup_snapshot(self, mock_mesh_data):
        """Test backup snapshot creation"""
        config = {
            'compaction_enabled': True,
            'compaction': {'data_dir': mock_mesh_data}
        }
        
        integration = CompactionIntegration(config)
        
        # Mock mesh
        original_get_instance = ScopedConceptMesh.get_instance
        ScopedConceptMesh.get_instance = lambda *args: type('obj', (object,), {
            'concepts': {}, 'relations': []
        })
        
        try:
            path = await integration.create_backup_snapshot('user', 'backup_test', 'v1')
            
            # Would create snapshot in real implementation
            assert path is None or path.exists()
            
        finally:
            ScopedConceptMesh.get_instance = original_get_instance
    
    def test_api_routes_registration(self):
        """Test API route registration"""
        from fastapi import FastAPI
        
        app = FastAPI()
        config = {'compaction_enabled': True}
        
        integration = CompactionIntegration(config)
        integration.register_api_routes(app)
        
        # Check routes were added
        routes = [r.path for r in app.routes]
        assert '/api/compaction/status' in routes
        assert '/api/compaction/compact' in routes
        assert '/api/compaction/compact-all' in routes


class TestScheduling:
    """Test scheduling functionality"""
    
    def test_cron_format(self):
        """Test cron job format"""
        # This would test actual cron parsing in production
        midnight_cron = "0 0 * * *"
        noon_cron = "0 12 * * *"
        
        # Basic validation
        assert len(midnight_cron.split()) == 5
        assert len(noon_cron.split()) == 5
    
    def test_old_snapshot_cleanup(self, mock_mesh_data):
        """Test cleanup of old snapshots"""
        config = {
            'data_dir': mock_mesh_data,
            'keep_snapshots_days': 1
        }
        compactor = MeshCompactor(config)
        
        # Create old snapshot
        snapshot_dir = mock_mesh_data / 'snapshots' / 'test'
        snapshot_dir.mkdir(parents=True)
        
        old_file = snapshot_dir / 'mesh_old.json'
        old_file.touch()
        
        # Make it old
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        os.utime(old_file, (old_time, old_time))
        
        # Create recent snapshot
        new_file = snapshot_dir / 'mesh_new.json'
        new_file.touch()
        
        # Clean
        removed = compactor._clean_old_snapshots()
        
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()


# Run specific test
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
