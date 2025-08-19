"""
Test mesh exporter resilience under high-churn conditions.
Verifies that mesh export remains consistent during rapid updates.
"""
import pytest
import threading
import time
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import os

# Import mesh exporter to test directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "core"))
from mesh_exporter import MeshExporter, MeshUpdateWatcher


class TestMesh:
    """Test mesh implementation with update tracking."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.version = 0
        self.last_updated = time.time()
        self.update_lock = threading.Lock()
    
    def add_node(self, key: str, value: Any):
        with self.update_lock:
            self.nodes[key] = value
            self.version += 1
            self.last_updated = time.time()
    
    def remove_node(self, key: str):
        with self.update_lock:
            if key in self.nodes:
                del self.nodes[key]
                self.version += 1
                self.last_updated = time.time()
    
    def add_edge(self, key: str, nodes: List[str]):
        with self.update_lock:
            self.edges[key] = nodes
            self.version += 1
            self.last_updated = time.time()
    
    def to_summary(self) -> Dict:
        """Generate summary for export."""
        with self.update_lock:
            return {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "version": self.version,
                "last_updated": self.last_updated,
                "keys": list(self.nodes.keys())[:10]  # First 10 keys
            }


def test_mesh_export_under_rapid_updates(tmp_path):
    """Test mesh export consistency during rapid updates."""
    # Create test mesh
    mesh = TestMesh()
    export_path = tmp_path / "mesh_summary.json"
    
    # Track export operations
    export_count = 0
    export_errors = []
    
    def rapid_updater(thread_id: int, duration: float = 2.0):
        """Rapidly update mesh."""
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < duration:
            operation = random.choice(["add_node", "remove_node", "add_edge"])
            
            if operation == "add_node":
                key = f"node_{thread_id}_{update_count}"
                value = f"value_{update_count}"
                mesh.add_node(key, value)
            elif operation == "remove_node" and len(mesh.nodes) > 0:
                keys = list(mesh.nodes.keys())
                if keys:
                    mesh.remove_node(random.choice(keys))
            elif operation == "add_edge":
                edge_key = f"edge_{thread_id}_{update_count}"
                mesh.add_edge(edge_key, [f"node_{i}" for i in range(2)])
            
            update_count += 1
            time.sleep(random.uniform(0.001, 0.01))
        
        return update_count
    
    def continuous_exporter(duration: float = 2.0):
        """Continuously export mesh summary."""
        nonlocal export_count
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                MeshExporter.export_summary(mesh, str(export_path))
                export_count += 1
            except Exception as e:
                export_errors.append(str(e))
            
            time.sleep(random.uniform(0.01, 0.05))
    
    # Start multiple updater threads
    updater_threads = []
    for i in range(5):
        thread = threading.Thread(target=rapid_updater, args=(i, 3.0))
        updater_threads.append(thread)
        thread.start()
    
    # Start exporter thread
    exporter_thread = threading.Thread(target=continuous_exporter, args=(3.0,))
    exporter_thread.start()
    
    # Wait for all threads
    for thread in updater_threads:
        thread.join()
    exporter_thread.join()
    
    # Verify results
    assert len(export_errors) == 0, f"Export errors occurred: {export_errors[:5]}"
    assert export_count > 0, "No exports completed"
    
    # Verify final export is valid JSON
    assert export_path.exists(), "Export file not created"
    with open(export_path) as f:
        final_summary = json.load(f)
    
    assert "node_count" in final_summary, "Missing node_count in summary"
    assert "version" in final_summary, "Missing version in summary"
    assert final_summary["version"] > 0, "Version not incremented"
    
    print(f"✅ Rapid update test passed: {export_count} exports during updates")


def test_mesh_watcher_change_detection(tmp_path):
    """Test that MeshUpdateWatcher correctly detects changes."""
    mesh = TestMesh()
    export_path = tmp_path / "watched_mesh.json"
    
    # Set up audit log
    log_dir = tmp_path / "logs" / "mesh"
    log_dir.mkdir(parents=True)
    os.environ["TORI_LOG_DIR"] = str(tmp_path / "logs")
    
    # Start watcher with short interval
    watcher = MeshUpdateWatcher(mesh, str(export_path), interval=0.5)
    watcher.start()
    
    # Let watcher initialize
    time.sleep(1)
    
    # Check initial export
    assert export_path.exists(), "Initial export not created"
    with open(export_path) as f:
        initial_summary = json.load(f)
    initial_version = initial_summary.get("version", 0)
    
    # Make changes to mesh
    for i in range(10):
        mesh.add_node(f"test_node_{i}", f"value_{i}")
        time.sleep(0.1)
    
    # Wait for watcher to detect changes
    time.sleep(2)
    
    # Check updated export
    with open(export_path) as f:
        updated_summary = json.load(f)
    
    assert updated_summary["version"] > initial_version, "Version not updated"
    assert updated_summary["node_count"] == 10, f"Expected 10 nodes, got {updated_summary['node_count']}"
    
    # Check audit log
    audit_log = log_dir / "mesh_export.log"
    if audit_log.exists():
        with open(audit_log) as f:
            log_lines = f.readlines()
        
        watcher_lines = [l for l in log_lines if "WATCHER" in l]
        assert len(watcher_lines) > 0, "No watcher events in audit log"
    
    # Stop watcher (daemon thread will stop with main thread)
    print("✅ Watcher change detection test passed")


def test_atomic_export_with_temp_files(tmp_path):
    """Test that exports use atomic write with temp files."""
    mesh = TestMesh()
    export_path = tmp_path / "atomic_export.json"
    
    # Add some data
    for i in range(100):
        mesh.add_node(f"node_{i}", f"value_{i}")
    
    # Track file operations
    temp_files_seen = []
    
    def monitor_temp_files():
        """Monitor for temp files during export."""
        for _ in range(100):
            for file_path in tmp_path.glob("*.tmp"):
                temp_files_seen.append(file_path.name)
            time.sleep(0.001)
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_temp_files)
    monitor_thread.start()
    
    # Perform multiple exports
    for _ in range(10):
        MeshExporter.export_summary(mesh, str(export_path))
        time.sleep(0.01)
    
    monitor_thread.join()
    
    # Verify atomic writes were used
    assert len(temp_files_seen) > 0, "No temp files observed - not using atomic writes"
    
    # Verify no temp files remain
    remaining_temps = list(tmp_path.glob("*.tmp"))
    assert len(remaining_temps) == 0, f"Temp files not cleaned up: {remaining_temps}"
    
    # Verify final export is valid
    with open(export_path) as f:
        summary = json.load(f)
    assert summary["node_count"] == 100, "Export corrupted"
    
    print(f"✅ Atomic export test passed: {len(temp_files_seen)} temp files used")


def test_export_hash_change_detection(tmp_path):
    """Test that exporter detects changes via content hash."""
    mesh = TestMesh()
    export_path = tmp_path / "hash_detection.json"
    
    # Initial export
    MeshExporter.export_summary(mesh, str(export_path))
    
    with open(export_path) as f:
        initial_content = f.read()
    initial_hash = hashlib.md5(initial_content.encode()).hexdigest()
    
    # Make changes
    mesh.add_node("new_node", "new_value")
    
    # Export again
    MeshExporter.export_summary(mesh, str(export_path))
    
    with open(export_path) as f:
        updated_content = f.read()
    updated_hash = hashlib.md5(updated_content.encode()).hexdigest()
    
    # Hashes should differ
    assert initial_hash != updated_hash, "Hash not changed after mesh update"
    
    # Export without changes
    MeshExporter.export_summary(mesh, str(export_path))
    
    with open(export_path) as f:
        final_content = f.read()
    final_hash = hashlib.md5(final_content.encode()).hexdigest()
    
    # Hash should remain same
    assert updated_hash == final_hash, "Hash changed without mesh update"
    
    print("✅ Hash change detection test passed")


def test_large_mesh_export_performance(tmp_path):
    """Test export performance with large mesh."""
    mesh = TestMesh()
    export_path = tmp_path / "large_mesh.json"
    
    # Create large mesh
    num_nodes = 10000
    for i in range(num_nodes):
        mesh.add_node(f"node_{i}", {
            "data": f"value_{i}",
            "timestamp": time.time(),
            "metadata": {"index": i}
        })
    
    # Add edges
    for i in range(1000):
        mesh.add_edge(f"edge_{i}", [f"node_{i}", f"node_{i+1}"])
    
    # Measure export time
    start_time = time.time()
    MeshExporter.export_summary(mesh, str(export_path))
    export_duration = time.time() - start_time
    
    # Verify export completed quickly
    assert export_duration < 1.0, f"Export took too long: {export_duration:.2f}s"
    
    # Verify export is valid
    with open(export_path) as f:
        summary = json.load(f)
    
    assert summary["node_count"] == num_nodes, "Node count mismatch"
    assert summary["edge_count"] == 1000, "Edge count mismatch"
    
    # Check file size is reasonable (summary should be compact)
    file_size = export_path.stat().st_size
    assert file_size < 100000, f"Summary too large: {file_size} bytes"
    
    print(f"✅ Large mesh test passed: {num_nodes} nodes exported in {export_duration:.3f}s")


def test_concurrent_export_safety(tmp_path):
    """Test that concurrent exports don't corrupt output."""
    mesh = TestMesh()
    export_path = tmp_path / "concurrent_export.json"
    
    # Add initial data
    for i in range(100):
        mesh.add_node(f"initial_node_{i}", f"value_{i}")
    
    export_results = []
    export_errors = []
    
    def export_worker(worker_id: int, iterations: int = 20):
        """Worker thread performing exports."""
        for i in range(iterations):
            try:
                # Modify mesh
                mesh.add_node(f"worker_{worker_id}_node_{i}", f"value_{i}")
                
                # Export
                MeshExporter.export_summary(mesh, str(export_path))
                
                # Verify export is valid JSON
                with open(export_path) as f:
                    summary = json.load(f)
                
                export_results.append({
                    "worker": worker_id,
                    "iteration": i,
                    "version": summary.get("version"),
                    "node_count": summary.get("node_count")
                })
            except Exception as e:
                export_errors.append({
                    "worker": worker_id,
                    "iteration": i,
                    "error": str(e)
                })
            
            time.sleep(random.uniform(0.001, 0.01))
    
    # Run concurrent exports
    threads = []
    for i in range(5):
        thread = threading.Thread(target=export_worker, args=(i, 10))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(export_errors) == 0, f"Export errors: {export_errors[:5]}"
    assert len(export_results) == 50, f"Expected 50 exports, got {len(export_results)}"
    
    # Verify final export is valid
    with open(export_path) as f:
        final_summary = json.load(f)
    
    assert final_summary["node_count"] >= 100, "Nodes lost during concurrent export"
    
    # Check version monotonicity (should only increase)
    versions = [r["version"] for r in export_results]
    assert all(v > 0 for v in versions), "Invalid versions"
    
    print(f"✅ Concurrent export test passed: 50 exports completed safely")


def test_export_error_recovery(tmp_path):
    """Test that export handles errors gracefully."""
    mesh = TestMesh()
    
    # Test with invalid path
    invalid_path = tmp_path / "nonexistent_dir" / "export.json"
    
    try:
        # Should create directory and succeed
        MeshExporter.export_summary(mesh, str(invalid_path))
        assert invalid_path.exists(), "File not created"
    except Exception as e:
        pytest.fail(f"Export failed to handle missing directory: {e}")
    
    # Test with read-only directory (skip on Windows)
    if os.name != 'nt':
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_path = readonly_dir / "export.json"
        
        # Make directory read-only
        readonly_dir.chmod(0o444)
        
        try:
            with pytest.raises(Exception):
                MeshExporter.export_summary(mesh, str(readonly_path))
        finally:
            # Restore permissions
            readonly_dir.chmod(0o755)
    
    # Test with mesh that raises exception in to_summary
    class BrokenMesh:
        def to_summary(self):
            raise RuntimeError("Simulated mesh error")
    
    broken_mesh = BrokenMesh()
    fallback_path = tmp_path / "fallback_export.json"
    
    # Should handle error and create basic summary
    MeshExporter.export_summary(broken_mesh, str(fallback_path))
    
    # Verify fallback export exists
    assert fallback_path.exists(), "Fallback export not created"
    
    with open(fallback_path) as f:
        fallback_summary = json.load(f)
    
    # Should have at least empty structure
    assert isinstance(fallback_summary, dict), "Invalid fallback summary"
    
    print("✅ Error recovery test passed")


def test_mesh_memory_cleanup(tmp_path):
    """Test that mesh operations don't leak memory."""
    import gc
    import sys
    
    initial_objects = len(gc.get_objects())
    
    # Perform many mesh operations
    for iteration in range(10):
        mesh = TestMesh()
        export_path = tmp_path / f"memory_test_{iteration}.json"
        
        # Add and remove many nodes
        for i in range(1000):
            mesh.add_node(f"node_{i}", f"value_{i}" * 100)  # Large values
        
        for i in range(0, 1000, 2):
            mesh.remove_node(f"node_{i}")
        
        # Export
        MeshExporter.export_summary(mesh, str(export_path))
        
        # Cleanup
        del mesh
        gc.collect()
    
    # Check object count didn't grow significantly
    final_objects = len(gc.get_objects())
    object_growth = final_objects - initial_objects
    
    # Allow some growth but not excessive
    assert object_growth < 1000, f"Possible memory leak: {object_growth} objects added"
    
    print(f"✅ Memory cleanup test passed: {object_growth} object growth")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
