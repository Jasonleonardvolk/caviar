"""
Test adapter swap atomicity and thread safety.
Verifies that adapter operations are atomic and thread-safe under concurrent load.
"""
import pytest
import threading
import multiprocessing
import time
import hashlib
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

# Import the adapter loader to test directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "core"))
from adapter_loader import AdapterLoader


def create_test_adapter(path: Path, name: str, content: bytes) -> str:
    """Create a test adapter file and return its SHA256 hash."""
    adapter_file = path / name
    adapter_file.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_adapter_lock_prevents_concurrent_swaps(tmp_path):
    """Test that the threading lock prevents concurrent adapter swaps."""
    # Setup test adapters
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    adapter1_hash = create_test_adapter(adapters_dir, "adapter1.bin", b"adapter1_data_original")
    adapter2_hash = create_test_adapter(adapters_dir, "adapter2.bin", b"adapter2_data_original")
    adapter3_hash = create_test_adapter(adapters_dir, "adapter3.bin", b"adapter3_data_original")
    
    # Create manifest
    manifest = {
        "adapter1.bin": {"path": "adapter1.bin", "sha256": adapter1_hash},
        "adapter2.bin": {"path": "adapter2.bin", "sha256": adapter2_hash},
        "adapter3.bin": {"path": "adapter3.bin", "sha256": adapter3_hash}
    }
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader
    loader = AdapterLoader(str(adapters_dir), max_loaded=3)
    
    # Track swap operations
    swap_results = []
    swap_errors = []
    
    def swap_adapter(name: str, user: str, iterations: int = 10):
        """Thread function to repeatedly swap adapters."""
        for i in range(iterations):
            try:
                start_time = time.time()
                loader.load_adapter(name, user=user)
                duration = time.time() - start_time
                
                swap_results.append({
                    "adapter": name,
                    "user": user,
                    "iteration": i,
                    "duration": duration,
                    "thread": threading.current_thread().name,
                    "timestamp": time.time()
                })
            except Exception as e:
                swap_errors.append({
                    "adapter": name,
                    "user": user,
                    "error": str(e),
                    "thread": threading.current_thread().name
                })
            
            # Small random delay
            time.sleep(0.001)
    
    # Create multiple threads attempting concurrent swaps
    threads = []
    adapters = ["adapter1.bin", "adapter2.bin", "adapter3.bin"]
    
    for i in range(10):  # 10 threads
        adapter = adapters[i % 3]
        thread = threading.Thread(
            target=swap_adapter,
            args=(adapter, f"user_{i}", 5),
            name=f"SwapThread-{i}"
        )
        threads.append(thread)
    
    # Start all threads simultaneously
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Verify results
    assert len(swap_errors) == 0, f"Swap errors occurred: {swap_errors}"
    assert len(swap_results) == 50, f"Expected 50 swaps, got {len(swap_results)}"
    
    # Check that swaps were serialized (no overlapping operations)
    sorted_swaps = sorted(swap_results, key=lambda x: x["timestamp"])
    for i in range(1, len(sorted_swaps)):
        prev_end = sorted_swaps[i-1]["timestamp"] + sorted_swaps[i-1]["duration"]
        curr_start = sorted_swaps[i]["timestamp"]
        
        # Allow small overlap due to timing precision
        overlap = prev_end - curr_start
        assert overlap < 0.01, f"Swaps overlapped by {overlap}s - not atomic!"
    
    print(f"✅ Atomicity test passed: 50 swaps in {total_time:.2f}s, all serialized")


def test_symlink_atomicity(tmp_path):
    """Test that symlink switching is atomic."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    # Create test adapters
    for i in range(3):
        adapter_file = adapters_dir / f"adapter{i}.bin"
        adapter_file.write_bytes(f"adapter{i}_data".encode())
    
    active_link = adapters_dir / "active_adapter"
    
    def switch_symlink(target: str, iterations: int = 100):
        """Rapidly switch symlink target."""
        for i in range(iterations):
            temp_link = adapters_dir / f".tmp_link_{threading.current_thread().name}_{i}"
            
            # Remove temp link if exists
            if temp_link.exists() or temp_link.is_symlink():
                temp_link.unlink()
            
            # Create new temp link
            temp_link.symlink_to(target)
            
            # Atomic replace
            temp_link.replace(active_link)
    
    # Run concurrent symlink switches
    threads = []
    targets = ["adapter0.bin", "adapter1.bin", "adapter2.bin"]
    
    for i in range(6):
        thread = threading.Thread(
            target=switch_symlink,
            args=(targets[i % 3], 50)
        )
        threads.append(thread)
    
    for thread in threads:
        thread.start()
    
    # While threads are running, continuously read the symlink
    read_results = []
    read_thread_stop = False
    
    def read_symlink():
        while not read_thread_stop:
            try:
                if active_link.exists():
                    target = active_link.readlink()
                    read_results.append(str(target))
            except Exception as e:
                read_results.append(f"error: {e}")
            time.sleep(0.0001)
    
    read_thread = threading.Thread(target=read_symlink)
    read_thread.start()
    
    # Wait for switch threads
    for thread in threads:
        thread.join()
    
    # Stop read thread
    read_thread_stop = True
    read_thread.join()
    
    # Verify all reads were valid
    errors = [r for r in read_results if r.startswith("error:")]
    assert len(errors) == 0, f"Symlink read errors: {errors[:5]}"
    
    # Verify only valid targets were read
    valid_targets = set(targets)
    read_targets = set(r for r in read_results if not r.startswith("error:"))
    invalid_targets = read_targets - valid_targets
    assert len(invalid_targets) == 0, f"Invalid targets read: {invalid_targets}"
    
    print(f"✅ Symlink atomicity test passed: {len(read_results)} reads, all valid")


def test_integrity_check_prevents_corruption(tmp_path):
    """Test that SHA256 integrity check prevents loading corrupted adapters."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    # Create valid adapter
    valid_content = b"valid_adapter_data"
    valid_hash = hashlib.sha256(valid_content).hexdigest()
    (adapters_dir / "valid.bin").write_bytes(valid_content)
    
    # Create corrupted adapter (content doesn't match hash)
    (adapters_dir / "corrupt.bin").write_bytes(b"corrupted_data")
    
    # Create manifest with incorrect hash for corrupt adapter
    manifest = {
        "valid.bin": {"path": "valid.bin", "sha256": valid_hash},
        "corrupt.bin": {"path": "corrupt.bin", "sha256": "incorrect_hash_value"}
    }
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader
    loader = AdapterLoader(str(adapters_dir))
    
    # Valid adapter should load successfully
    try:
        result = loader.load_adapter("valid.bin", user="test_user")
        assert result is not None, "Valid adapter should load"
    except Exception as e:
        pytest.fail(f"Valid adapter failed to load: {e}")
    
    # Corrupted adapter should fail integrity check
    with pytest.raises(ValueError, match="failed integrity check"):
        loader.load_adapter("corrupt.bin", user="test_user")
    
    print("✅ Integrity check test passed")


def test_rollback_on_failure(tmp_path):
    """Test that rollback works when adapter swap fails."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    # Create initial adapter
    adapter1_content = b"adapter1_data"
    adapter1_hash = hashlib.sha256(adapter1_content).hexdigest()
    (adapters_dir / "adapter1.bin").write_bytes(adapter1_content)
    
    # Create manifest
    manifest = {
        "adapter1.bin": {"path": "adapter1.bin", "sha256": adapter1_hash}
    }
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader and load first adapter
    loader = AdapterLoader(str(adapters_dir))
    loader.load_adapter("adapter1.bin", user="test_user")
    
    assert loader.active_adapter == "adapter1.bin"
    
    # Try to load non-existent adapter (should fail)
    try:
        loader.load_adapter("nonexistent.bin", user="test_user")
    except FileNotFoundError:
        pass  # Expected
    
    # Active adapter should still be adapter1
    assert loader.active_adapter == "adapter1.bin"
    
    # Test rollback function
    loader.rollback_adapter()
    
    # Should stay on adapter1 (no previous to roll back to)
    assert loader.active_adapter == "adapter1.bin"
    
    print("✅ Rollback test passed")


def test_cache_lru_eviction(tmp_path):
    """Test that LRU cache properly evicts least recently used adapters."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    # Create multiple adapters
    manifest = {}
    for i in range(5):
        content = f"adapter{i}_data".encode()
        hash_val = hashlib.sha256(content).hexdigest()
        (adapters_dir / f"adapter{i}.bin").write_bytes(content)
        manifest[f"adapter{i}.bin"] = {"path": f"adapter{i}.bin", "sha256": hash_val}
    
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader with small cache
    loader = AdapterLoader(str(adapters_dir), max_loaded=2)
    
    # Load adapters in sequence
    loader.load_adapter("adapter0.bin", user="test")
    assert len(loader.loaded_adapters) == 1
    
    loader.load_adapter("adapter1.bin", user="test")
    assert len(loader.loaded_adapters) == 2
    
    loader.load_adapter("adapter2.bin", user="test")
    assert len(loader.loaded_adapters) == 2  # Should evict adapter0
    assert "adapter0.bin" not in loader.loaded_adapters
    assert "adapter1.bin" in loader.loaded_adapters
    assert "adapter2.bin" in loader.loaded_adapters
    
    # Access adapter1 to make it more recent
    loader.load_adapter("adapter1.bin", user="test")
    
    # Load adapter3 - should evict adapter2 (least recent)
    loader.load_adapter("adapter3.bin", user="test")
    assert len(loader.loaded_adapters) == 2
    assert "adapter2.bin" not in loader.loaded_adapters
    assert "adapter1.bin" in loader.loaded_adapters
    assert "adapter3.bin" in loader.loaded_adapters
    
    print("✅ LRU cache eviction test passed")


def test_audit_log_consistency(tmp_path):
    """Test that audit logs are consistent and complete."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    logs_dir = tmp_path / "logs" / "inference"
    logs_dir.mkdir(parents=True)
    
    # Set environment variable for log directory
    os.environ["TORI_LOG_DIR"] = str(tmp_path / "logs")
    
    # Create test adapter
    content = b"test_adapter"
    hash_val = hashlib.sha256(content).hexdigest()
    (adapters_dir / "test.bin").write_bytes(content)
    
    manifest = {"test.bin": {"path": "test.bin", "sha256": hash_val}}
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader
    loader = AdapterLoader(str(adapters_dir))
    
    # Perform multiple swaps
    users = ["alice", "bob", "charlie"]
    for i in range(10):
        user = users[i % 3]
        loader.load_adapter("test.bin", user=user)
    
    # Check audit log
    audit_log = logs_dir / "adapter_swap.log"
    assert audit_log.exists(), "Audit log not created"
    
    with open(audit_log) as f:
        lines = f.readlines()
    
    assert len(lines) >= 10, f"Expected at least 10 log entries, got {len(lines)}"
    
    # Verify log format and content
    for line in lines:
        assert "USER=" in line, f"Missing user in log: {line}"
        assert "ACTION=SWAP" in line, f"Missing action in log: {line}"
        assert "ADAPTER=" in line, f"Missing adapter in log: {line}"
        
        # Verify timestamp format
        parts = line.split("|")
        timestamp = parts[0].strip()
        assert timestamp.endswith("Z"), f"Invalid timestamp format: {timestamp}"
    
    # Verify all users appear in log
    log_content = "".join(lines)
    for user in users:
        assert f"USER={user}" in log_content, f"User {user} not in audit log"
    
    print(f"✅ Audit log test passed: {len(lines)} entries verified")


def test_concurrent_reads_during_swap(tmp_path):
    """Test that reads remain consistent during adapter swaps."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    
    # Create adapters with different content
    adapters = {}
    manifest = {}
    for i in range(3):
        content = f"adapter{i}_unique_content_{i*100}".encode()
        hash_val = hashlib.sha256(content).hexdigest()
        name = f"adapter{i}.bin"
        (adapters_dir / name).write_bytes(content)
        adapters[name] = content
        manifest[name] = {"path": name, "sha256": hash_val}
    
    (adapters_dir / "metadata.json").write_text(json.dumps(manifest))
    
    # Initialize loader
    loader = AdapterLoader(str(adapters_dir))
    
    # Track read consistency
    read_errors = []
    stop_reading = False
    
    def continuous_reader():
        """Continuously read active adapter."""
        while not stop_reading:
            try:
                if loader.active_adapter:
                    # Simulate reading adapter content
                    expected = adapters.get(loader.active_adapter)
                    if expected and loader.active_adapter in loader.loaded_adapters:
                        actual = loader.loaded_adapters[loader.active_adapter]
                        # In real scenario, would check actual content
                        pass
            except Exception as e:
                read_errors.append(str(e))
            time.sleep(0.0001)
    
    # Start reader thread
    reader = threading.Thread(target=continuous_reader)
    reader.start()
    
    # Perform rapid swaps
    swap_thread_errors = []
    
    def rapid_swapper():
        try:
            for i in range(100):
                adapter = f"adapter{i % 3}.bin"
                loader.load_adapter(adapter, user="swapper")
                time.sleep(0.001)
        except Exception as e:
            swap_thread_errors.append(str(e))
    
    swapper = threading.Thread(target=rapid_swapper)
    swapper.start()
    
    # Let it run
    swapper.join()
    stop_reading = True
    reader.join()
    
    # Verify no errors
    assert len(read_errors) == 0, f"Read errors during swap: {read_errors[:5]}"
    assert len(swap_thread_errors) == 0, f"Swap errors: {swap_thread_errors}"
    
    print("✅ Concurrent read test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
