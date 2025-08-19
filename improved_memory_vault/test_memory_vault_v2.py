"""
Comprehensive Test Suite for Memory Vault V2
Tests all major functionality including edge cases and performance
"""

import asyncio
import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List
import signal
import os

from memory_vault_v2 import UnifiedMemoryVaultV2, MemoryType, MemoryEntry


class TestMemoryVaultV2:
    """Test suite for UnifiedMemoryVaultV2"""
    
    @pytest.fixture
    async def vault(self):
        """Create a test vault instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'storage_path': tmpdir,
                'max_working_memory': 10,
                'ghost_memory_ttl': 2,  # 2 seconds for testing
                'batch_size': 5
            }
            vault = UnifiedMemoryVaultV2(config)
            await vault.initialize()
            yield vault
            await vault.shutdown()
    
    @pytest.mark.asyncio
    async def test_basic_store_retrieve(self, vault):
        """Test basic store and retrieve operations"""
        # Store a memory
        content = "Test memory content"
        memory_id = await vault.store(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            metadata={'test': True},
            importance=0.8
        )
        
        assert memory_id is not None
        
        # Retrieve the memory
        memory = await vault.retrieve(memory_id)
        assert memory is not None
        assert memory.content == content
        assert memory.type == MemoryType.SEMANTIC
        assert memory.importance == 0.8
        assert memory.metadata['test'] is True
    
    @pytest.mark.asyncio
    async def test_deduplication(self, vault):
        """Test that duplicate memories are detected"""
        content = "Duplicate test"
        metadata = {'key': 'value'}
        
        # Store same memory twice
        id1 = await vault.store(content, MemoryType.SEMANTIC, metadata)
        id2 = await vault.store(content, MemoryType.SEMANTIC, metadata)
        
        # Should get same ID due to deduplication
        assert id1 == id2
        
        # Should only have one memory stored
        stats = await vault.get_statistics()
        assert stats['total_memories'] == 1
    
    @pytest.mark.asyncio
    async def test_working_memory_eviction(self, vault):
        """Test LRU eviction in working memory"""
        # Fill working memory beyond limit
        memory_ids = []
        for i in range(15):  # Limit is 10
            memory_id = await vault.store(
                content=f"Working memory {i}",
                memory_type=MemoryType.WORKING,
                metadata={'index': i}
            )
            memory_ids.append(memory_id)
            
            # Access some memories to affect LRU
            if i == 5:
                await vault.retrieve(memory_ids[0])  # Access first memory
        
        # Check that working memory respects limit
        assert len(vault.working_memory) <= vault.max_working_memory
        
        # First memory should still be there (was accessed)
        assert memory_ids[0] in vault.working_memory
        
        # Some middle memories should be evicted
        evicted_count = sum(1 for mid in memory_ids[1:5] 
                           if mid not in vault.working_memory)
        assert evicted_count > 0
    
    @pytest.mark.asyncio
    async def test_ghost_memory_cleanup(self, vault):
        """Test automatic cleanup of expired ghost memories"""
        # Store ghost memory
        ghost_id = await vault.store(
            content="Ghost memory",
            memory_type=MemoryType.GHOST,
            metadata={'ephemeral': True}
        )
        
        # Should be retrievable immediately
        memory = await vault.retrieve(ghost_id)
        assert memory is not None
        
        # Wait for TTL to expire
        await asyncio.sleep(3)  # TTL is 2 seconds
        
        # Trigger cleanup
        await vault._cleanup_ghost_memory()
        
        # Should be gone
        assert ghost_id not in vault.ghost_memory
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, vault):
        """Test search with various criteria"""
        # Store diverse memories
        await vault.store(
            "Important semantic memory",
            MemoryType.SEMANTIC,
            metadata={'tags': ['important', 'test']},
            importance=0.9
        )
        
        await vault.store(
            "Less important memory",
            MemoryType.EPISODIC,
            metadata={'tags': ['test']},
            importance=0.3
        )
        
        await vault.store(
            "Another semantic memory",
            MemoryType.SEMANTIC,
            metadata={'tags': ['other']},
            importance=0.5
        )
        
        # Search by type
        semantic_results = await vault.search(memory_type=MemoryType.SEMANTIC)
        assert len(semantic_results) == 2
        
        # Search by tag
        test_results = await vault.search(tags=['test'])
        assert len(test_results) == 2
        
        # Search by importance
        important_results = await vault.search(min_importance=0.7)
        assert len(important_results) == 1
        assert important_results[0].importance >= 0.7
        
        # Combined search
        combined_results = await vault.search(
            memory_type=MemoryType.SEMANTIC,
            tags=['test'],
            min_importance=0.5
        )
        assert len(combined_results) == 1
    
    @pytest.mark.asyncio
    async def test_embedding_similarity(self, vault):
        """Test finding similar memories by embedding"""
        # Create test embeddings
        embedding1 = np.random.randn(128)
        embedding2 = embedding1 + np.random.randn(128) * 0.1  # Similar
        embedding3 = np.random.randn(128)  # Different
        
        # Store memories with embeddings
        id1 = await vault.store(
            "Memory 1",
            MemoryType.SEMANTIC,
            embedding=embedding1
        )
        
        id2 = await vault.store(
            "Memory 2",
            MemoryType.SEMANTIC,
            embedding=embedding2
        )
        
        id3 = await vault.store(
            "Memory 3",
            MemoryType.SEMANTIC,
            embedding=embedding3
        )
        
        # Find similar to embedding1
        similar = await vault.find_similar(embedding1, threshold=0.5)
        
        # Should find itself and the similar one
        similar_ids = [m[0].id for m in similar]
        assert id1 in similar_ids
        assert id2 in similar_ids
        assert id3 not in similar_ids  # Too different
        
        # Check similarity scores
        for memory, score in similar:
            if memory.id == id1:
                assert score > 0.99  # Should be ~1.0 for itself
    
    @pytest.mark.asyncio
    async def test_update_memory(self, vault):
        """Test updating existing memories"""
        # Store initial memory
        memory_id = await vault.store(
            "Original content",
            MemoryType.SEMANTIC,
            metadata={'version': 1},
            importance=0.5
        )
        
        # Update it
        success = await vault.update(
            memory_id,
            content="Updated content",
            metadata={'version': 2, 'updated': True},
            importance=0.8
        )
        
        assert success
        
        # Retrieve and verify
        memory = await vault.retrieve(memory_id)
        assert memory.content == "Updated content"
        assert memory.metadata['version'] == 2
        assert memory.metadata['updated'] is True
        assert memory.importance == 0.8
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, vault):
        """Test memory deletion"""
        # Store memories in different types
        working_id = await vault.store("Working", MemoryType.WORKING)
        ghost_id = await vault.store("Ghost", MemoryType.GHOST)
        persistent_id = await vault.store("Persistent", MemoryType.SEMANTIC)
        
        # Delete each type
        assert await vault.delete(working_id)
        assert await vault.delete(ghost_id)
        assert await vault.delete(persistent_id)
        
        # Verify they're gone
        assert await vault.retrieve(working_id) is None
        assert await vault.retrieve(ghost_id) is None
        assert await vault.retrieve(persistent_id) is None
        
        # Verify removal from indices
        assert working_id not in vault.main_index
    
    @pytest.mark.asyncio
    async def test_atomic_file_operations(self, vault):
        """Test that file operations are atomic"""
        # This tests the atomic write functionality
        test_path = vault.storage_path / "test_atomic.txt"
        
        # Write with simulated interruption
        async def interrupted_write():
            # Start multiple concurrent writes
            tasks = []
            for i in range(10):
                data = f"Write attempt {i}"
                tasks.append(vault._atomic_write(test_path, data))
            
            await asyncio.gather(*tasks)
        
        await interrupted_write()
        
        # File should contain last write and be valid
        assert test_path.exists()
        content = test_path.read_text()
        assert content.startswith("Write attempt")
    
    @pytest.mark.asyncio
    async def test_streaming_export_import(self, vault):
        """Test streaming export/import functionality"""
        # Store many memories
        for i in range(100):
            await vault.store(
                f"Memory {i}",
                MemoryType.SEMANTIC,
                metadata={'index': i},
                importance=i / 100
            )
        
        # Export to file
        export_path = vault.storage_path / "export.jsonl"
        count = await vault.export_memories(export_path)
        assert count == 100
        
        # Create new vault and import
        with tempfile.TemporaryDirectory() as tmpdir2:
            vault2 = UnifiedMemoryVaultV2({'storage_path': tmpdir2})
            await vault2.initialize()
            
            imported = await vault2.import_memories(export_path)
            assert imported == 100
            
            # Verify data integrity
            stats = await vault2.get_statistics()
            assert stats['total_memories'] == 100
            
            await vault2.shutdown()
    
    @pytest.mark.asyncio
    async def test_consolidation(self, vault):
        """Test memory consolidation"""
        # Create memories with different importance and access patterns
        important_id = await vault.store(
            "Important frequently accessed",
            MemoryType.WORKING,
            importance=0.9
        )
        
        # Access it multiple times
        for _ in range(10):
            await vault.retrieve(important_id)
        
        # Create unimportant old memory
        await vault.store(
            "Unimportant old",
            MemoryType.SEMANTIC,
            importance=0.05
        )
        
        # Run consolidation
        stats = await vault.consolidate()
        
        # Important working memory should be consolidated to persistent
        assert important_id not in vault.working_memory
        memory = await vault.retrieve(important_id)
        assert memory is not None  # Still accessible from file storage
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, vault):
        """Test concurrent read/write operations"""
        async def writer(start_idx: int):
            for i in range(10):
                await vault.store(
                    f"Concurrent write {start_idx + i}",
                    MemoryType.SEMANTIC,
                    metadata={'writer': start_idx}
                )
        
        async def reader():
            results = []
            for _ in range(10):
                memories = await vault.search(max_results=5)
                results.append(len(memories))
                await asyncio.sleep(0.01)
            return results
        
        # Run concurrent operations
        tasks = [
            writer(0),
            writer(100),
            writer(200),
            reader(),
            reader()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify no data corruption
        stats = await vault.get_statistics()
        assert stats['total_memories'] == 30
    
    @pytest.mark.asyncio
    async def test_crash_recovery(self, vault):
        """Test recovery after simulated crash"""
        # Store some memories
        memory_ids = []
        for i in range(5):
            memory_id = await vault.store(
                f"Pre-crash memory {i}",
                MemoryType.SEMANTIC,
                metadata={'pre_crash': True}
            )
            memory_ids.append(memory_id)
        
        # Simulate crash by not calling shutdown
        # Just save current state
        await vault.save_all()
        
        # Create new vault instance (simulating restart)
        vault2 = UnifiedMemoryVaultV2({'storage_path': vault.storage_path})
        await vault2.initialize()
        
        # Verify memories are recovered
        for memory_id in memory_ids:
            memory = await vault2.retrieve(memory_id)
            assert memory is not None
            assert memory.metadata['pre_crash'] is True
        
        await vault2.shutdown()
    
    @pytest.mark.asyncio
    async def test_signal_handling(self, vault):
        """Test graceful shutdown on signals"""
        # Store a memory
        await vault.store("Signal test", MemoryType.SEMANTIC)
        
        # Send SIGTERM (on Unix) or simulate shutdown
        if os.name != 'nt':  # Unix/Linux/Mac
            os.kill(os.getpid(), signal.SIGTERM)
        else:  # Windows
            vault._shutdown_event.set()
        
        # Give time for signal handler
        await asyncio.sleep(0.1)
        
        # Verify shutdown was initiated
        assert vault._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_memory_decay(self, vault):
        """Test importance decay over time"""
        # Store memory with high importance
        memory_id = await vault.store(
            "Decaying memory",
            MemoryType.SEMANTIC,
            importance=1.0
        )
        
        # Manually trigger decay with old timestamp
        memory = await vault.retrieve(memory_id)
        memory.timestamp = time.time() - 86400 * 7  # 7 days old
        memory.last_accessed = memory.timestamp
        
        # Apply decay
        await vault._apply_decay()
        
        # Retrieve and check importance decreased
        memory = await vault.retrieve(memory_id)
        assert memory.importance < 1.0
    
    @pytest.mark.asyncio
    async def test_packfile_optimization(self, vault):
        """Test packfile creation for many small files"""
        # Store many memories to trigger packfile optimization
        for i in range(20):
            await vault.store(
                f"Memory for packing {i}",
                MemoryType.SEMANTIC,
                metadata={'pack_test': True}
            )
        
        # Manually trigger packfile optimization
        vault.packfile_threshold = 10  # Lower threshold for testing
        await vault._optimize_packfiles()
        
        # Verify memories are still accessible
        results = await vault.search(tags=['pack_test'])
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, vault):
        """Test performance tracking and metrics"""
        start_time = time.time()
        
        # Perform various operations
        for i in range(50):
            await vault.store(f"Perf test {i}", MemoryType.SEMANTIC)
        
        # Get statistics
        stats = await vault.get_statistics()
        
        assert stats['operation_count'] >= 50
        assert stats['uptime_seconds'] > 0
        assert 'session_id' in stats
        assert 'total_size_mb' in stats
        
        # Verify operation rate
        elapsed = time.time() - start_time
        ops_per_second = stats['operation_count'] / elapsed
        assert ops_per_second > 0


@pytest.mark.asyncio
async def test_memory_vault_integration():
    """Integration test for complete workflow"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'storage_path': tmpdir,
            'max_working_memory': 50,
            'batch_size': 10
        }
        
        vault = UnifiedMemoryVaultV2(config)
        await vault.initialize()
        
        try:
            # Simulate a complete workflow
            
            # 1. Store various types of memories
            context_id = await vault.store(
                "Current project context",
                MemoryType.WORKING,
                metadata={'project': 'TORI', 'tags': ['context', 'active']}
            )
            
            fact_id = await vault.store(
                "Python asyncio provides concurrent execution",
                MemoryType.SEMANTIC,
                metadata={'tags': ['programming', 'python']},
                importance=0.8
            )
            
            # 2. Store with embeddings
            embedding = np.random.randn(128)
            embedded_id = await vault.store(
                "Memory with embedding",
                MemoryType.SEMANTIC,
                embedding=embedding,
                metadata={'tags': ['ml', 'embedding']}
            )
            
            # 3. Search and retrieve
            python_memories = await vault.search(tags=['python'])
            assert len(python_memories) > 0
            
            # 4. Find similar by embedding
            similar = await vault.find_similar(embedding, threshold=0.5)
            assert len(similar) > 0
            
            # 5. Update memory
            await vault.update(
                context_id,
                metadata={'project': 'TORI', 'phase': 'testing', 'tags': ['context', 'active', 'test']}
            )
            
            # 6. Stream all memories
            all_memories = []
            async for memory in vault.stream_all():
                all_memories.append(memory)
            assert len(all_memories) >= 3
            
            # 7. Export and verify
            export_path = Path(tmpdir) / "integration_export.jsonl"
            exported = await vault.export_memories(export_path)
            assert exported >= 3
            
            # 8. Get final statistics
            stats = await vault.get_statistics()
            print(f"Integration test stats: {stats}")
            
            # 9. Graceful shutdown
            await vault.shutdown()
            
            print("✅ Integration test passed!")
            
        except Exception as e:
            await vault.shutdown()
            raise e


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_memory_vault_integration())
