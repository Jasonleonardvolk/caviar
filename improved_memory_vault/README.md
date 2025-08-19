# Memory Vault V2 - Production-Ready Async Memory System

A fully asynchronous, file-based memory storage system designed for production use with TORI/KHA. This implementation addresses all critical issues identified in the production readiness review.

## 🚀 Key Improvements from V1

### 1. **Fully Async Architecture**
- Eliminated all `threading.Thread` and `asyncio.run()` calls
- Pure async/await design for better integration with modern frameworks
- No more event loop conflicts

### 2. **Fine-Grained Concurrency**
- Replaced global `RLock` with async read-write locks
- Separate locks for indices, working memory, and ghost memory
- Dramatically improved concurrent access performance

### 3. **Stable Deduplication**
- Deterministic hashing using sorted JSON serialization
- Consistent duplicate detection across restarts
- SHA-256 based content addressing

### 4. **Atomic File Operations**
- All writes use temp file + atomic rename pattern
- Explicit fsync for durability
- Crash-safe at every operation

### 5. **Optimized Storage**
- Embeddings stored as compressed numpy arrays
- MessagePack for efficient serialization
- Packfile support for consolidating small files
- Streaming APIs to handle large datasets

### 6. **Enhanced Observability**
- Dual-mode logging (live NDJSON + snapshots)
- Comprehensive metrics and statistics
- Session tracking and crash recovery
- Signal handling for graceful shutdown

## 📋 Features

- **Multiple Memory Types**: Episodic, Semantic, Procedural, Working, Ghost, Soliton
- **Embedding Support**: Vector similarity search with numpy
- **Tagging System**: Flexible metadata and tag-based retrieval
- **Importance Decay**: Automatic importance adjustment based on access patterns
- **Deduplication**: Content-based deduplication to prevent redundant storage
- **Export/Import**: JSONL format for data portability
- **Crash Recovery**: Automatic recovery from unexpected shutdowns

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository>
cd improved_memory_vault

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Basic Usage

```python
import asyncio
from memory_vault_v2 import UnifiedMemoryVaultV2, MemoryType

async def example():
    # Initialize vault
    vault = UnifiedMemoryVaultV2({
        'storage_path': 'data/my_memories',
        'max_working_memory': 100,
        'batch_size': 50
    })
    await vault.initialize()
    
    # Store a memory
    memory_id = await vault.store(
        content="Important information",
        memory_type=MemoryType.SEMANTIC,
        metadata={'tags': ['important', 'example']},
        importance=0.9
    )
    
    # Retrieve a memory
    memory = await vault.retrieve(memory_id)
    print(f"Retrieved: {memory.content}")
    
    # Search memories
    results = await vault.search(
        tags=['important'],
        min_importance=0.5,
        max_results=10
    )
    
    # Find similar memories by embedding
    import numpy as np
    embedding = np.random.randn(128)
    similar = await vault.find_similar(embedding, threshold=0.7)
    
    # Stream all memories (memory efficient)
    async for memory in vault.stream_all():
        print(f"Memory: {memory.id}")
    
    # Graceful shutdown
    await vault.shutdown()

# Run the example
asyncio.run(example())
```

## 🔄 Migration from V1

```bash
# Migrate existing V1 vault to V2
python migrate.py /path/to/v1/vault /path/to/v2/vault

# Verify migration
python migrate.py /path/to/v1/vault /path/to/v2/vault --verify-only

# Migrate and cleanup old data
python migrate.py /path/to/v1/vault /path/to/v2/vault --cleanup
```

## 📊 Performance Benchmarks

Run the comprehensive benchmark suite:

```bash
python benchmark.py
```

Typical performance metrics:
- **Store**: 500-1000 ops/sec
- **Retrieve**: 2000-5000 ops/sec  
- **Search**: 100-500 ops/sec
- **Concurrent Access**: 10x improvement over V1

## 🧪 Testing

```bash
# Run all tests
pytest test_memory_vault_v2.py -v

# Run with coverage
pytest test_memory_vault_v2.py --cov=memory_vault_v2 --cov-report=html

# Run specific test
pytest test_memory_vault_v2.py::TestMemoryVaultV2::test_deduplication -v
```

## 📁 Storage Structure

```
memory_vault_v2/
├── memories/          # Individual memory files (msgpack)
├── embeddings/        # Compressed numpy arrays
├── packfiles/         # Consolidated memory packs
├── index/             # Msgpack indices
│   ├── main_index.msgpack
│   ├── type_index.msgpack
│   └── tag_index.msgpack
└── logs/              # Operational logs
    ├── vault_live.jsonl     # Live append-only log
    ├── vault_snapshot.json  # Periodic snapshots
    └── session_*.jsonl      # Per-session logs
```

## ⚙️ Configuration Options

```python
config = {
    'storage_path': 'data/memory_vault_v2',  # Base storage directory
    'max_working_memory': 100,               # Max items in working memory
    'ghost_memory_ttl': 3600,                # Ghost memory TTL (seconds)
    'decay_enabled': True,                   # Enable importance decay
    'packfile_threshold': 1000,              # Files before packing
    'batch_size': 100                        # Batch size for operations
}
```

## 🔍 Advanced Features

### Custom Memory Types

```python
# Store procedural memory with custom decay
await vault.store(
    content={"action": "compile", "command": "gcc -O2"},
    memory_type=MemoryType.PROCEDURAL,
    metadata={
        'executable': True,
        'tags': ['build', 'commands']
    }
)
```

### Embedding-based Retrieval

```python
# Store with embedding
embedding = model.encode("semantic content")
await vault.store(
    content="Semantic memory with embedding",
    memory_type=MemoryType.SEMANTIC,
    embedding=embedding
)

# Find similar
similar_embedding = model.encode("related content")
similar_memories = await vault.find_similar(
    similar_embedding,
    threshold=0.8,
    max_results=5
)
```

### Bulk Operations

```python
# Export all memories
await vault.export_memories(Path("backup.jsonl"))

# Import from backup
imported = await vault.import_memories(Path("backup.jsonl"))

# Consolidate and optimize
stats = await vault.consolidate()
```

## 🚨 Error Handling

The vault includes comprehensive error handling:

```python
try:
    memory = await vault.retrieve("invalid_id")
except Exception as e:
    logger.error(f"Retrieval failed: {e}")

# Automatic recovery on restart
vault = UnifiedMemoryVaultV2(config)
await vault.initialize()  # Automatically recovers from crashes
```

## 📈 Monitoring

Access real-time statistics:

```python
stats = await vault.get_statistics()
print(f"Total memories: {stats['total_memories']}")
print(f"Storage size: {stats['total_size_mb']} MB")
print(f"Operations: {stats['operation_count']}")

status = await vault.get_status()
print(f"Session: {status['session_id']}")
print(f"Uptime: {status['uptime']} seconds")
```

## 🤝 Contributing

1. Ensure all tests pass
2. Add tests for new features
3. Run benchmarks to verify performance
4. Update documentation

## 📝 License

[Your License Here]

## 🔗 Related Projects

- TORI/KHA Memory System
- Enhanced Launcher
- Unified ID Generator

## 📞 Support

For issues and questions:
- Check existing issues
- Run diagnostics: `python benchmark.py`
- Review logs in `logs/` directory

---

Built with ❤️ for production-grade AI memory systems.
