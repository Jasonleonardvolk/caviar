"""
TORI/KHA Unified Memory Vault V2 - Production-Ready Implementation
Fully async, atomic operations, optimized for scale
"""

import json
import pickle
import hashlib
import time
import logging
import asyncio
import os
import signal
import atexit
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from enum import Enum
import numpy as np
import gzip
import shutil
import aiofiles
import aiofiles.os
from contextlib import asynccontextmanager
import msgpack
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    GHOST = "ghost"
    SOLITON = "soliton"

@dataclass
class MemoryEntry:
    """Single memory entry with optimized serialization"""
    id: str
    type: MemoryType
    content: Any
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: Optional[float] = None
    decay_rate: float = 0.0
    importance: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['type'] = self.type.value
        # Don't include embedding in main dict - store separately
        data.pop('embedding', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> 'MemoryEntry':
        """Create from dictionary"""
        data = data.copy()
        data['type'] = MemoryType(data['type'])
        data['embedding'] = embedding
        return cls(**data)

class AsyncReadWriteLock:
    """Async read-write lock for better concurrency"""
    def __init__(self):
        self._read_count = 0
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._read_condition = asyncio.Condition(self._read_lock)
    
    @asynccontextmanager
    async def read_lock(self):
        """Acquire read lock - multiple readers allowed"""
        async with self._read_lock:
            while self._write_lock.locked():
                await self._read_condition.wait()
            self._read_count += 1
        try:
            yield
        finally:
            async with self._read_lock:
                self._read_count -= 1
                if self._read_count == 0:
                    self._read_condition.notify_all()
    
    @asynccontextmanager
    async def write_lock(self):
        """Acquire write lock - exclusive access"""
        await self._write_lock.acquire()
        async with self._read_lock:
            while self._read_count > 0:
                await self._read_condition.wait()
        try:
            yield
        finally:
            self._write_lock.release()
            async with self._read_lock:
                self._read_condition.notify_all()

class UnifiedMemoryVaultV2:
    """
    Production-ready unified memory system
    - Fully async architecture
    - Atomic file operations
    - Optimized storage format
    - Fine-grained locking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unified memory vault"""
        self.config = config or {}
        
        # Storage configuration
        self.storage_path = Path(self.config.get('storage_path', 'data/memory_vault_v2'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create storage subdirectories
        self.memories_dir = self.storage_path / 'memories'
        self.index_dir = self.storage_path / 'index'
        self.embeddings_dir = self.storage_path / 'embeddings'
        self.packfiles_dir = self.storage_path / 'packfiles'
        self.logs_dir = self.storage_path / 'logs'
        
        for directory in [self.memories_dir, self.index_dir, self.embeddings_dir, 
                         self.packfiles_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Logging configuration
        self.live_log_path = self.logs_dir / 'vault_live.jsonl'
        self.snapshot_path = self.logs_dir / 'vault_snapshot.json'
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_path = self.logs_dir / f'session_{self.session_id}.jsonl'
        
        # Deduplication tracking
        self.seen_hashes_file = self.logs_dir / 'seen_hashes.msgpack'
        self.seen_hashes: Dict[str, float] = {}
        
        # Memory configuration
        self.max_working_memory = self.config.get('max_working_memory', 100)
        self.ghost_memory_ttl = self.config.get('ghost_memory_ttl', 3600)
        self.decay_enabled = self.config.get('decay_enabled', True)
        self.packfile_threshold = self.config.get('packfile_threshold', 1000)
        self.batch_size = self.config.get('batch_size', 100)
        
        # In-memory caches
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.ghost_memory: Dict[str, MemoryEntry] = {}
        self.access_cache: Dict[str, float] = defaultdict(float)
        
        # Index files
        self.main_index_file = self.index_dir / 'main_index.msgpack'
        self.type_index_file = self.index_dir / 'type_index.msgpack'
        self.tag_index_file = self.index_dir / 'tag_index.msgpack'
        
        # Indices
        self.main_index: Dict[str, str] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}
        
        # Locks for fine-grained concurrency
        self.index_lock = AsyncReadWriteLock()
        self.working_lock = asyncio.Lock()
        self.ghost_lock = asyncio.Lock()
        self.log_lock = asyncio.Lock()
        
        # Log buffer for crash safety
        self._log_buffer: List[str] = []
        self._last_flush = time.time()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Metrics
        self._start_time = time.time()
        self._operation_count = 0
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._setup_shutdown_handlers()
        
        logger.info(f"UnifiedMemoryVaultV2 initialized at {self.storage_path}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _setup_shutdown_handlers(self):
        """Setup graceful shutdown handlers"""
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
        
        # Register atexit handler
        atexit.register(lambda: asyncio.run(self._emergency_flush()))
    
    async def _emergency_flush(self):
        """Emergency flush on unexpected exit"""
        try:
            await self._flush_log_buffer(force=True)
            await self._write_snapshot()
        except Exception as e:
            logger.error(f"Emergency flush failed: {e}")
    
    async def initialize(self):
        """Async initialization - must be called after creation"""
        # Load indices
        await self._load_indices()
        
        # Load seen hashes
        await self._load_seen_hashes()
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Write session start marker
        await self._append_to_live_log({
            'session_id': self.session_id,
            'timestamp': self._start_time,
            'action': 'session_start',
            'pid': os.getpid()
        })
    
    async def _load_seen_hashes(self):
        """Load seen hashes using msgpack for efficiency"""
        if self.seen_hashes_file.exists():
            try:
                async with aiofiles.open(self.seen_hashes_file, 'rb') as f:
                    data = await f.read()
                    self.seen_hashes = msgpack.unpackb(data, raw=False)
            except Exception as e:
                logger.error(f"Failed to load seen hashes: {e}")
                self.seen_hashes = {}
    
    async def _save_seen_hashes(self):
        """Save seen hashes atomically"""
        try:
            data = msgpack.packb(self.seen_hashes)
            await self._atomic_write(self.seen_hashes_file, data, binary=True)
        except Exception as e:
            logger.error(f"Failed to save seen hashes: {e}")
    
    def _calculate_entry_hash(self, entry: MemoryEntry) -> str:
        """Calculate stable SHA-256 hash for deduplication"""
        # Create deterministic hash from entry content
        entry_dict = {
            'type': entry.type.value,
            'content': self._stable_json(entry.content),
            'metadata': self._stable_json(entry.metadata)
        }
        entry_str = json.dumps(entry_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(entry_str.encode()).hexdigest()
    
    def _stable_json(self, obj: Any) -> str:
        """Convert object to stable JSON string"""
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, sort_keys=True, separators=(',', ':'))
        return str(obj)
    
    async def _atomic_write(self, path: Path, data: Union[str, bytes], binary: bool = False):
        """Atomic file write with fsync"""
        temp_path = path.with_suffix('.tmp')
        
        mode = 'wb' if binary else 'w'
        encoding = None if binary else 'utf-8'
        
        async with aiofiles.open(temp_path, mode, encoding=encoding) as f:
            await f.write(data)
            await f.flush()
            await aiofiles.os.fsync(f.fileno())
        
        # Atomic rename
        await aiofiles.os.rename(temp_path, path)
    
    async def _append_to_live_log(self, log_entry: Dict[str, Any]):
        """Append to live log with buffering"""
        async with self.log_lock:
            line = json.dumps(log_entry) + '\n'
            self._log_buffer.append(line)
            
            # Flush if buffer is large or time elapsed
            if len(self._log_buffer) >= 10 or (time.time() - self._last_flush) > 5:
                await self._flush_log_buffer()
    
    async def _flush_log_buffer(self, force: bool = False):
        """Flush log buffer to disk"""
        if not self._log_buffer and not force:
            return
        
        async with self.log_lock:
            if self._log_buffer:
                async with aiofiles.open(self.live_log_path, 'a', encoding='utf-8') as f:
                    await f.writelines(self._log_buffer)
                    await f.flush()
                    await aiofiles.os.fsync(f.fileno())
                
                # Also write to session log
                async with aiofiles.open(self.session_log_path, 'a', encoding='utf-8') as f:
                    await f.writelines(self._log_buffer)
                    await f.flush()
                
                self._log_buffer.clear()
                self._last_flush = time.time()
    
    async def _write_snapshot(self):
        """Write complete memory snapshot atomically"""
        try:
            snapshot_data = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'statistics': await self.get_statistics(),
                'memories': {
                    'working': [m.to_dict() for m in self.working_memory.values()],
                    'ghost': [m.to_dict() for m in self.ghost_memory.values()],
                    'persistent_count': len(self.main_index)
                },
                'indices': {
                    'main_index_size': len(self.main_index),
                    'type_index_size': len(self.type_index),
                    'tag_index_size': len(self.tag_index)
                },
                'seen_hashes_count': len(self.seen_hashes)
            }
            
            data = json.dumps(snapshot_data, indent=2)
            await self._atomic_write(self.snapshot_path, data)
            
            logger.debug(f"Snapshot written: {self.snapshot_path}")
            
        except Exception as e:
            logger.error(f"Failed to write snapshot: {e}")
    
    async def _load_indices(self):
        """Load indices using msgpack for efficiency"""
        try:
            # Load main index
            if self.main_index_file.exists():
                async with aiofiles.open(self.main_index_file, 'rb') as f:
                    data = await f.read()
                    self.main_index = msgpack.unpackb(data, raw=False)
            
            # Load type index
            if self.type_index_file.exists():
                async with aiofiles.open(self.type_index_file, 'rb') as f:
                    data = await f.read()
                    self.type_index = msgpack.unpackb(data, raw=False)
            
            # Load tag index
            if self.tag_index_file.exists():
                async with aiofiles.open(self.tag_index_file, 'rb') as f:
                    data = await f.read()
                    self.tag_index = msgpack.unpackb(data, raw=False)
            
            logger.info(f"Loaded indices: {len(self.main_index)} memories indexed")
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            self.main_index = {}
            self.type_index = {}
            self.tag_index = {}
    
    async def _save_indices(self):
        """Save indices atomically"""
        try:
            # Save main index
            data = msgpack.packb(self.main_index)
            await self._atomic_write(self.main_index_file, data, binary=True)
            
            # Save type index
            data = msgpack.packb(self.type_index)
            await self._atomic_write(self.type_index_file, data, binary=True)
            
            # Save tag index
            data = msgpack.packb(self.tag_index)
            await self._atomic_write(self.tag_index_file, data, binary=True)
            
        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Decay task
        if self.decay_enabled:
            self._background_tasks.append(
                asyncio.create_task(self._decay_loop())
            )
        
        # Ghost memory cleanup
        self._background_tasks.append(
            asyncio.create_task(self._ghost_cleanup_loop())
        )
        
        # Index maintenance
        self._background_tasks.append(
            asyncio.create_task(self._index_maintenance_loop())
        )
        
        # Log flusher
        self._background_tasks.append(
            asyncio.create_task(self._log_flush_loop())
        )
        
        # Packfile optimizer
        self._background_tasks.append(
            asyncio.create_task(self._packfile_loop())
        )
    
    async def _decay_loop(self):
        """Background task to decay memories"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._apply_decay()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decay loop error: {e}")
    
    async def _ghost_cleanup_loop(self):
        """Background task to clean expired ghost memories"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # 1 minute
                await self._cleanup_ghost_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ghost cleanup error: {e}")
    
    async def _index_maintenance_loop(self):
        """Background task to maintain indices"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # 10 minutes
                await self._save_indices()
                await self._save_seen_hashes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Index maintenance error: {e}")
    
    async def _log_flush_loop(self):
        """Background task to flush logs periodically"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)
                await self._flush_log_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Log flush error: {e}")
    
    async def _packfile_loop(self):
        """Background task to pack small files into larger ones"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self._optimize_packfiles()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Packfile optimization error: {e}")
    
    async def store(
        self,
        content: Any,
        memory_type: Union[MemoryType, str],
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a memory entry with deduplication"""
        # Convert string to MemoryType
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        
        # Generate ID
        memory_id = self._generate_id(content, metadata)
        
        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            type=memory_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time(),
            importance=importance
        )
        
        # Check for duplicates
        entry_hash = self._calculate_entry_hash(entry)
        if entry_hash in self.seen_hashes:
            logger.debug(f"Duplicate entry detected (hash: {entry_hash[:8]}...)")
            self.seen_hashes[entry_hash] = time.time()
            await self._append_to_live_log({
                'session_id': self.session_id,
                'timestamp': time.time(),
                'action': 'duplicate',
                'entry_id': memory_id,
                'hash': entry_hash[:8]
            })
            return memory_id
        
        # Mark as seen
        self.seen_hashes[entry_hash] = time.time()
        
        # Store based on type
        if memory_type == MemoryType.WORKING:
            async with self.working_lock:
                self.working_memory[memory_id] = entry
                
                # Enforce size limit with LRU eviction
                if len(self.working_memory) > self.max_working_memory:
                    # Find least recently used
                    lru_id = min(self.working_memory.keys(), 
                               key=lambda k: self.access_cache.get(k, 0))
                    evicted = self.working_memory.pop(lru_id)
                    
                    await self._append_to_live_log({
                        'session_id': self.session_id,
                        'timestamp': time.time(),
                        'action': 'evicted',
                        'entry_id': lru_id
                    })
        
        elif memory_type == MemoryType.GHOST:
            async with self.ghost_lock:
                self.ghost_memory[memory_id] = entry
        
        else:
            # Store to persistent storage
            await self._save_memory_to_file(entry)
            
            # Update indices
            async with self.index_lock.write_lock():
                self._update_indices(memory_id, memory_type, tags)
        
        # Log the store operation
        await self._append_to_live_log({
            'session_id': self.session_id,
            'timestamp': time.time(),
            'action': 'store',
            'entry': entry.to_dict(),
            'checksum': entry_hash
        })
        
        # Increment operation count
        self._operation_count += 1
        
        # Periodic snapshot
        if self._operation_count % 100 == 0:
            asyncio.create_task(self._write_snapshot())
        
        logger.debug(f"Stored memory {memory_id} of type {memory_type.value}")
        return memory_id
    
    async def _save_memory_to_file(self, entry: MemoryEntry):
        """Save memory entry to file with optimized storage"""
        # Save embedding separately if present
        if entry.embedding is not None:
            embedding_path = self.embeddings_dir / f"{entry.id}.npy"
            # Save as compressed numpy array
            np.savez_compressed(embedding_path, embedding=entry.embedding)
        
        # Save memory metadata
        memory_data = entry.to_dict()
        
        # Check if we should use packfiles
        file_count = len(list(self.memories_dir.glob('*.msgpack')))
        if file_count < self.packfile_threshold:
            # Individual file
            file_path = self.memories_dir / f"{entry.id}.msgpack"
            data = msgpack.packb(memory_data)
            await self._atomic_write(file_path, data, binary=True)
            
            async with self.index_lock.write_lock():
                self.main_index[entry.id] = str(file_path.relative_to(self.storage_path))
        else:
            # Add to packfile queue
            await self._queue_for_packfile(entry.id, memory_data)
    
    async def _queue_for_packfile(self, memory_id: str, memory_data: Dict[str, Any]):
        """Queue memory for packfile storage"""
        # For now, still write individual file
        # Packfile optimization will batch these later
        file_path = self.memories_dir / f"{memory_id}.msgpack"
        data = msgpack.packb(memory_data)
        await self._atomic_write(file_path, data, binary=True)
        
        async with self.index_lock.write_lock():
            self.main_index[memory_id] = str(file_path.relative_to(self.storage_path))
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry"""
        # Check working memory
        async with self.working_lock:
            if memory_id in self.working_memory:
                entry = self.working_memory[memory_id]
                self._update_access(entry)
                return entry
        
        # Check ghost memory
        async with self.ghost_lock:
            if memory_id in self.ghost_memory:
                entry = self.ghost_memory[memory_id]
                self._update_access(entry)
                return entry
        
        # Load from file storage
        async with self.index_lock.read_lock():
            if memory_id not in self.main_index:
                return None
            file_path = self.storage_path / self.main_index[memory_id]
        
        return await self._load_memory_from_file(memory_id, file_path)
    
    async def _load_memory_from_file(self, memory_id: str, file_path: Path) -> Optional[MemoryEntry]:
        """Load memory entry from file"""
        try:
            if not file_path.exists():
                # Clean up stale index entry
                async with self.index_lock.write_lock():
                    if memory_id in self.main_index:
                        del self.main_index[memory_id]
                return None
            
            # Load memory metadata
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                memory_data = msgpack.unpackb(data, raw=False)
            
            # Load embedding if exists
            embedding = None
            embedding_path = self.embeddings_dir / f"{memory_id}.npy.npz"
            if embedding_path.exists():
                embedding_data = np.load(embedding_path)
                embedding = embedding_data['embedding']
            
            # Create memory entry
            entry = MemoryEntry.from_dict(memory_data, embedding=embedding)
            
            # Update access
            self._update_access(entry)
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to load memory {memory_id}: {e}")
            return None
    
    async def search(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        max_results: int = 100,
        include_embeddings: bool = False
    ) -> List[MemoryEntry]:
        """Search memories with various criteria"""
        results = []
        
        # Search in-memory stores
        async with self.working_lock:
            for memory in self.working_memory.values():
                if self._matches_criteria(memory, memory_type, tags, min_importance):
                    results.append(memory)
        
        async with self.ghost_lock:
            for memory in self.ghost_memory.values():
                if self._matches_criteria(memory, memory_type, tags, min_importance):
                    results.append(memory)
        
        # Get candidate IDs from indices
        async with self.index_lock.read_lock():
            candidate_ids = await self._get_candidate_ids(memory_type, tags)
        
        # Load candidates in batches to avoid memory explosion
        for batch_start in range(0, len(candidate_ids), self.batch_size):
            if len(results) >= max_results:
                break
            
            batch_ids = candidate_ids[batch_start:batch_start + self.batch_size]
            batch_results = await asyncio.gather(*[
                self.retrieve(memory_id) for memory_id in batch_ids
            ])
            
            for memory in batch_results:
                if memory and self._matches_criteria(memory, memory_type, tags, min_importance):
                    results.append(memory)
                    if len(results) >= max_results:
                        break
        
        # Sort by importance and recency
        results.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return results[:max_results]
    
    async def _get_candidate_ids(
        self,
        memory_type: Optional[MemoryType],
        tags: Optional[List[str]]
    ) -> List[str]:
        """Get candidate memory IDs from indices"""
        if memory_type:
            candidates = set(self.type_index.get(memory_type.value, []))
        else:
            candidates = set(self.main_index.keys())
        
        if tags:
            tag_candidates = set()
            for tag in tags:
                tag_candidates.update(self.tag_index.get(tag, []))
            candidates &= tag_candidates
        
        return list(candidates)
    
    async def find_similar(
        self,
        embedding: np.ndarray,
        memory_type: Optional[MemoryType] = None,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[MemoryEntry, float]]:
        """Find memories with similar embeddings using streaming"""
        similar_memories = []
        
        # Helper to process memory
        async def check_similarity(memory: MemoryEntry) -> Optional[Tuple[MemoryEntry, float]]:
            if memory.embedding is not None:
                if memory_type is None or memory.type == memory_type:
                    similarity = self._cosine_similarity(embedding, memory.embedding)
                    if similarity >= threshold:
                        return (memory, similarity)
            return None
        
        # Check in-memory stores
        async with self.working_lock:
            for memory in self.working_memory.values():
                result = await check_similarity(memory)
                if result:
                    similar_memories.append(result)
        
        async with self.ghost_lock:
            for memory in self.ghost_memory.values():
                result = await check_similarity(memory)
                if result:
                    similar_memories.append(result)
        
        # Stream through file storage
        async with self.index_lock.read_lock():
            # Only check memories that have embeddings
            embedding_files = set(p.stem for p in self.embeddings_dir.glob('*.npy.npz'))
            candidate_ids = [mid for mid in self.main_index.keys() if mid in embedding_files]
        
        # Process in batches
        for batch_start in range(0, len(candidate_ids), self.batch_size):
            if len(similar_memories) >= max_results * 2:
                break
            
            batch_ids = candidate_ids[batch_start:batch_start + self.batch_size]
            batch_memories = await asyncio.gather(*[
                self.retrieve(memory_id) for memory_id in batch_ids
            ])
            
            for memory in batch_memories:
                if memory:
                    result = await check_similarity(memory)
                    if result:
                        similar_memories.append(result)
        
        # Sort by similarity and return top results
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        return similar_memories[:max_results]
    
    async def update(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update an existing memory"""
        # Find and update memory
        memory = await self.retrieve(memory_id)
        if not memory:
            return False
        
        # Update fields
        if content is not None:
            memory.content = content
        if metadata is not None:
            memory.metadata.update(metadata)
        if importance is not None:
            memory.importance = importance
        
        # Update in appropriate store
        if memory.type == MemoryType.WORKING:
            async with self.working_lock:
                self.working_memory[memory_id] = memory
        elif memory.type == MemoryType.GHOST:
            async with self.ghost_lock:
                self.ghost_memory[memory_id] = memory
        else:
            await self._save_memory_to_file(memory)
            if tags is not None:
                async with self.index_lock.write_lock():
                    self._update_indices(memory_id, memory.type, tags)
        
        # Log update
        await self._append_to_live_log({
            'session_id': self.session_id,
            'timestamp': time.time(),
            'action': 'update',
            'entry_id': memory_id
        })
        
        return True
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        deleted = False
        
        # Check working memory
        async with self.working_lock:
            if memory_id in self.working_memory:
                del self.working_memory[memory_id]
                deleted = True
        
        # Check ghost memory
        async with self.ghost_lock:
            if memory_id in self.ghost_memory:
                del self.ghost_memory[memory_id]
                deleted = True
        
        # Delete from file storage
        if not deleted:
            deleted = await self._delete_from_files(memory_id)
        
        if deleted:
            # Log deletion
            await self._append_to_live_log({
                'session_id': self.session_id,
                'timestamp': time.time(),
                'action': 'delete',
                'entry_id': memory_id
            })
        
        return deleted
    
    async def _delete_from_files(self, memory_id: str) -> bool:
        """Delete memory from file storage"""
        try:
            async with self.index_lock.write_lock():
                if memory_id not in self.main_index:
                    return False
                
                file_path = self.storage_path / self.main_index[memory_id]
                del self.main_index[memory_id]
                
                # Remove from type index
                for memory_type, ids in self.type_index.items():
                    if memory_id in ids:
                        ids.remove(memory_id)
                        break
                
                # Remove from tag index
                for tag, ids in self.tag_index.items():
                    if memory_id in ids:
                        ids.remove(memory_id)
            
            # Remove files
            if file_path.exists():
                await aiofiles.os.remove(file_path)
            
            # Remove embedding if exists
            embedding_path = self.embeddings_dir / f"{memory_id}.npy.npz"
            if embedding_path.exists():
                await aiofiles.os.remove(embedding_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def stream_all(self, memory_type: Optional[MemoryType] = None) -> AsyncIterator[MemoryEntry]:
        """Stream all memories without loading everything into memory"""
        # Stream from in-memory stores
        async with self.working_lock:
            for memory in self.working_memory.values():
                if memory_type is None or memory.type == memory_type:
                    yield memory
        
        async with self.ghost_lock:
            for memory in self.ghost_memory.values():
                if memory_type is None or memory.type == memory_type:
                    yield memory
        
        # Stream from file storage
        async with self.index_lock.read_lock():
            if memory_type:
                memory_ids = self.type_index.get(memory_type.value, [])
            else:
                memory_ids = list(self.main_index.keys())
        
        # Process in batches
        for batch_start in range(0, len(memory_ids), self.batch_size):
            batch_ids = memory_ids[batch_start:batch_start + self.batch_size]
            batch_memories = await asyncio.gather(*[
                self.retrieve(memory_id) for memory_id in batch_ids
            ])
            
            for memory in batch_memories:
                if memory:
                    yield memory
    
    async def export_memories(self, export_path: Path, memory_type: Optional[MemoryType] = None) -> int:
        """Export memories to JSONL file using streaming"""
        count = 0
        
        async with aiofiles.open(export_path, 'w', encoding='utf-8') as f:
            async for memory in self.stream_all(memory_type):
                line = json.dumps(memory.to_dict()) + '\n'
                await f.write(line)
                count += 1
                
                # Flush periodically
                if count % 100 == 0:
                    await f.flush()
        
        logger.info(f"Exported {count} memories to {export_path}")
        return count
    
    async def import_memories(self, import_path: Path) -> int:
        """Import memories from JSONL file"""
        imported = 0
        
        async with aiofiles.open(import_path, 'r', encoding='utf-8') as f:
            async for line in f:
                try:
                    memory_data = json.loads(line.strip())
                    memory = MemoryEntry.from_dict(memory_data)
                    
                    await self.store(
                        content=memory.content,
                        memory_type=memory.type,
                        metadata=memory.metadata,
                        embedding=memory.embedding,
                        importance=memory.importance
                    )
                    imported += 1
                    
                    # Log progress
                    if imported % 100 == 0:
                        logger.info(f"Imported {imported} memories...")
                        
                except Exception as e:
                    logger.error(f"Failed to import memory: {e}")
        
        logger.info(f"Imported {imported} memories from {import_path}")
        return imported
    
    async def consolidate(self) -> Dict[str, Any]:
        """Consolidate and optimize memory storage"""
        stats = {
            'working_memory_size': len(self.working_memory),
            'ghost_memory_size': len(self.ghost_memory),
            'file_storage_size': len(self.main_index),
            'consolidated': 0,
            'deleted': 0
        }
        
        # Move important working memories to persistent storage
        to_consolidate = []
        async with self.working_lock:
            for memory_id, memory in self.working_memory.items():
                if memory.importance > 0.7 and memory.access_count > 5:
                    to_consolidate.append((memory_id, memory))
        
        for memory_id, memory in to_consolidate:
            await self._save_memory_to_file(memory)
            async with self.index_lock.write_lock():
                self._update_indices(memory_id, memory.type, memory.metadata.get('tags'))
            async with self.working_lock:
                del self.working_memory[memory_id]
            stats['consolidated'] += 1
        
        # Clean up old memories
        current_time = time.time()
        to_delete = []
        
        async for memory in self.stream_all():
            age_days = (current_time - memory.timestamp) / 86400
            if memory.importance < 0.1 and age_days > 30:
                to_delete.append(memory.id)
        
        # Delete in batches
        for memory_id in to_delete:
            if await self.delete(memory_id):
                stats['deleted'] += 1
        
        # Save indices
        await self._save_indices()
        
        logger.info(f"Memory consolidation complete: {stats}")
        return stats
    
    async def _optimize_packfiles(self):
        """Optimize storage by packing small files"""
        # Count individual files
        individual_files = list(self.memories_dir.glob('*.msgpack'))
        
        if len(individual_files) < self.packfile_threshold:
            return
        
        logger.info(f"Starting packfile optimization for {len(individual_files)} files")
        
        # Create new packfile
        packfile_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        packfile_path = self.packfiles_dir / f"pack_{packfile_id}.msgpack"
        packfile_index_path = self.packfiles_dir / f"pack_{packfile_id}_index.msgpack"
        
        pack_data = {}
        pack_index = {}
        
        # Pack files in batches
        for i in range(0, len(individual_files), self.batch_size):
            batch_files = individual_files[i:i + self.batch_size]
            
            for file_path in batch_files:
                memory_id = file_path.stem
                
                async with aiofiles.open(file_path, 'rb') as f:
                    data = await f.read()
                    memory_data = msgpack.unpackb(data, raw=False)
                
                pack_data[memory_id] = memory_data
                pack_index[memory_id] = len(pack_data) - 1
                
                # Update main index
                async with self.index_lock.write_lock():
                    self.main_index[memory_id] = f"packfiles/pack_{packfile_id}.msgpack#{memory_id}"
        
        # Write packfile
        pack_bytes = msgpack.packb(pack_data)
        await self._atomic_write(packfile_path, pack_bytes, binary=True)
        
        # Write packfile index
        index_bytes = msgpack.packb(pack_index)
        await self._atomic_write(packfile_index_path, index_bytes, binary=True)
        
        # Remove individual files
        for file_path in individual_files:
            await aiofiles.os.remove(file_path)
        
        logger.info(f"Packed {len(individual_files)} files into {packfile_path}")
    
    def _generate_id(self, content: Any, metadata: Optional[Dict[str, Any]]) -> str:
        """Generate unique ID for memory"""
        content_str = self._stable_json(content)
        metadata_str = self._stable_json(metadata or {})
        combined = f"{content_str}:{metadata_str}:{time.time()}"
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _update_access(self, memory: MemoryEntry):
        """Update access statistics"""
        memory.access_count += 1
        memory.last_accessed = time.time()
        self.access_cache[memory.id] = time.time()
    
    def _update_indices(self, memory_id: str, memory_type: MemoryType, tags: Optional[List[str]]):
        """Update indices (must be called within write lock)"""
        # Update type index
        type_name = memory_type.value
        if type_name not in self.type_index:
            self.type_index[type_name] = []
        if memory_id not in self.type_index[type_name]:
            self.type_index[type_name].append(memory_id)
        
        # Update tag index
        if tags:
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                if memory_id not in self.tag_index[tag]:
                    self.tag_index[tag].append(memory_id)
    
    def _matches_criteria(
        self,
        memory: MemoryEntry,
        memory_type: Optional[MemoryType],
        tags: Optional[List[str]],
        min_importance: float
    ) -> bool:
        """Check if memory matches search criteria"""
        if memory_type and memory.type != memory_type:
            return False
        
        if memory.importance < min_importance:
            return False
        
        if tags:
            memory_tags = memory.metadata.get('tags', [])
            if not any(tag in memory_tags for tag in tags):
                return False
        
        return True
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _apply_decay(self):
        """Apply decay to memories based on access patterns"""
        current_time = time.time()
        
        # Process in batches to avoid loading everything
        async with self.index_lock.read_lock():
            memory_ids = list(self.main_index.keys())
        
        for batch_start in range(0, len(memory_ids), self.batch_size):
            batch_ids = memory_ids[batch_start:batch_start + self.batch_size]
            
            for memory_id in batch_ids:
                try:
                    memory = await self.retrieve(memory_id)
                    if not memory:
                        continue
                    
                    # Calculate decay
                    time_since_access = current_time - (memory.last_accessed or memory.timestamp)
                    days_since_access = time_since_access / 86400
                    
                    if days_since_access > 1:
                        decay_factor = 0.99 ** days_since_access
                        
                        # Adjust importance
                        new_importance = memory.importance * decay_factor
                        
                        # Boost based on access count
                        access_boost = min(0.1, memory.access_count / 100.0)
                        new_importance = min(1.0, new_importance + access_boost)
                        
                        # Update if changed significantly
                        if abs(memory.importance - new_importance) > 0.01:
                            await self.update(
                                memory_id,
                                importance=new_importance
                            )
                            
                except Exception as e:
                    logger.debug(f"Decay error for {memory_id}: {e}")
        
        logger.debug("Memory decay applied")
    
    async def _cleanup_ghost_memory(self):
        """Remove expired ghost memories"""
        current_time = time.time()
        expired = []
        
        async with self.ghost_lock:
            for memory_id, memory in self.ghost_memory.items():
                age = current_time - memory.timestamp
                if age > self.ghost_memory_ttl:
                    expired.append(memory_id)
            
            for memory_id in expired:
                del self.ghost_memory[memory_id]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired ghost memories")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory vault statistics"""
        # Calculate storage sizes
        total_size = 0
        for directory in [self.memories_dir, self.embeddings_dir, self.packfiles_dir]:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        stats = {
            'total_memories': (
                len(self.working_memory) + 
                len(self.ghost_memory) + 
                len(self.main_index)
            ),
            'working_memory_count': len(self.working_memory),
            'ghost_memory_count': len(self.ghost_memory),
            'file_storage_count': len(self.main_index),
            'storage_path': str(self.storage_path),
            'total_size_mb': total_size / (1024 * 1024),
            'storage_type': 'FILE-BASED V2 (ASYNC)',
            'type_distribution': {k: len(v) for k, v in self.type_index.items()},
            'tag_count': len(self.tag_index),
            'session_id': self.session_id,
            'uptime_seconds': time.time() - self._start_time,
            'operation_count': self._operation_count
        }
        
        return stats
    
    async def get_status(self) -> Dict[str, Any]:
        """Get memory vault status"""
        stats = await self.get_statistics()
        
        # Find last modified time
        last_modified = 0
        for p in self.storage_path.rglob('*'):
            if p.is_file():
                last_modified = max(last_modified, p.stat().st_mtime)
        
        return {
            "entries": stats['total_memories'],
            "working_memory": stats['working_memory_count'],
            "ghost_memory": stats['ghost_memory_count'],
            "file_storage": stats['file_storage_count'],
            "path": str(self.storage_path),
            "last_modified": datetime.fromtimestamp(last_modified).isoformat() if last_modified else "Never",
            "session_id": self.session_id,
            "uptime": stats['uptime_seconds']
        }
    
    async def save_all(self):
        """Save all memories to disk"""
        try:
            # Flush log buffer
            await self._flush_log_buffer(force=True)
            
            # Save indices
            await self._save_indices()
            
            # Save seen hashes
            await self._save_seen_hashes()
            
            # Write final snapshot
            await self._write_snapshot()
            
            # Write session summary
            stats = await self.get_statistics()
            session_summary = {
                'session_id': self.session_id,
                'start_time': self._start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self._start_time,
                'operation_count': self._operation_count,
                'final_stats': stats
            }
            
            summary_path = self.logs_dir / f'session_{self.session_id}_summary.json'
            data = json.dumps(session_summary, indent=2)
            await self._atomic_write(summary_path, data)
            
            logger.info("✅ UnifiedMemoryVaultV2 saved all data")
            logger.info(f"  - Total memories: {stats['total_memories']}")
            logger.info(f"  - Operations: {self._operation_count}")
            logger.info(f"  - Session summary: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save UnifiedMemoryVaultV2: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating UnifiedMemoryVaultV2 shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save everything
        await self.save_all()
        
        # Final consolidation
        await self.consolidate()
        
        logger.info("UnifiedMemoryVaultV2 shutdown complete")

# Example usage
if __name__ == "__main__":
    async def test_memory_vault_v2():
        """Test the improved memory vault"""
        config = {
            'storage_path': 'data/memory_test_v2',
            'max_working_memory': 50,
            'ghost_memory_ttl': 60,
            'batch_size': 10
        }
        
        vault = UnifiedMemoryVaultV2(config)
        await vault.initialize()
        
        print("Testing UnifiedMemoryVaultV2 (Fully Async)...")
        
        # Test storage
        memory_ids = []
        for i in range(5):
            memory_id = await vault.store(
                f"Test memory {i}",
                MemoryType.SEMANTIC,
                metadata={'index': i, 'tags': ['test']},
                importance=0.5 + i * 0.1
            )
            memory_ids.append(memory_id)
            print(f"Stored: {memory_id}")
        
        # Test retrieval
        print("\nTesting retrieval...")
        memory = await vault.retrieve(memory_ids[0])
        if memory:
            print(f"Retrieved: {memory.content}")
        
        # Test search
        print("\nTesting search...")
        results = await vault.search(tags=['test'], max_results=3)
        print(f"Found {len(results)} memories with 'test' tag")
        
        # Test streaming
        print("\nTesting streaming...")
        count = 0
        async for memory in vault.stream_all():
            count += 1
        print(f"Streamed {count} total memories")
        
        # Get statistics
        stats = await vault.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Storage type: {stats['storage_type']}")
        print(f"  Operations: {stats['operation_count']}")
        
        # Graceful shutdown
        await vault.shutdown()
    
    # Run test
    asyncio.run(test_memory_vault_v2())
