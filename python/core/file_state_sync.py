#!/usr/bin/env python3
"""
File-Based State Synchronization for TORI/KHA
Distributed state management without Redis or databases
Uses file system and MCP servers for coordination
"""

import json
import time
import pickle
import hashlib
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, List, Set, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import uuid
import struct
import platform

logger = logging.getLogger(__name__)

# ========== State Types ==========

class StateType(Enum):
    """Types of distributed state"""
    COGNITIVE = "cognitive"
    EIGENVALUE = "eigenvalue"
    MEMORY = "memory"
    CHAOS = "chaos"
    SAFETY = "safety"
    METRICS = "metrics"

@dataclass
class StateEntry:
    """Single state entry"""
    key: str
    value: Any
    version: int
    timestamp: float
    node_id: str
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes"""
        return pickle.dumps(asdict(self))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'StateEntry':
        """Deserialize from bytes"""
        return cls(**pickle.loads(data))

@dataclass
class StateDelta:
    """Change in state"""
    key: str
    old_value: Any
    new_value: Any
    version: int
    timestamp: float
    operation: str  # "set", "delete", "expire"

# ========== File Locking ==========

class FileLock:
    """Cross-platform file locking"""
    
    def __init__(self, filepath: Path, timeout: float = 5.0):
        self.filepath = filepath
        self.timeout = timeout
        self.lock_file = None
        self.is_windows = platform.system() == "Windows"
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def acquire(self):
        """Acquire file lock"""
        start_time = time.time()
        
        while True:
            try:
                # Try to create lock file exclusively
                self.lock_file = open(self.filepath, 'x')
                break  # Lock acquired
                
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.filepath}")
                time.sleep(0.01)
    
    def release(self):
        """Release file lock"""
        if self.lock_file:
            try:
                self.lock_file.close()
                self.filepath.unlink(missing_ok=True)
            except:
                pass
            finally:
                self.lock_file = None

# ========== State Store ==========

class FileStateStore:
    """
    File-based distributed state store
    Alternative to Redis with similar interface
    """
    
    def __init__(self, storage_path: Path = Path("data/state_store")):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage directories
        self.state_dir = self.storage_path / "state"
        self.index_dir = self.storage_path / "index"
        self.wal_dir = self.storage_path / "wal"  # Write-ahead log
        self.snapshot_dir = self.storage_path / "snapshots"
        
        for dir_path in [self.state_dir, self.index_dir, self.wal_dir, self.snapshot_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Node identification
        self.node_id = f"{platform.node()}_{uuid.uuid4().hex[:8]}"
        
        # In-memory cache
        self.cache: Dict[str, StateEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Indexes
        self.key_index: Dict[str, Path] = {}  # key -> file path
        self.ttl_index: Dict[float, Set[str]] = defaultdict(set)  # expiry time -> keys
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)  # pattern -> keys
        
        # Write-ahead log
        self.wal_file = self.wal_dir / f"wal_{self.node_id}.log"
        self.wal_lock = threading.Lock()
        
        # Background tasks
        self.running = True
        self.background_threads = []
        self._start_background_tasks()
        
        # Load existing state
        self._load_state()
        
        logger.info(f"FileStateStore initialized at {self.storage_path}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Expiry cleaner
        expiry_thread = threading.Thread(target=self._expiry_loop, daemon=True)
        expiry_thread.start()
        self.background_threads.append(expiry_thread)
        
        # WAL compactor
        wal_thread = threading.Thread(target=self._wal_compaction_loop, daemon=True)
        wal_thread.start()
        self.background_threads.append(wal_thread)
        
        # Snapshot creator
        snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        snapshot_thread.start()
        self.background_threads.append(snapshot_thread)
    
    def _load_state(self):
        """Load existing state from disk"""
        # Load key index
        index_file = self.index_dir / "key_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
                self.key_index = {k: Path(v) for k, v in index_data.items()}
        
        # Load state files into cache
        loaded = 0
        for key, filepath in self.key_index.items():
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        entry = StateEntry.from_bytes(f.read())
                    
                    if not entry.is_expired():
                        self.cache[key] = entry
                        
                        # Update TTL index
                        if entry.ttl:
                            expiry_time = entry.timestamp + entry.ttl
                            self.ttl_index[expiry_time].add(key)
                        
                        loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load state for key {key}: {e}")
        
        logger.info(f"Loaded {loaded} state entries from disk")
    
    def _get_state_file(self, key: str) -> Path:
        """Get file path for a state key"""
        # Use hash to distribute files across subdirectories
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        subdir = self.state_dir / key_hash[:2] / key_hash[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key_hash}.state"
    
    def _write_wal(self, operation: str, key: str, value: Any = None, version: int = 0):
        """Write to write-ahead log"""
        with self.wal_lock:
            entry = {
                'timestamp': time.time(),
                'operation': operation,
                'key': key,
                'value': value,
                'version': version,
                'node_id': self.node_id
            }
            
            with open(self.wal_file, 'ab') as f:
                data = pickle.dumps(entry)
                f.write(struct.pack('I', len(data)))  # Write length prefix
                f.write(data)
    
    # ========== Core Operations ==========
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key"""
        with self.cache_lock:
            # Check cache first
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            # Load from disk if not in cache
            filepath = self.key_index.get(key)
            if filepath and filepath.exists():
                try:
                    lock_path = filepath.with_suffix('.lock')
                    with FileLock(lock_path):
                        with open(filepath, 'rb') as f:
                            entry = StateEntry.from_bytes(f.read())
                    
                    if not entry.is_expired():
                        self.cache[key] = entry
                        return entry.value
                except Exception as e:
                    logger.error(f"Failed to load key {key}: {e}")
            
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set key-value pair"""
        with self.cache_lock:
            # Get current version
            current_entry = self.cache.get(key)
            version = (current_entry.version + 1) if current_entry else 1
            
            # Create new entry
            entry = StateEntry(
                key=key,
                value=value,
                version=version,
                timestamp=time.time(),
                node_id=self.node_id,
                ttl=ttl
            )
            
            # Get file path
            filepath = self._get_state_file(key)
            
            try:
                # Write to disk
                lock_path = filepath.with_suffix('.lock')
                with FileLock(lock_path):
                    with open(filepath, 'wb') as f:
                        f.write(entry.to_bytes())
                
                # Update cache
                self.cache[key] = entry
                
                # Update indexes
                self.key_index[key] = filepath
                
                if ttl:
                    expiry_time = entry.timestamp + ttl
                    self.ttl_index[expiry_time].add(key)
                
                # Write to WAL
                self._write_wal('set', key, value, version)
                
                # Save index
                self._save_index()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key"""
        with self.cache_lock:
            # Remove from cache
            if key in self.cache:
                del self.cache[key]
            
            # Remove file
            filepath = self.key_index.get(key)
            if filepath and filepath.exists():
                try:
                    filepath.unlink()
                except:
                    pass
            
            # Update indexes
            if key in self.key_index:
                del self.key_index[key]
            
            # Remove from TTL index
            for expiry_time, keys in list(self.ttl_index.items()):
                if key in keys:
                    keys.remove(key)
                    if not keys:
                        del self.ttl_index[expiry_time]
            
            # Write to WAL
            self._write_wal('delete', key)
            
            # Save index
            self._save_index()
            
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        with self.cache_lock:
            # Check cache
            if key in self.cache and not self.cache[key].is_expired():
                return True
            
            # Check disk
            filepath = self.key_index.get(key)
            return filepath is not None and filepath.exists()
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        import fnmatch
        
        with self.cache_lock:
            all_keys = set(self.key_index.keys())
            
            if pattern == "*":
                return list(all_keys)
            
            # Filter by pattern
            matching_keys = []
            for key in all_keys:
                if fnmatch.fnmatch(key, pattern):
                    # Check if not expired
                    if self.exists(key):
                        matching_keys.append(key)
            
            return matching_keys
    
    def mget(self, keys: List[str]) -> List[Any]:
        """Get multiple keys"""
        return [self.get(key) for key in keys]
    
    def mset(self, mapping: Dict[str, Any]) -> bool:
        """Set multiple keys"""
        success = True
        for key, value in mapping.items():
            if not self.set(key, value):
                success = False
        return success
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment numeric value"""
        with self.cache_lock:
            current = self.get(key, 0)
            new_value = int(current) + amount
            self.set(key, new_value)
            return new_value
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement numeric value"""
        return self.incr(key, -amount)
    
    def expire(self, key: str, ttl: float) -> bool:
        """Set expiry time for key"""
        with self.cache_lock:
            entry = self.cache.get(key)
            if entry:
                entry.ttl = ttl
                entry.timestamp = time.time()  # Reset timestamp
                
                # Update on disk
                filepath = self.key_index.get(key)
                if filepath:
                    lock_path = filepath.with_suffix('.lock')
                    with FileLock(lock_path):
                        with open(filepath, 'wb') as f:
                            f.write(entry.to_bytes())
                
                # Update TTL index
                expiry_time = entry.timestamp + ttl
                self.ttl_index[expiry_time].add(key)
                
                return True
        
        return False
    
    def ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for key"""
        with self.cache_lock:
            entry = self.cache.get(key)
            if entry and entry.ttl:
                remaining = entry.ttl - (time.time() - entry.timestamp)
                return max(0, remaining)
        return None
    
    # ========== Hash Operations ==========
    
    def hget(self, name: str, key: str, default: Any = None) -> Any:
        """Get hash field value"""
        hash_data = self.get(name, {})
        return hash_data.get(key, default)
    
    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field"""
        with self.cache_lock:
            hash_data = self.get(name, {})
            hash_data[key] = value
            return self.set(name, hash_data)
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        return self.get(name, {})
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        with self.cache_lock:
            hash_data = self.get(name, {})
            deleted = 0
            for key in keys:
                if key in hash_data:
                    del hash_data[key]
                    deleted += 1
            
            if deleted > 0:
                self.set(name, hash_data)
            
            return deleted
    
    # ========== List Operations ==========
    
    def lpush(self, name: str, *values: Any) -> int:
        """Push values to list head"""
        with self.cache_lock:
            lst = self.get(name, [])
            lst = list(values) + lst
            self.set(name, lst)
            return len(lst)
    
    def rpush(self, name: str, *values: Any) -> int:
        """Push values to list tail"""
        with self.cache_lock:
            lst = self.get(name, [])
            lst.extend(values)
            self.set(name, lst)
            return len(lst)
    
    def lpop(self, name: str) -> Any:
        """Pop from list head"""
        with self.cache_lock:
            lst = self.get(name, [])
            if lst:
                value = lst.pop(0)
                self.set(name, lst)
                return value
            return None
    
    def rpop(self, name: str) -> Any:
        """Pop from list tail"""
        with self.cache_lock:
            lst = self.get(name, [])
            if lst:
                value = lst.pop()
                self.set(name, lst)
                return value
            return None
    
    def lrange(self, name: str, start: int, stop: int) -> List[Any]:
        """Get list range"""
        lst = self.get(name, [])
        if stop == -1:
            return lst[start:]
        return lst[start:stop+1]
    
    def llen(self, name: str) -> int:
        """Get list length"""
        lst = self.get(name, [])
        return len(lst)
    
    # ========== Set Operations ==========
    
    def sadd(self, name: str, *values: Any) -> int:
        """Add to set"""
        with self.cache_lock:
            s = set(self.get(name, []))
            added = 0
            for value in values:
                if value not in s:
                    s.add(value)
                    added += 1
            
            if added > 0:
                self.set(name, list(s))
            
            return added
    
    def srem(self, name: str, *values: Any) -> int:
        """Remove from set"""
        with self.cache_lock:
            s = set(self.get(name, []))
            removed = 0
            for value in values:
                if value in s:
                    s.remove(value)
                    removed += 1
            
            if removed > 0:
                self.set(name, list(s))
            
            return removed
    
    def smembers(self, name: str) -> Set[Any]:
        """Get set members"""
        return set(self.get(name, []))
    
    def sismember(self, name: str, value: Any) -> bool:
        """Check set membership"""
        s = set(self.get(name, []))
        return value in s
    
    # ========== Pub/Sub Operations ==========
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        # Simple file-based pub/sub using a ring buffer
        pubsub_dir = self.storage_path / "pubsub" / channel
        pubsub_dir.mkdir(parents=True, exist_ok=True)
        
        # Write message with timestamp
        msg_file = pubsub_dir / f"{time.time():.6f}_{uuid.uuid4().hex[:8]}.msg"
        with open(msg_file, 'wb') as f:
            pickle.dump(message, f)
        
        # Clean old messages (keep last 1000)
        messages = sorted(pubsub_dir.glob("*.msg"))
        if len(messages) > 1000:
            for old_msg in messages[:-1000]:
                old_msg.unlink()
        
        return 1  # Number of subscribers (simplified)
    
    def subscribe(self, *channels: str) -> 'PubSubHandler':
        """Subscribe to channels"""
        return PubSubHandler(self, channels)
    
    # ========== Background Tasks ==========
    
    def _expiry_loop(self):
        """Background task to clean expired keys"""
        while self.running:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self.cache_lock:
                    # Check TTL index
                    for expiry_time, keys in list(self.ttl_index.items()):
                        if expiry_time <= current_time:
                            expired_keys.extend(keys)
                            del self.ttl_index[expiry_time]
                
                # Delete expired keys
                for key in expired_keys:
                    self.delete(key)
                    logger.debug(f"Expired key: {key}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Expiry loop error: {e}")
                time.sleep(5)
    
    def _wal_compaction_loop(self):
        """Background task to compact WAL"""
        while self.running:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Check WAL size
                if self.wal_file.exists() and self.wal_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                    logger.info("Compacting WAL...")
                    
                    # Create new WAL file
                    new_wal = self.wal_file.with_suffix('.new')
                    
                    # Copy recent entries only
                    # (In production, would implement proper compaction)
                    
                    # Rotate files
                    old_wal = self.wal_file.with_suffix('.old')
                    self.wal_file.rename(old_wal)
                    if new_wal.exists():
                        new_wal.rename(self.wal_file)
                    old_wal.unlink()
                    
            except Exception as e:
                logger.error(f"WAL compaction error: {e}")
    
    def _snapshot_loop(self):
        """Background task to create snapshots"""
        while self.running:
            try:
                time.sleep(3600)  # Every hour
                self.create_snapshot()
            except Exception as e:
                logger.error(f"Snapshot error: {e}")
    
    def _save_index(self):
        """Save key index to disk"""
        index_data = {k: str(v) for k, v in self.key_index.items()}
        temp_file = self.index_dir / "key_index.tmp"
        
        with open(temp_file, 'w') as f:
            json.dump(index_data, f)
        
        # Atomic rename
        temp_file.replace(self.index_dir / "key_index.json")
    
    def create_snapshot(self) -> str:
        """Create state snapshot"""
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id}"
        snapshot_path.mkdir()
        
        with self.cache_lock:
            # Save all current state
            state_data = {}
            for key, entry in self.cache.items():
                if not entry.is_expired():
                    state_data[key] = asdict(entry)
            
            # Write snapshot
            with open(snapshot_path / "state.json", 'w') as f:
                json.dump(state_data, f)
            
            # Copy indexes
            import shutil
            shutil.copy2(self.index_dir / "key_index.json", snapshot_path / "index.json")
        
        logger.info(f"Created snapshot: {snapshot_id}")
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str):
        """Restore from snapshot"""
        snapshot_path = self.snapshot_dir / f"snapshot_{snapshot_id}"
        
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        # Clear current state
        with self.cache_lock:
            self.cache.clear()
            self.key_index.clear()
            self.ttl_index.clear()
        
        # Load snapshot
        with open(snapshot_path / "state.json", 'r') as f:
            state_data = json.load(f)
        
        # Restore state
        for key, entry_data in state_data.items():
            entry = StateEntry(**entry_data)
            self.cache[key] = entry
            
            # Restore file
            filepath = self._get_state_file(key)
            with open(filepath, 'wb') as f:
                f.write(entry.to_bytes())
            
            self.key_index[key] = filepath
        
        logger.info(f"Restored from snapshot: {snapshot_id}")
    
    def close(self):
        """Close state store"""
        self.running = False
        
        # Wait for background threads
        for thread in self.background_threads:
            thread.join(timeout=5)
        
        # Final save
        self._save_index()
        
        logger.info("FileStateStore closed")

# ========== Pub/Sub Handler ==========

class PubSubHandler:
    """Handle pub/sub subscriptions"""
    
    def __init__(self, store: FileStateStore, channels: Tuple[str, ...]):
        self.store = store
        self.channels = channels
        self.message_queue = asyncio.Queue()
        self.running = True
        
        # Start monitoring threads for each channel
        for channel in channels:
            thread = threading.Thread(
                target=self._monitor_channel,
                args=(channel,),
                daemon=True
            )
            thread.start()
    
    def _monitor_channel(self, channel: str):
        """Monitor channel for new messages"""
        pubsub_dir = self.store.storage_path / "pubsub" / channel
        pubsub_dir.mkdir(parents=True, exist_ok=True)
        
        seen_messages = set()
        
        while self.running:
            try:
                # Check for new messages
                for msg_file in sorted(pubsub_dir.glob("*.msg")):
                    if msg_file.name not in seen_messages:
                        seen_messages.add(msg_file.name)
                        
                        # Read message
                        with open(msg_file, 'rb') as f:
                            message = pickle.load(f)
                        
                        # Add to queue
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            # Create new event loop if needed
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        asyncio.run_coroutine_threadsafe(
                            self.message_queue.put({
                                'channel': channel,
                                'data': message
                            }),
                            loop
                        )
                
                time.sleep(0.1)  # Poll every 100ms
                
            except Exception as e:
                logger.error(f"Channel monitor error: {e}")
    
    async def get_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get next message"""
        try:
            if timeout:
                return await asyncio.wait_for(self.message_queue.get(), timeout)
            else:
                return await self.message_queue.get()
        except asyncio.TimeoutError:
            return None
    
    def unsubscribe(self):
        """Unsubscribe from channels"""
        self.running = False

# ========== Distributed Lock ==========

class DistributedLock:
    """Distributed lock using file-based coordination"""
    
    def __init__(self, store: FileStateStore, name: str, timeout: float = 30.0):
        self.store = store
        self.name = f"__lock__{name}"
        self.timeout = timeout
        self.lock_id = str(uuid.uuid4())
        self.acquired = False
    
    async def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire lock"""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while True:
            # Try to acquire
            current = self.store.get(self.name)
            
            if current is None:
                # Lock is free
                lock_data = {
                    'lock_id': self.lock_id,
                    'node_id': self.store.node_id,
                    'acquired_at': time.time()
                }
                
                # Set with TTL to prevent deadlocks
                if self.store.set(self.name, lock_data, ttl=self.timeout):
                    self.acquired = True
                    return True
            
            elif current.get('lock_id') == self.lock_id:
                # We already have the lock
                return True
            
            if not blocking:
                return False
            
            if time.time() - start_time > timeout:
                return False
            
            await asyncio.sleep(0.1)
    
    async def release(self):
        """Release lock"""
        if self.acquired:
            current = self.store.get(self.name)
            if current and current.get('lock_id') == self.lock_id:
                self.store.delete(self.name)
                self.acquired = False
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

# ========== Integration with TORI ==========

class TORIStateSync:
    """Integrate state synchronization with TORI system"""
    
    def __init__(self, tori_system, storage_path: Path = Path("data/tori_state")):
        self.tori = tori_system
        self.store = FileStateStore(storage_path)
        
        # State prefixes
        self.prefixes = {
            StateType.COGNITIVE: "cognitive:",
            StateType.EIGENVALUE: "eigen:",
            StateType.MEMORY: "memory:",
            StateType.CHAOS: "chaos:",
            StateType.SAFETY: "safety:",
            StateType.METRICS: "metrics:"
        }
        
        # Start sync tasks
        self.sync_task = asyncio.create_task(self._sync_loop())
    
    async def _sync_loop(self):
        """Periodic state synchronization"""
        while True:
            try:
                await asyncio.sleep(5)  # Sync every 5 seconds
                
                # Sync cognitive state
                cognitive_state = self.tori.state_manager.get_state_dict()
                self.store.set(
                    f"{self.prefixes[StateType.COGNITIVE]}current",
                    cognitive_state,
                    ttl=60  # 1 minute TTL
                )
                
                # Sync eigenvalue metrics
                eigen_status = self.tori.eigen_sentry.get_status()
                self.store.set(
                    f"{self.prefixes[StateType.EIGENVALUE]}status",
                    eigen_status,
                    ttl=30
                )
                
                # Sync chaos metrics
                ccl_status = self.tori.ccl.get_status()
                self.store.set(
                    f"{self.prefixes[StateType.CHAOS]}status",
                    ccl_status,
                    ttl=30
                )
                
                # Sync safety state
                safety_report = self.tori.safety_system.get_safety_report()
                self.store.set(
                    f"{self.prefixes[StateType.SAFETY]}report",
                    safety_report,
                    ttl=60
                )
                
                # Increment metrics
                self.store.incr(f"{self.prefixes[StateType.METRICS]}sync_count")
                
            except Exception as e:
                logger.error(f"State sync error: {e}")
    
    def get_distributed_state(self) -> Dict[str, Any]:
        """Get all distributed state"""
        state = {}
        
        for state_type, prefix in self.prefixes.items():
            keys = self.store.keys(f"{prefix}*")
            for key in keys:
                value = self.store.get(key)
                if value is not None:
                    state[key] = value
        
        return state
    
    async def coordinate_chaos_task(self, task_id: str, task_data: Dict[str, Any]):
        """Coordinate distributed chaos task"""
        task_key = f"{self.prefixes[StateType.CHAOS]}task:{task_id}"
        
        # Acquire distributed lock
        async with DistributedLock(self.store, f"chaos_task_{task_id}"):
            # Set task state
            self.store.set(task_key, {
                'status': 'processing',
                'node_id': self.store.node_id,
                'data': task_data,
                'started_at': time.time()
            }, ttl=300)  # 5 minute TTL
            
            # Process task
            # ... task processing ...
            
            # Update completion
            self.store.set(task_key, {
                'status': 'completed',
                'node_id': self.store.node_id,
                'completed_at': time.time()
            }, ttl=3600)  # Keep for 1 hour
    
    def close(self):
        """Close state sync"""
        self.sync_task.cancel()
        self.store.close()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_state_store():
        """Test file-based state store"""
        store = FileStateStore(Path("data/test_state"))
        
        # Test basic operations
        print("Testing basic operations...")
        store.set("test:key1", "value1")
        print(f"get('test:key1') = {store.get('test:key1')}")
        
        # Test TTL
        store.set("test:expiring", "expires soon", ttl=2)
        print(f"TTL of 'test:expiring' = {store.ttl('test:expiring')}")
        
        # Test hash operations
        store.hset("test:hash", "field1", "value1")
        store.hset("test:hash", "field2", "value2")
        print(f"Hash contents: {store.hgetall('test:hash')}")
        
        # Test list operations
        store.rpush("test:list", "item1", "item2", "item3")
        print(f"List contents: {store.lrange('test:list', 0, -1)}")
        
        # Test set operations
        store.sadd("test:set", "member1", "member2", "member1")
        print(f"Set members: {store.smembers('test:set')}")
        
        # Test pub/sub
        print("\nTesting pub/sub...")
        subscriber = store.subscribe("test:channel")
        
        # Publish message
        store.publish("test:channel", {"type": "test", "data": "Hello!"})
        
        # Get message
        message = await subscriber.get_message(timeout=2)
        print(f"Received message: {message}")
        
        # Test distributed lock
        print("\nTesting distributed lock...")
        async with DistributedLock(store, "test_resource") as lock:
            print("Lock acquired")
            # Simulate work
            await asyncio.sleep(1)
        print("Lock released")
        
        # Test snapshot
        print("\nCreating snapshot...")
        snapshot_id = store.create_snapshot()
        print(f"Created snapshot: {snapshot_id}")
        
        # Clean up
        store.close()
        print("\nTest complete!")
    
    asyncio.run(test_state_store())
