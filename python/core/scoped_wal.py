"""
Scoped Write-Ahead Log (WAL) System
Per-tenant WAL isolation for safe multi-user operations
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import gzip

logger = logging.getLogger(__name__)

@dataclass
class WALEntry:
    """Single WAL entry"""
    sequence: int
    timestamp: float
    operation: str  # "add", "remove", "update", "relate"
    data: Dict[str, Any]
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WALEntry':
        return cls(**data)

class ScopedWAL:
    """
    Write-Ahead Log scoped to a specific tenant (user or group)
    Ensures durability and recovery for concept mesh operations
    """
    
    def __init__(self, scope: str, scope_id: str, data_dir: Optional[Path] = None):
        self.scope = scope  # "user" or "group"
        self.scope_id = scope_id
        self.scope_key = f"{scope}_{scope_id}"
        
        # Storage configuration
        self.data_dir = data_dir or Path(f"data/wal/{scope}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # WAL files
        self.wal_file = self.data_dir / f"{scope_id}.wal"
        self.checkpoint_file = self.data_dir / f"{scope_id}.checkpoint"
        
        # State
        self.sequence = 0
        self.entries: List[WALEntry] = []
        self.file_handle = None
        self.lock = threading.Lock()
        
        # Configuration
        self.max_entries_before_checkpoint = 1000
        self.compress_on_checkpoint = True
        
        # Initialize
        self._load_checkpoint()
        self._open_wal()
        
        logger.info(f"📝 Initialized {scope}-scoped WAL for {scope_id}")
    
    def _open_wal(self):
        """Open WAL file for appending"""
        try:
            self.file_handle = open(self.wal_file, 'a', encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to open WAL file: {e}")
            raise
    
    def write(self, operation: str, data: Dict[str, Any]) -> int:
        """
        Write an entry to the WAL
        
        Args:
            operation: Type of operation
            data: Operation data
            
        Returns:
            Sequence number of the entry
        """
        with self.lock:
            # Create entry
            self.sequence += 1
            entry = WALEntry(
                sequence=self.sequence,
                timestamp=time.time(),
                operation=operation,
                data=data
            )
            
            # Write to file
            if self.file_handle:
                json_line = json.dumps(entry.to_dict()) + '\n'
                self.file_handle.write(json_line)
                self.file_handle.flush()
                os.fsync(self.file_handle.fileno())  # Force write to disk
            
            # Keep in memory
            self.entries.append(entry)
            
            # Check if checkpoint needed
            if len(self.entries) >= self.max_entries_before_checkpoint:
                self._create_checkpoint()
            
            logger.debug(f"WAL[{self.scope_key}] wrote seq={self.sequence} op={operation}")
            return self.sequence
    
    def replay(self, target_func: callable) -> int:
        """
        Replay all WAL entries through a target function
        
        Args:
            target_func: Function to call for each entry (operation, data) -> None
            
        Returns:
            Number of entries replayed
        """
        replayed = 0
        
        # First replay from checkpoint if exists
        if self.checkpoint_file.exists():
            checkpoint_entries = self._load_checkpoint_entries()
            for entry in checkpoint_entries:
                target_func(entry.operation, entry.data)
                replayed += 1
        
        # Then replay from current WAL
        if self.wal_file.exists():
            with open(self.wal_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry_dict = json.loads(line)
                        entry = WALEntry.from_dict(entry_dict)
                        target_func(entry.operation, entry.data)
                        replayed += 1
        
        logger.info(f"WAL[{self.scope_key}] replayed {replayed} entries")
        return replayed
    
    def _create_checkpoint(self):
        """Create a checkpoint and rotate WAL"""
        try:
            # Close current WAL
            if self.file_handle:
                self.file_handle.close()
            
            # Write checkpoint
            checkpoint_data = {
                "scope": self.scope,
                "scope_id": self.scope_id,
                "sequence": self.sequence,
                "timestamp": time.time(),
                "entries": [e.to_dict() for e in self.entries]
            }
            
            if self.compress_on_checkpoint:
                with gzip.open(self.checkpoint_file.with_suffix('.checkpoint.gz'), 'wt', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f)
            else:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
            
            # Clear in-memory entries
            self.entries.clear()
            
            # Rotate WAL file
            if self.wal_file.exists():
                backup_file = self.wal_file.with_suffix(f'.wal.{int(time.time())}')
                self.wal_file.rename(backup_file)
            
            # Reopen new WAL
            self._open_wal()
            
            logger.info(f"WAL[{self.scope_key}] created checkpoint at seq={self.sequence}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        checkpoint_path = self.checkpoint_file
        
        # Check for compressed version first
        if self.checkpoint_file.with_suffix('.checkpoint.gz').exists():
            checkpoint_path = self.checkpoint_file.with_suffix('.checkpoint.gz')
            
        if checkpoint_path.exists():
            try:
                if checkpoint_path.suffix == '.gz':
                    with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                self.sequence = data.get('sequence', 0)
                logger.info(f"WAL[{self.scope_key}] loaded checkpoint at seq={self.sequence}")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
    
    def _load_checkpoint_entries(self) -> List[WALEntry]:
        """Load entries from checkpoint file"""
        entries = []
        checkpoint_path = self.checkpoint_file
        
        if self.checkpoint_file.with_suffix('.checkpoint.gz').exists():
            checkpoint_path = self.checkpoint_file.with_suffix('.checkpoint.gz')
            
        if checkpoint_path.exists():
            try:
                if checkpoint_path.suffix == '.gz':
                    with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                for entry_dict in data.get('entries', []):
                    entries.append(WALEntry.from_dict(entry_dict))
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint entries: {e}")
        
        return entries
    
    def close(self):
        """Close the WAL file"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WAL statistics"""
        wal_size = self.wal_file.stat().st_size if self.wal_file.exists() else 0
        checkpoint_size = 0
        
        if self.checkpoint_file.exists():
            checkpoint_size = self.checkpoint_file.stat().st_size
        elif self.checkpoint_file.with_suffix('.checkpoint.gz').exists():
            checkpoint_size = self.checkpoint_file.with_suffix('.checkpoint.gz').stat().st_size
        
        return {
            "scope": self.scope,
            "scope_id": self.scope_id,
            "sequence": self.sequence,
            "entries_in_memory": len(self.entries),
            "wal_size_bytes": wal_size,
            "checkpoint_size_bytes": checkpoint_size,
            "total_size_bytes": wal_size + checkpoint_size
        }


class WALManager:
    """
    Manager for all scoped WAL instances
    """
    
    _instances: Dict[str, ScopedWAL] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_wal(cls, scope: str, scope_id: str) -> ScopedWAL:
        """Get or create a scoped WAL instance"""
        scope_key = f"{scope}_{scope_id}"
        
        if scope_key in cls._instances:
            return cls._instances[scope_key]
        
        with cls._lock:
            # Double-check pattern
            if scope_key in cls._instances:
                return cls._instances[scope_key]
            
            # Create new instance
            wal = ScopedWAL(scope, scope_id)
            cls._instances[scope_key] = wal
            return wal
    
    @classmethod
    def close_all(cls):
        """Close all WAL instances"""
        with cls._lock:
            for wal in cls._instances.values():
                wal.close()
            cls._instances.clear()
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get statistics for all WALs"""
        stats = {
            "total_wals": len(cls._instances),
            "wals": {}
        }
        
        for scope_key, wal in cls._instances.items():
            stats["wals"][scope_key] = wal.get_stats()
        
        return stats
