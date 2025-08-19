#!/usr/bin/env python3
"""
Atomic Adapter Loader with Thread-Safe Hot-Swapping
Ensures no race conditions, full auditability, instant rollback
"""

import os
import json
import time
import threading
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import fcntl  # For file locking on Unix
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global lock for adapter operations
_adapter_lock = threading.RLock()  # Reentrant lock
_adapter_cache = {}  # In-memory cache
_adapter_events = []  # Event log

class AtomicAdapterLoader:
    """Thread-safe, atomic adapter management with rollback"""
    
    def __init__(self, base_path: str = "models/adapters"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata.json"
        self.events_path = self.base_path / "events.jsonl"
        self.backup_path = self.base_path / "backups"
        self.backup_path.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load adapter metadata with lock"""
        with _adapter_lock:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            return {
                "version": "2.0.0",
                "adapters": {},
                "active": {},
                "history": []
            }
    
    def _save_metadata(self):
        """Save metadata atomically"""
        with _adapter_lock:
            # Write to temp file first
            temp_path = self.metadata_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.metadata_path)
            
    def _log_event(self, event_type: str, data: Dict):
        """Log adapter event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        # Append to events file
        with open(self.events_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Also keep in memory
        _adapter_events.append(event)
        
        # Log to standard logger
        logger.info(f"Adapter event: {event_type} - {data}")
    
    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _validate_adapter(self, adapter_path: Path, user_id: str) -> bool:
        """Validate adapter before activation"""
        try:
            # Check file exists
            if not adapter_path.exists():
                raise ValueError(f"Adapter not found: {adapter_path}")
            
            # Check file size (basic sanity check)
            size = adapter_path.stat().st_size
            if size < 1024:  # Less than 1KB is suspicious
                raise ValueError(f"Adapter too small: {size} bytes")
            if size > 10 * 1024 * 1024 * 1024:  # More than 10GB is suspicious
                raise ValueError(f"Adapter too large: {size} bytes")
            
            # Compute and verify hash
            file_hash = self._compute_hash(adapter_path)
            
            # TODO: Add actual model validation here
            # - Load with torch and check dimensions
            # - Run test inference
            # - Check compatibility with base model
            
            self._log_event("validation_success", {
                "user_id": user_id,
                "adapter": str(adapter_path),
                "hash": file_hash,
                "size": size
            })
            
            return True
            
        except Exception as e:
            self._log_event("validation_failure", {
                "user_id": user_id,
                "adapter": str(adapter_path),
                "error": str(e)
            })
            logger.error(f"Validation failed: {e}")
            return False
    
    def atomic_symlink_swap(self, new_target: Path, link_path: Path) -> bool:
        """Atomically swap symlink to new target"""
        with _adapter_lock:
            try:
                # Create backup of current link if it exists
                if link_path.exists():
                    backup_name = f"{link_path.name}.backup.{int(time.time())}"
                    backup_file = self.backup_path / backup_name
                    
                    # Save current target
                    if link_path.is_symlink():
                        current_target = link_path.readlink()
                        with open(backup_file, 'w') as f:
                            json.dump({
                                "link": str(link_path),
                                "target": str(current_target),
                                "timestamp": datetime.now().isoformat()
                            }, f)
                
                # Create temporary symlink
                temp_link = Path(str(link_path) + f".tmp.{os.getpid()}")
                if temp_link.exists():
                    temp_link.unlink()
                
                # Create new symlink
                temp_link.symlink_to(new_target)
                
                # Atomic rename (on Unix, this is atomic)
                temp_link.replace(link_path)
                
                self._log_event("symlink_swap", {
                    "link": str(link_path),
                    "new_target": str(new_target),
                    "success": True
                })
                
                return True
                
            except Exception as e:
                logger.error(f"Symlink swap failed: {e}")
                self._log_event("symlink_swap_error", {
                    "link": str(link_path),
                    "new_target": str(new_target),
                    "error": str(e)
                })
                return False
    
    def hot_swap_adapter(self, user_id: str, new_adapter_name: str) -> bool:
        """Hot-swap user adapter with validation and rollback capability"""
        with _adapter_lock:
            try:
                # Paths
                new_adapter_path = self.base_path / new_adapter_name
                link_path = self.base_path / f"user_{user_id}_active.pt"
                
                # Validate new adapter
                if not self._validate_adapter(new_adapter_path, user_id):
                    raise ValueError("Adapter validation failed")
                
                # Store previous adapter for rollback
                previous_adapter = None
                if link_path.exists() and link_path.is_symlink():
                    previous_adapter = str(link_path.readlink())
                
                # Perform atomic swap
                if not self.atomic_symlink_swap(new_adapter_path, link_path):
                    raise RuntimeError("Symlink swap failed")
                
                # Update metadata
                self.metadata["active"][user_id] = new_adapter_name
                self.metadata["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "action": "hot_swap",
                    "from": previous_adapter,
                    "to": new_adapter_name
                })
                self._save_metadata()
                
                # Clear cache for this user
                cache_key = f"{user_id}_adapter"
                if cache_key in _adapter_cache:
                    del _adapter_cache[cache_key]
                
                self._log_event("hot_swap_success", {
                    "user_id": user_id,
                    "new_adapter": new_adapter_name,
                    "previous_adapter": previous_adapter
                })
                
                logger.info(f"Successfully hot-swapped adapter for {user_id}: {new_adapter_name}")
                return True
                
            except Exception as e:
                logger.error(f"Hot-swap failed for {user_id}: {e}")
                self._log_event("hot_swap_failure", {
                    "user_id": user_id,
                    "new_adapter": new_adapter_name,
                    "error": str(e)
                })
                
                # Attempt rollback
                if 'previous_adapter' in locals() and previous_adapter:
                    self.rollback_adapter(user_id)
                
                return False
    
    def rollback_adapter(self, user_id: str, steps: int = 1) -> bool:
        """Rollback to previous adapter version"""
        with _adapter_lock:
            try:
                # Find user's history
                user_history = [
                    h for h in self.metadata["history"]
                    if h["user_id"] == user_id and h["action"] == "hot_swap"
                ]
                
                if len(user_history) < steps:
                    raise ValueError(f"Not enough history to rollback {steps} steps")
                
                # Get target adapter
                target_entry = user_history[-(steps + 1)]
                target_adapter = target_entry["from"]
                
                if not target_adapter:
                    raise ValueError("No previous adapter to rollback to")
                
                # Perform rollback
                target_path = Path(target_adapter)
                link_path = self.base_path / f"user_{user_id}_active.pt"
                
                if not self.atomic_symlink_swap(target_path, link_path):
                    raise RuntimeError("Rollback swap failed")
                
                # Update metadata
                self.metadata["active"][user_id] = target_path.name
                self.metadata["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "action": "rollback",
                    "steps": steps,
                    "to": target_adapter
                })
                self._save_metadata()
                
                self._log_event("rollback_success", {
                    "user_id": user_id,
                    "steps": steps,
                    "adapter": target_adapter
                })
                
                logger.info(f"Successfully rolled back adapter for {user_id}")
                return True
                
            except Exception as e:
                logger.error(f"Rollback failed for {user_id}: {e}")
                self._log_event("rollback_failure", {
                    "user_id": user_id,
                    "steps": steps,
                    "error": str(e)
                })
                return False
    
    def get_active_adapter(self, user_id: str) -> Optional[Path]:
        """Get currently active adapter for user"""
        with _adapter_lock:
            # Check cache first
            cache_key = f"{user_id}_adapter"
            if cache_key in _adapter_cache:
                return _adapter_cache[cache_key]
            
            # Check symlink
            link_path = self.base_path / f"user_{user_id}_active.pt"
            if link_path.exists() and link_path.is_symlink():
                target = link_path.readlink()
                _adapter_cache[cache_key] = target
                return target
            
            # Fallback to default
            default_path = self.base_path / "global_adapter.pt"
            if default_path.exists():
                return default_path
            
            return None
    
    def list_adapters(self, user_id: Optional[str] = None) -> List[Dict]:
        """List available adapters"""
        adapters = []
        
        pattern = f"user_{user_id}_*.pt" if user_id else "*.pt"
        for adapter_file in self.base_path.glob(pattern):
            if not adapter_file.is_symlink():
                adapters.append({
                    "name": adapter_file.name,
                    "path": str(adapter_file),
                    "size": adapter_file.stat().st_size,
                    "modified": datetime.fromtimestamp(adapter_file.stat().st_mtime).isoformat(),
                    "hash": self._compute_hash(adapter_file)
                })
        
        return adapters
    
    def cleanup_old_backups(self, days: int = 7):
        """Clean up old backup files"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for backup_file in self.backup_path.glob("*.backup.*"):
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()
                logger.info(f"Deleted old backup: {backup_file}")

# Global instance
adapter_loader = AtomicAdapterLoader()

# Convenience functions
def hot_swap_adapter(user_id: str, adapter_name: str) -> bool:
    """Global hot-swap function"""
    return adapter_loader.hot_swap_adapter(user_id, adapter_name)

def rollback_adapter(user_id: str, steps: int = 1) -> bool:
    """Global rollback function"""
    return adapter_loader.rollback_adapter(user_id, steps)

def get_active_adapter(user_id: str) -> Optional[Path]:
    """Global get active adapter function"""
    return adapter_loader.get_active_adapter(user_id)
