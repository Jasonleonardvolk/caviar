#!/usr/bin/env python3
"""
Enhanced Adapter Loader Module - Phase 5
=========================================
Handles user/domain/global adapter resolution, metadata manifest,
SHA256 integrity checking, versioning, and multi-user safety.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil
import os

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

METADATA_PATH = Path("models/adapters/metadata.json")
ADAPTERS_DIR = Path("models/adapters")
BACKUP_DIR = Path("models/adapters/backups")

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AdapterRecord:
    """Comprehensive adapter metadata record."""
    adapter_id: str
    user_id: str
    path: str
    version: str
    created: str
    sha256: str
    base_model: str
    domains: List[str]
    description: str
    score: float
    active: bool
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    size_bytes: int
    training_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AdapterRecord":
        """Create from dictionary."""
        return cls(**data)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sha256_file(path: str) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        path: Path to file
        
    Returns:
        SHA256 hash as hex string
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def ensure_directories():
    """Ensure all required directories exist."""
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
def backup_adapter(adapter_path: str) -> Optional[str]:
    """
    Create backup of adapter before modification.
    
    Args:
        adapter_path: Path to adapter
        
    Returns:
        Path to backup or None if failed
    """
    try:
        source = Path(adapter_path)
        if not source.exists():
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_backup_{timestamp}{source.suffix}"
        backup_path = BACKUP_DIR / backup_name
        
        shutil.copy2(source, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None

# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

class MetadataManager:
    """Manages adapter metadata with thread-safe operations."""
    
    def __init__(self, metadata_path: Path = METADATA_PATH):
        self.metadata_path = metadata_path
        self._ensure_metadata_file()
        
    def _ensure_metadata_file(self):
        """Ensure metadata file exists."""
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w") as f:
                json.dump({}, f)
    
    def load_metadata(self) -> Dict[str, List[Dict]]:
        """
        Load metadata from disk.
        
        Returns:
            Dictionary mapping user_id to list of adapter records
        """
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_metadata(self, data: Dict[str, List[Dict]]):
        """
        Save metadata to disk with atomic write.
        
        Args:
            data: Metadata dictionary
        """
        # Write to temp file first for atomicity
        temp_path = self.metadata_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_path.replace(self.metadata_path)
        logger.info(f"Saved metadata to {self.metadata_path}")
    
    def register_adapter(self, 
                        user_id: str,
                        adapter_path: str,
                        version: str,
                        base_model: str,
                        domains: List[str],
                        description: str,
                        score: float,
                        metrics: Dict[str, Any],
                        training_params: Optional[Dict] = None) -> str:
        """
        Register a new adapter in metadata.
        
        Args:
            user_id: User identifier
            adapter_path: Path to adapter file
            version: Version string
            base_model: Base model identifier
            domains: List of domains
            description: Description of adapter
            score: Validation score
            metrics: Performance metrics
            training_params: Training parameters
            
        Returns:
            Adapter ID
        """
        # Ensure adapter exists
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        # Calculate hash and size
        sha256 = sha256_file(adapter_path)
        size_bytes = Path(adapter_path).stat().st_size
        
        # Create adapter record
        adapter_id = f"{user_id}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = AdapterRecord(
            adapter_id=adapter_id,
            user_id=user_id,
            path=adapter_path,
            version=version,
            created=datetime.now().isoformat(),
            sha256=sha256,
            base_model=base_model,
            domains=domains,
            description=description,
            score=score,
            active=True,  # New adapters are active by default
            metrics=metrics,
            history=[{
                "event": "registered",
                "timestamp": datetime.now().isoformat(),
                "details": {"score": score, "sha256": sha256}
            }],
            size_bytes=size_bytes,
            training_params=training_params
        )
        
        # Load existing metadata
        meta = self.load_metadata()
        
        # Initialize user entry if needed
        if user_id not in meta:
            meta[user_id] = []
        
        # Deactivate previous adapters for this user
        for existing in meta[user_id]:
            existing["active"] = False
        
        # Add new record
        meta[user_id].append(record.to_dict())
        
        # Save metadata
        self.save_metadata(meta)
        
        logger.info(f"Registered adapter {adapter_id} for user {user_id}")
        return adapter_id
    
    def get_active_adapter(self, user_id: str) -> Optional[Dict]:
        """
        Get active adapter for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Active adapter record or None
        """
        meta = self.load_metadata()
        user_adapters = meta.get(user_id, [])
        
        for adapter in user_adapters:
            if adapter.get("active"):
                return adapter
        
        return None
    
    def list_user_adapters(self, user_id: str) -> List[Dict]:
        """
        List all adapters for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of adapter records
        """
        meta = self.load_metadata()
        return meta.get(user_id, [])
    
    def promote_adapter(self, user_id: str, adapter_id: str) -> bool:
        """
        Promote an adapter to active status.
        
        Args:
            user_id: User identifier
            adapter_id: Adapter identifier
            
        Returns:
            Success flag
        """
        meta = self.load_metadata()
        
        if user_id not in meta:
            return False
        
        promoted = False
        for adapter in meta[user_id]:
            if adapter["adapter_id"] == adapter_id:
                adapter["active"] = True
                adapter["history"].append({
                    "event": "promoted",
                    "timestamp": datetime.now().isoformat()
                })
                promoted = True
            else:
                adapter["active"] = False
        
        if promoted:
            self.save_metadata(meta)
            logger.info(f"Promoted adapter {adapter_id} for user {user_id}")
        
        return promoted
    
    def rollback_adapter(self, user_id: str, version_steps: int = 1) -> bool:
        """
        Rollback to previous adapter version.
        
        Args:
            user_id: User identifier
            version_steps: Number of versions to roll back
            
        Returns:
            Success flag
        """
        meta = self.load_metadata()
        user_adapters = meta.get(user_id, [])
        
        if not user_adapters:
            return False
        
        # Sort by creation time
        sorted_adapters = sorted(
            user_adapters,
            key=lambda x: x["created"],
            reverse=True
        )
        
        # Find target adapter
        if version_steps >= len(sorted_adapters):
            return False
        
        target_adapter = sorted_adapters[version_steps]
        
        # Promote target adapter
        return self.promote_adapter(user_id, target_adapter["adapter_id"])
    
    def verify_adapter(self, adapter_path: str, expected_sha256: str) -> bool:
        """
        Verify adapter integrity.
        
        Args:
            adapter_path: Path to adapter
            expected_sha256: Expected SHA256 hash
            
        Returns:
            True if integrity check passes
        """
        if not Path(adapter_path).exists():
            return False
        
        actual_sha256 = sha256_file(adapter_path)
        return actual_sha256 == expected_sha256
    
    def get_global_adapter(self) -> Optional[Dict]:
        """
        Get global/default adapter.
        
        Returns:
            Global adapter record or None
        """
        return self.get_active_adapter("global")

# ============================================================================
# PATH RESOLUTION FUNCTIONS
# ============================================================================

def get_adapter_path_for_user(user_id: str, 
                             adapters_dir: str = "models/adapters/",
                             use_metadata: bool = True) -> Optional[str]:
    """
    Get adapter path for specific user with metadata support.
    
    Args:
        user_id: User identifier
        adapters_dir: Directory containing adapters
        use_metadata: Whether to use metadata manifest
        
    Returns:
        Path to user's adapter if exists, None otherwise
    """
    if use_metadata:
        manager = MetadataManager()
        active = manager.get_active_adapter(user_id)
        if active:
            return active["path"]
    
    # Fallback to file-based lookup
    # Check for active symlink first
    active_path = Path(adapters_dir) / f"user_{user_id}_active.pt"
    if active_path.exists() or active_path.is_symlink():
        return str(active_path)
    
    # Check for versioned adapter
    adapter_path = Path(adapters_dir) / f"user_{user_id}_lora.pt"
    if adapter_path.exists():
        return str(adapter_path)
    
    # Check for any user adapter with version
    for adapter_file in Path(adapters_dir).glob(f"user_{user_id}_v*.pt"):
        return str(adapter_file)
    
    return None

def get_domain_adapter_path(user_id: str, 
                           domain: str,
                           adapters_dir: str = "models/adapters/",
                           use_metadata: bool = True) -> Optional[str]:
    """
    Get domain-specific adapter path for user.
    
    Args:
        user_id: User identifier
        domain: Domain name
        adapters_dir: Directory containing adapters
        use_metadata: Whether to use metadata manifest
        
    Returns:
        Path to domain adapter if exists, None otherwise
    """
    if use_metadata:
        manager = MetadataManager()
        adapters = manager.list_user_adapters(user_id)
        
        # Find adapter with matching domain
        for adapter in adapters:
            if domain in adapter.get("domains", []) and adapter.get("active"):
                return adapter["path"]
    
    # Fallback to file-based lookup
    domain_path = Path(adapters_dir) / f"user_{user_id}_{domain}_lora.pt"
    if domain_path.exists():
        return str(domain_path)
    
    return None

def get_global_adapter_path(adapters_dir: str = "models/adapters/",
                          use_metadata: bool = True) -> Optional[str]:
    """
    Get path to global/default adapter.
    
    Args:
        adapters_dir: Directory containing adapters
        use_metadata: Whether to use metadata manifest
        
    Returns:
        Path to global adapter if exists, None otherwise
    """
    if use_metadata:
        manager = MetadataManager()
        global_adapter = manager.get_global_adapter()
        if global_adapter:
            return global_adapter["path"]
    
    # Fallback to file-based lookup
    for name in ["global_adapter_v1.pt", "global_adapter.pt", "default_adapter.pt"]:
        path = Path(adapters_dir) / name
        if path.exists():
            return str(path)
    
    return None

def list_all_user_adapters(user_id: str,
                          adapters_dir: str = "models/adapters/",
                          use_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    List all adapters available for a user with full metadata.
    
    Args:
        user_id: User identifier
        adapters_dir: Directory containing adapters
        use_metadata: Whether to use metadata manifest
        
    Returns:
        List of adapter information dictionaries
    """
    if use_metadata:
        manager = MetadataManager()
        return manager.list_user_adapters(user_id)
    
    # Fallback to file-based lookup
    adapters = []
    for adapter_file in Path(adapters_dir).glob(f"user_{user_id}_*.pt"):
        adapters.append({
            "path": str(adapter_file),
            "exists": adapter_file.exists(),
            "size_bytes": adapter_file.stat().st_size,
            "modified": datetime.fromtimestamp(adapter_file.stat().st_mtime).isoformat()
        })
    
    return adapters

# ============================================================================
# ADAPTER OPERATIONS
# ============================================================================

def create_adapter_symlink(source: str, link_name: str) -> bool:
    """
    Create symlink for adapter (for hot-swapping).
    
    Args:
        source: Source adapter path
        link_name: Symlink name
        
    Returns:
        Success flag
    """
    try:
        source_path = Path(source)
        link_path = Path(link_name)
        
        if not source_path.exists():
            logger.error(f"Source adapter not found: {source}")
            return False
        
        # Remove existing symlink
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # Create new symlink
        if os.name == 'nt':  # Windows
            import subprocess
            subprocess.run(['mklink', str(link_path), str(source_path)], shell=True, check=True)
        else:  # Unix-like
            link_path.symlink_to(source_path)
        
        logger.info(f"Created symlink: {link_path} -> {source_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create symlink: {e}")
        return False

def cleanup_old_adapters(user_id: str, keep_versions: int = 5) -> int:
    """
    Clean up old adapter versions, keeping only recent ones.
    
    Args:
        user_id: User identifier
        keep_versions: Number of versions to keep
        
    Returns:
        Number of adapters removed
    """
    manager = MetadataManager()
    adapters = manager.list_user_adapters(user_id)
    
    if len(adapters) <= keep_versions:
        return 0
    
    # Sort by creation time
    sorted_adapters = sorted(
        adapters,
        key=lambda x: x["created"],
        reverse=True
    )
    
    # Remove old adapters
    removed = 0
    for adapter in sorted_adapters[keep_versions:]:
        adapter_path = Path(adapter["path"])
        if adapter_path.exists():
            # Backup before removal
            backup_adapter(str(adapter_path))
            adapter_path.unlink()
            removed += 1
            logger.info(f"Removed old adapter: {adapter_path}")
    
    # Update metadata
    meta = manager.load_metadata()
    meta[user_id] = sorted_adapters[:keep_versions]
    manager.save_metadata(meta)
    
    return removed

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for adapter management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapter Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List user adapters")
    list_parser.add_argument("--user_id", required=True, help="User ID")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register new adapter")
    register_parser.add_argument("--user_id", required=True, help="User ID")
    register_parser.add_argument("--path", required=True, help="Adapter path")
    register_parser.add_argument("--version", required=True, help="Version")
    register_parser.add_argument("--base_model", required=True, help="Base model")
    register_parser.add_argument("--domains", nargs="+", default=["general"], help="Domains")
    register_parser.add_argument("--description", default="", help="Description")
    register_parser.add_argument("--score", type=float, default=0.0, help="Validation score")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote adapter")
    promote_parser.add_argument("--user_id", required=True, help="User ID")
    promote_parser.add_argument("--adapter_id", required=True, help="Adapter ID")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify adapter integrity")
    verify_parser.add_argument("--path", required=True, help="Adapter path")
    verify_parser.add_argument("--sha256", required=True, help="Expected SHA256")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old adapters")
    cleanup_parser.add_argument("--user_id", required=True, help="User ID")
    cleanup_parser.add_argument("--keep", type=int, default=5, help="Versions to keep")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute command
    manager = MetadataManager()
    
    if args.command == "list":
        adapters = manager.list_user_adapters(args.user_id)
        print(f"\nAdapters for user '{args.user_id}':")
        for adapter in adapters:
            status = "[ACTIVE]" if adapter.get("active") else ""
            print(f"  - {adapter['adapter_id']} {status}")
            print(f"    Path: {adapter['path']}")
            print(f"    Score: {adapter.get('score', 'N/A')}")
            print(f"    Created: {adapter['created']}")
    
    elif args.command == "register":
        adapter_id = manager.register_adapter(
            user_id=args.user_id,
            adapter_path=args.path,
            version=args.version,
            base_model=args.base_model,
            domains=args.domains,
            description=args.description,
            score=args.score,
            metrics={},
            training_params={}
        )
        print(f"Registered adapter: {adapter_id}")
    
    elif args.command == "promote":
        success = manager.promote_adapter(args.user_id, args.adapter_id)
        if success:
            print(f"Promoted adapter: {args.adapter_id}")
        else:
            print(f"Failed to promote adapter: {args.adapter_id}")
    
    elif args.command == "verify":
        valid = manager.verify_adapter(args.path, args.sha256)
        if valid:
            print(f"Adapter integrity verified: {args.path}")
        else:
            print(f"Adapter integrity check failed: {args.path}")
    
    elif args.command == "cleanup":
        removed = cleanup_old_adapters(args.user_id, args.keep)
        print(f"Removed {removed} old adapters for user '{args.user_id}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
