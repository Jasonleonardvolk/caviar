#!/usr/bin/env python3
"""
Adapter Rollback System
=======================
Instantly reverts to last stable adapter by updating manifest and symlinks.
Provides atomic rollback with full audit logging.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import os

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.adapter_loader_v5 import MetadataManager, backup_adapter
from python.core.user_context import get_user_context

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_ADAPTERS_DIR = "models/adapters"
DEFAULT_BACKUP_DIR = "models/adapters/backups"
ROLLBACK_LOG_DIR = Path("logs/rollback")

# ============================================================================
# ROLLBACK MANAGER
# ============================================================================

class RollbackManager:
    """Manages adapter rollback operations."""
    
    def __init__(self,
                 adapters_dir: str = DEFAULT_ADAPTERS_DIR,
                 backup_dir: str = DEFAULT_BACKUP_DIR):
        """
        Initialize rollback manager.
        
        Args:
            adapters_dir: Directory containing adapters
            backup_dir: Directory for backups
        """
        self.adapters_dir = Path(adapters_dir)
        self.backup_dir = Path(backup_dir)
        self.metadata_manager = MetadataManager()
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ROLLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    def rollback_adapter(self,
                        user_id: str,
                        version_steps: int = 1,
                        force: bool = False) -> Dict[str, Any]:
        """
        Rollback to previous adapter version.
        
        Args:
            user_id: User identifier
            version_steps: Number of versions to roll back
            force: Force rollback even if current is stable
            
        Returns:
            Rollback result dictionary
        """
        start_time = datetime.now()
        
        # Get current adapter
        current_adapter = self.metadata_manager.get_active_adapter(user_id)
        
        if not current_adapter:
            logger.warning(f"No active adapter for user {user_id}")
            return self._create_result(False, user_id, "No active adapter", start_time)
        
        # Check if rollback is needed
        if not force and current_adapter.get("metrics", {}).get("validation_score", 0) > 0.8:
            logger.info(f"Current adapter is stable, skipping rollback for {user_id}")
            return self._create_result(False, user_id, "Current adapter is stable", start_time)
        
        # Backup current adapter before rollback
        backup_path = self._backup_current_adapter(current_adapter["path"])
        
        # Get rollback target
        target_adapter = self._find_rollback_target(user_id, version_steps)
        
        if not target_adapter:
            logger.error(f"No valid rollback target for user {user_id}")
            return self._create_result(False, user_id, "No rollback target", start_time)
        
        # Perform atomic rollback
        success = self._perform_atomic_rollback(user_id, current_adapter, target_adapter)
        
        if success:
            # Update symlinks
            self._update_symlinks(user_id, target_adapter["path"])
            
            # Log rollback event
            self._log_rollback(user_id, current_adapter, target_adapter)
            
            # Update user context
            self._update_user_context(user_id, target_adapter["path"])
            
            logger.info(f"Successfully rolled back adapter for user {user_id}")
            
            return self._create_result(
                True,
                user_id,
                f"Rolled back from {current_adapter['adapter_id']} to {target_adapter['adapter_id']}",
                start_time,
                {
                    "from_adapter": current_adapter["adapter_id"],
                    "to_adapter": target_adapter["adapter_id"],
                    "backup_path": backup_path
                }
            )
        else:
            logger.error(f"Rollback failed for user {user_id}")
            return self._create_result(False, user_id, "Rollback operation failed", start_time)
    
    def _find_rollback_target(self, user_id: str, version_steps: int) -> Optional[Dict]:
        """Find the target adapter for rollback."""
        user_adapters = self.metadata_manager.list_user_adapters(user_id)
        
        if not user_adapters:
            return None
        
        # Sort by creation time (newest first)
        sorted_adapters = sorted(
            user_adapters,
            key=lambda x: x.get("created", ""),
            reverse=True
        )
        
        # Find stable adapter
        for i, adapter in enumerate(sorted_adapters):
            if i < version_steps:
                continue  # Skip recent versions
            
            # Check if adapter is stable
            validation_score = adapter.get("metrics", {}).get("validation_score", 0)
            if validation_score >= 0.8:
                return adapter
        
        # Fall back to oldest if no stable found
        if len(sorted_adapters) > version_steps:
            return sorted_adapters[version_steps]
        
        return None
    
    def _backup_current_adapter(self, adapter_path: str) -> Optional[str]:
        """Backup current adapter before rollback."""
        try:
            if not Path(adapter_path).exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{Path(adapter_path).stem}_rollback_{timestamp}.pt"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(adapter_path, backup_path)
            logger.info(f"Backed up adapter to {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup adapter: {e}")
            return None
    
    def _perform_atomic_rollback(self,
                                user_id: str,
                                current: Dict,
                                target: Dict) -> bool:
        """
        Perform atomic rollback operation.
        
        This is the critical section - must be atomic.
        """
        try:
            # Load metadata
            meta = self.metadata_manager.load_metadata()
            
            if user_id not in meta:
                return False
            
            # Update metadata atomically
            for adapter in meta[user_id]:
                if adapter["adapter_id"] == current["adapter_id"]:
                    adapter["active"] = False
                    adapter["history"].append({
                        "event": "deactivated_rollback",
                        "timestamp": datetime.now().isoformat()
                    })
                elif adapter["adapter_id"] == target["adapter_id"]:
                    adapter["active"] = True
                    adapter["history"].append({
                        "event": "activated_rollback",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    adapter["active"] = False
            
            # Save metadata atomically
            self.metadata_manager.save_metadata(meta)
            
            return True
            
        except Exception as e:
            logger.error(f"Atomic rollback failed: {e}")
            return False
    
    def _update_symlinks(self, user_id: str, target_path: str):
        """Update symlinks to point to rollback target."""
        try:
            symlink_path = self.adapters_dir / f"user_{user_id}_active.pt"
            
            # Remove existing symlink
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            
            # Create new symlink
            if os.name == 'nt':  # Windows
                import subprocess
                subprocess.run(
                    ['mklink', str(symlink_path), str(target_path)],
                    shell=True,
                    check=True
                )
            else:  # Unix-like
                symlink_path.symlink_to(target_path)
            
            logger.info(f"Updated symlink: {symlink_path} -> {target_path}")
            
        except Exception as e:
            logger.error(f"Failed to update symlinks: {e}")
    
    def _update_user_context(self, user_id: str, adapter_path: str):
        """Update user context with rollback adapter."""
        try:
            context = get_user_context(user_id)
            context.active_adapter = adapter_path
            
            # Update active session if exists
            active_session = context.get_active_session()
            if active_session:
                active_session.adapter_path = adapter_path
            
            logger.info(f"Updated user context for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user context: {e}")
    
    def _log_rollback(self, user_id: str, from_adapter: Dict, to_adapter: Dict):
        """Log rollback event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "adapter_rollback",
            "user_id": user_id,
            "from_adapter": from_adapter["adapter_id"],
            "to_adapter": to_adapter["adapter_id"],
            "from_score": from_adapter.get("metrics", {}).get("validation_score"),
            "to_score": to_adapter.get("metrics", {}).get("validation_score")
        }
        
        log_file = ROLLBACK_LOG_DIR / f"rollback_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log rollback: {e}")
    
    def _create_result(self,
                      success: bool,
                      user_id: str,
                      message: str,
                      start_time: datetime,
                      details: Optional[Dict] = None) -> Dict[str, Any]:
        """Create rollback result dictionary."""
        return {
            "success": success,
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "details": details or {}
        }
    
    def get_rollback_history(self, user_id: str) -> List[Dict]:
        """Get rollback history for user."""
        history = []
        
        # Read from rollback logs
        for log_file in ROLLBACK_LOG_DIR.glob("rollback_*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("user_id") == user_id:
                            history.append(entry)
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return history
    
    def can_rollback(self, user_id: str) -> bool:
        """Check if rollback is possible for user."""
        user_adapters = self.metadata_manager.list_user_adapters(user_id)
        
        # Need at least 2 adapters to rollback
        return len(user_adapters) >= 2

# ============================================================================
# MAIN ROLLBACK FUNCTION
# ============================================================================

def rollback_adapter(user_id: str,
                    version_steps: int = 1,
                    force: bool = False) -> bool:
    """
    Main function to rollback adapter.
    
    Args:
        user_id: User identifier
        version_steps: Number of versions to roll back
        force: Force rollback even if current is stable
        
    Returns:
        True if rollback successful
    """
    manager = RollbackManager()
    result = manager.rollback_adapter(user_id, version_steps, force)
    return result["success"]

def emergency_rollback_all_users() -> Dict[str, bool]:
    """
    Emergency rollback for all users.
    
    Returns:
        Dictionary mapping user_id to success status
    """
    manager = RollbackManager()
    meta = manager.metadata_manager.load_metadata()
    
    results = {}
    for user_id in meta.keys():
        if user_id != "global":  # Skip global adapter
            result = manager.rollback_adapter(user_id, force=True)
            results[user_id] = result["success"]
    
    return results

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for adapter rollback."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapter Rollback")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback adapter")
    rollback_parser.add_argument("--user_id", required=True, help="User ID")
    rollback_parser.add_argument("--steps", type=int, default=1,
                                help="Version steps to roll back")
    rollback_parser.add_argument("--force", action="store_true",
                                help="Force rollback")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show rollback history")
    history_parser.add_argument("--user_id", required=True, help="User ID")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check if rollback possible")
    check_parser.add_argument("--user_id", required=True, help="User ID")
    
    # Emergency command
    emergency_parser = subparsers.add_parser("emergency",
                                            help="Emergency rollback all users")
    emergency_parser.add_argument("--confirm", action="store_true",
                                 help="Confirm emergency rollback")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = RollbackManager()
    
    if args.command == "rollback":
        result = manager.rollback_adapter(
            args.user_id,
            args.steps,
            args.force
        )
        
        print("\n" + "="*60)
        print("ROLLBACK RESULT")
        print("="*60)
        print(f"User: {args.user_id}")
        print(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Message: {result['message']}")
        if result['details']:
            print("Details:")
            for key, value in result['details'].items():
                print(f"  {key}: {value}")
        print("="*60)
    
    elif args.command == "history":
        history = manager.get_rollback_history(args.user_id)
        
        print(f"\nRollback History for '{args.user_id}':")
        if history:
            for entry in history[:10]:  # Last 10 entries
                print(f"  {entry['timestamp']}: {entry['from_adapter']} -> {entry['to_adapter']}")
        else:
            print("  No rollback history")
    
    elif args.command == "check":
        can_rollback = manager.can_rollback(args.user_id)
        print(f"User '{args.user_id}' can rollback: {can_rollback}")
    
    elif args.command == "emergency":
        if not args.confirm:
            print("Emergency rollback requires --confirm flag")
        else:
            print("Initiating emergency rollback for all users...")
            results = emergency_rollback_all_users()
            
            print("\nEmergency Rollback Results:")
            for user_id, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                print(f"  {user_id}: {status}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
