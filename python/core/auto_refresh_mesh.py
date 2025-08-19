#!/usr/bin/env python3
"""
Auto-Refreshing Mesh Context Manager
Ensures mesh summaries are always up-to-date
"""

import json
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshUpdateHandler(FileSystemEventHandler):
    """Watches for mesh changes and triggers auto-refresh"""
    
    def __init__(self, mesh_manager):
        self.mesh_manager = mesh_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('_mesh.json'):
            # Extract user_id from filename
            filename = Path(event.src_path).name
            if filename.startswith('user_') and filename.endswith('_mesh.json'):
                user_id = filename.replace('user_', '').replace('_mesh.json', '')
                logger.info(f"Mesh change detected for user {user_id}")
                self.mesh_manager.refresh_summary(user_id)

class AutoRefreshMeshManager:
    """Manages mesh contexts with automatic refresh on updates"""
    
    def __init__(self, mesh_path: str = "data/mesh_contexts"):
        self.mesh_path = Path(mesh_path)
        self.mesh_path.mkdir(parents=True, exist_ok=True)
        
        self.summary_cache = {}
        self.refresh_lock = threading.RLock()
        self.observer = None
        
        # Start file watcher
        self.start_watcher()
        
        # Start periodic refresh daemon
        self.start_refresh_daemon()
    
    def start_watcher(self):
        """Start watching for mesh file changes"""
        self.observer = Observer()
        handler = MeshUpdateHandler(self)
        self.observer.schedule(handler, str(self.mesh_path), recursive=False)
        self.observer.start()
        logger.info(f"Started mesh file watcher on {self.mesh_path}")
    
    def start_refresh_daemon(self):
        """Start background daemon for periodic refresh"""
        def daemon_loop():
            while True:
                time.sleep(300)  # Every 5 minutes
                self.refresh_all_stale()
        
        daemon = threading.Thread(target=daemon_loop, daemon=True)
        daemon.start()
        logger.info("Started mesh refresh daemon (5 min interval)")
    
    def update_mesh(self, user_id: str, change: Dict[str, Any]) -> bool:
        """Update mesh and automatically refresh summary"""
        with self.refresh_lock:
            try:
                # Load existing mesh
                mesh_file = self.mesh_path / f"user_{user_id}_mesh.json"
                if mesh_file.exists():
                    with open(mesh_file, 'r') as f:
                        mesh_data = json.load(f)
                else:
                    mesh_data = {
                        "user_id": user_id,
                        "concepts": [],
                        "relationships": [],
                        "metadata": {
                            "created": datetime.now().isoformat(),
                            "version": "1.0.0"
                        }
                    }
                
                # Apply change
                if change["action"] == "add_concept":
                    mesh_data["concepts"].append(change["data"])
                elif change["action"] == "add_relationship":
                    mesh_data["relationships"].append(change["data"])
                elif change["action"] == "update_metadata":
                    mesh_data["metadata"].update(change["data"])
                
                # Update timestamp
                mesh_data["metadata"]["updated"] = datetime.now().isoformat()
                
                # Save mesh
                self.save_mesh(user_id, mesh_data)
                
                # Auto-refresh summary
                self.refresh_summary(user_id)
                
                # Log event
                self.log_mesh_event(user_id, "mesh_updated", change)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to update mesh for {user_id}: {e}")
                return False
    
    def save_mesh(self, user_id: str, mesh_data: Dict):
        """Save mesh data atomically"""
        mesh_file = self.mesh_path / f"user_{user_id}_mesh.json"
        temp_file = mesh_file.with_suffix('.tmp')
        
        # Write to temp file
        with open(temp_file, 'w') as f:
            json.dump(mesh_data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(mesh_file)
        
        logger.info(f"Saved mesh for user {user_id}")
    
    def refresh_summary(self, user_id: str) -> Dict:
        """Generate and cache fresh mesh summary"""
        with self.refresh_lock:
            try:
                # Load full mesh
                mesh_file = self.mesh_path / f"user_{user_id}_mesh.json"
                if not mesh_file.exists():
                    logger.warning(f"No mesh found for user {user_id}")
                    return {}
                
                with open(mesh_file, 'r') as f:
                    mesh_data = json.load(f)
                
                # Generate summary
                summary = self.generate_summary(mesh_data)
                
                # Save summary
                summary_file = self.mesh_path / f"user_{user_id}_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Update cache
                self.summary_cache[user_id] = {
                    "summary": summary,
                    "timestamp": time.time(),
                    "hash": self.compute_hash(summary)
                }
                
                logger.info(f"Refreshed summary for user {user_id}")
                
                # Notify inference system
                self.notify_inference_system(user_id, summary)
                
                return summary
                
            except Exception as e:
                logger.error(f"Failed to refresh summary for {user_id}: {e}")
                return {}
    
    def generate_summary(self, mesh_data: Dict) -> Dict:
        """Generate concise summary from full mesh"""
        summary = {
            "user_id": mesh_data.get("user_id"),
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "total_concepts": len(mesh_data.get("concepts", [])),
                "total_relationships": len(mesh_data.get("relationships", [])),
                "last_updated": mesh_data.get("metadata", {}).get("updated")
            },
            "top_concepts": [],
            "recent_concepts": [],
            "active_themes": [],
            "context_prompt": ""
        }
        
        # Extract top concepts (by frequency/importance)
        concepts = mesh_data.get("concepts", [])
        if concepts:
            # Sort by importance score if available
            sorted_concepts = sorted(
                concepts,
                key=lambda x: x.get("importance", 0),
                reverse=True
            )
            summary["top_concepts"] = [c["name"] for c in sorted_concepts[:10]]
            
            # Recent concepts (last 5)
            recent = sorted(
                concepts,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            summary["recent_concepts"] = [c["name"] for c in recent[:5]]
        
        # Extract active themes
        relationships = mesh_data.get("relationships", [])
        theme_counts = {}
        for rel in relationships:
            theme = rel.get("type", "unknown")
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        summary["active_themes"] = sorted(
            theme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Generate context prompt for injection
        summary["context_prompt"] = self.build_context_prompt(summary)
        
        return summary
    
    def build_context_prompt(self, summary: Dict) -> str:
        """Build context prompt for inference injection"""
        prompt_parts = []
        
        if summary["top_concepts"]:
            prompt_parts.append(
                f"User's key interests: {', '.join(summary['top_concepts'][:5])}"
            )
        
        if summary["recent_concepts"]:
            prompt_parts.append(
                f"Recent topics: {', '.join(summary['recent_concepts'][:3])}"
            )
        
        if summary["active_themes"]:
            themes = [t[0] for t in summary["active_themes"][:3]]
            prompt_parts.append(f"Active themes: {', '.join(themes)}")
        
        return " | ".join(prompt_parts) if prompt_parts else ""
    
    def get_summary(self, user_id: str, max_age: int = 60) -> Dict:
        """Get cached summary or refresh if stale"""
        # Check cache
        if user_id in self.summary_cache:
            cached = self.summary_cache[user_id]
            age = time.time() - cached["timestamp"]
            
            if age < max_age:
                logger.debug(f"Using cached summary for {user_id} (age: {age:.1f}s)")
                return cached["summary"]
        
        # Refresh if not cached or stale
        return self.refresh_summary(user_id)
    
    def refresh_all_stale(self, max_age: int = 300):
        """Refresh all summaries older than max_age seconds"""
        current_time = time.time()
        stale_users = []
        
        for user_id, cached in self.summary_cache.items():
            age = current_time - cached["timestamp"]
            if age > max_age:
                stale_users.append(user_id)
        
        if stale_users:
            logger.info(f"Refreshing {len(stale_users)} stale summaries")
            for user_id in stale_users:
                self.refresh_summary(user_id)
    
    def compute_hash(self, data: Dict) -> str:
        """Compute hash of summary for change detection"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def notify_inference_system(self, user_id: str, summary: Dict):
        """Notify inference system of summary update"""
        try:
            # Send notification via HTTP
            import requests
            response = requests.post(
                "http://localhost:8001/api/saigon/mesh/notify",
                json={
                    "user_id": user_id,
                    "summary_hash": self.compute_hash(summary),
                    "timestamp": datetime.now().isoformat()
                },
                timeout=1
            )
            if response.status_code == 200:
                logger.debug(f"Notified inference system of mesh update for {user_id}")
        except Exception as e:
            logger.debug(f"Could not notify inference system: {e}")
    
    def log_mesh_event(self, user_id: str, event_type: str, data: Dict):
        """Log mesh event for audit trail"""
        event_file = self.mesh_path / "mesh_events.jsonl"
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "type": event_type,
            "data": data
        }
        
        with open(event_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def validate_summary_freshness(self, user_id: str) -> bool:
        """Check if summary is fresh (for diagnostics)"""
        mesh_file = self.mesh_path / f"user_{user_id}_mesh.json"
        summary_file = self.mesh_path / f"user_{user_id}_summary.json"
        
        if not mesh_file.exists() or not summary_file.exists():
            return False
        
        mesh_mtime = mesh_file.stat().st_mtime
        summary_mtime = summary_file.stat().st_mtime
        
        # Summary should be newer than mesh
        return summary_mtime >= mesh_mtime
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information"""
        return {
            "cache_size": len(self.summary_cache),
            "mesh_count": len(list(self.mesh_path.glob("*_mesh.json"))),
            "summary_count": len(list(self.mesh_path.glob("*_summary.json"))),
            "observer_alive": self.observer.is_alive() if self.observer else False,
            "stale_summaries": sum(
                1 for user_id in self.summary_cache
                if not self.validate_summary_freshness(user_id)
            )
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("Mesh manager cleanup complete")

# Global instance
mesh_manager = AutoRefreshMeshManager()

# Convenience functions
def update_mesh(user_id: str, change: Dict) -> bool:
    """Update mesh and auto-refresh summary"""
    return mesh_manager.update_mesh(user_id, change)

def get_mesh_summary(user_id: str) -> Dict:
    """Get current mesh summary"""
    return mesh_manager.get_summary(user_id)

def refresh_mesh_summary(user_id: str) -> Dict:
    """Force refresh mesh summary"""
    return mesh_manager.refresh_summary(user_id)
