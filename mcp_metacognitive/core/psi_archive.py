"""
PsiArchive - Event logging and monitoring system
Migrated from mcp_server_arch
"""

import json
import os
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class PsiArchive:
    """Event logging and monitoring system with SSE support"""
    
    def __init__(self, file_path: Optional[str] = None):
        # Use default path if not provided
        if file_path is None:
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            file_path = str(data_dir / "psi_archive.log")
        
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Open log file
        self._file = open(file_path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._subscribers = []  # SSE subscribers
        self._counter = 0  # Event ID counter
        
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Record an event with timestamp and broadcast to subscribers"""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_type
        }
        
        if data is not None:
            event["data"] = data
        
        # Assign event ID
        self._counter += 1
        event["id"] = self._counter
        
        # Write to file and broadcast
        with self._lock:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()
            
            # Broadcast to subscribers
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(event)
                except:
                    pass
    
    def subscribe(self, queue: asyncio.Queue):
        """Add SSE subscriber"""
        with self._lock:
            self._subscribers.append(queue)
    
    def unsubscribe(self, queue: asyncio.Queue):
        """Remove SSE subscriber"""
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass
    
    def log_step(self, description: str):
        """Decorator to log function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.log_event("step_start", {"step": description})
                try:
                    result = func(*args, **kwargs)
                    self.log_event("step_end", {"step": description, "status": "success"})
                    return result
                except Exception as e:
                    self.log_event("step_end", {
                        "step": description, 
                        "status": "error", 
                        "error": str(e)
                    })
                    raise
            return wrapper
        return decorator
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from log file"""
        events = []
        try:
            with open(self._file.name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        events.append(json.loads(line.strip()))
                    except:
                        pass
        except:
            pass
        return events
    
    def close(self):
        """Close log file and clear subscribers"""
        with self._lock:
            try:
                self._file.close()
            finally:
                self._subscribers.clear()

# Global instance
psi_archive = PsiArchive()
