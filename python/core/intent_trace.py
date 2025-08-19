#!/usr/bin/env python3
"""
Intent Trace Integration with Live Mesh Export
Triggers mesh export on intent closure and other high-impact events
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum

# Import mesh exporter
from mesh_summary_exporter import trigger_intent_closed_export, trigger_manual_export

logger = logging.getLogger(__name__)

# ============================================================================
# INTENT STATUS
# ============================================================================

class IntentStatus(Enum):
    """Intent lifecycle states."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    CONFIRMED = "CONFIRMED"
    SATISFIED = "SATISFIED"
    CLOSED = "CLOSED"
    ABANDONED = "ABANDONED"

# ============================================================================
# INTENT TRACE WITH EXPORT TRIGGERS
# ============================================================================

class IntentTraceWithExport:
    """
    Enhanced IntentTrace that triggers mesh export on significant events.
    """
    
    def __init__(self, 
                 memory_vault_dir: str = "memory_vault",
                 enable_live_export: bool = True):
        """
        Initialize intent trace with export capabilities.
        
        Args:
            memory_vault_dir: Directory for intent storage
            enable_live_export: Whether to trigger exports on events
        """
        self.memory_vault_dir = Path(memory_vault_dir)
        self.intents_dir = self.memory_vault_dir / "intents"
        self.intents_dir.mkdir(parents=True, exist_ok=True)
        self.enable_live_export = enable_live_export
        
        # Track open intents
        self.open_intents: Dict[str, Dict[str, Any]] = {}
        self._load_open_intents()
        
        logger.info(f"IntentTraceWithExport initialized, live export: {enable_live_export}")
    
    def _load_open_intents(self):
        """Load existing open intents from storage."""
        for intent_file in self.intents_dir.glob("*_open_intents.json"):
            try:
                with open(intent_file, 'r') as f:
                    data = json.load(f)
                    user_id = intent_file.stem.replace("_open_intents", "")
                    self.open_intents[user_id] = data.get("open_intents", [])
            except Exception as e:
                logger.error(f"Failed to load intents from {intent_file}: {e}")
    
    def _save_open_intents(self, user_id: str):
        """Save open intents for a user."""
        intent_file = self.intents_dir / f"{user_id}_open_intents.json"
        data = {
            "user_id": user_id,
            "updated_at": datetime.now().isoformat(),
            "open_intents": self.open_intents.get(user_id, [])
        }
        
        with open(intent_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def open_intent(self, 
                   user_id: str,
                   intent_id: str,
                   description: str,
                   intent_type: str = "unknown",
                   priority: str = "normal") -> Dict[str, Any]:
        """
        Open a new intent.
        
        Args:
            user_id: User identifier
            intent_id: Unique intent ID
            description: Intent description
            intent_type: Type of intent
            priority: Priority level
            
        Returns:
            Intent object
        """
        intent = {
            "id": intent_id,
            "description": description,
            "type": intent_type,
            "priority": priority,
            "status": IntentStatus.OPEN.value,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        }
        
        # Add to open intents
        if user_id not in self.open_intents:
            self.open_intents[user_id] = []
        
        self.open_intents[user_id].append(intent)
        self._save_open_intents(user_id)
        
        logger.info(f"Opened intent {intent_id} for user {user_id}")
        
        # Log to trace file
        self._log_intent_event(user_id, "intent_opened", intent)
        
        return intent
    
    def close_intent(self,
                    user_id: str,
                    intent_id: str,
                    resolution: str = "completed",
                    trigger_export: bool = True) -> bool:
        """
        Close/resolve an intent and optionally trigger mesh export.
        
        Args:
            user_id: User identifier
            intent_id: Intent to close
            resolution: How it was resolved
            trigger_export: Whether to trigger mesh export
            
        Returns:
            Success flag
        """
        # Find and remove intent
        if user_id not in self.open_intents:
            logger.warning(f"No open intents for user {user_id}")
            return False
        
        intent_found = None
        for i, intent in enumerate(self.open_intents[user_id]):
            if intent["id"] == intent_id:
                intent_found = intent
                self.open_intents[user_id].pop(i)
                break
        
        if not intent_found:
            logger.warning(f"Intent {intent_id} not found for user {user_id}")
            return False
        
        # Update intent status
        intent_found["status"] = IntentStatus.CLOSED.value
        intent_found["closed_at"] = datetime.now().isoformat()
        intent_found["resolution"] = resolution
        
        # Save updated intents
        self._save_open_intents(user_id)
        
        # Log event
        self._log_intent_event(user_id, "intent_closed", intent_found)
        
        logger.info(f"Closed intent {intent_id} for user {user_id}: {resolution}")
        
        # Trigger mesh export if enabled
        if trigger_export and self.enable_live_export:
            try:
                export_path = trigger_intent_closed_export(
                    user_id=user_id,
                    intent_id=intent_id,
                    intent_type=intent_found.get("type", "unknown")
                )
                if export_path:
                    logger.info(f"Triggered mesh export after intent closure: {export_path}")
            except Exception as e:
                logger.error(f"Failed to trigger export: {e}")
        
        return True
    
    def satisfy_intent(self, user_id: str, intent_id: str) -> bool:
        """
        Mark intent as satisfied (triggers export).
        
        Args:
            user_id: User identifier
            intent_id: Intent to satisfy
            
        Returns:
            Success flag
        """
        return self.close_intent(user_id, intent_id, "satisfied", trigger_export=True)
    
    def abandon_intent(self, user_id: str, intent_id: str) -> bool:
        """
        Abandon an intent (no export trigger).
        
        Args:
            user_id: User identifier
            intent_id: Intent to abandon
            
        Returns:
            Success flag
        """
        return self.close_intent(user_id, intent_id, "abandoned", trigger_export=False)
    
    def _log_intent_event(self, user_id: str, event: str, intent_data: Dict[str, Any]):
        """Log intent event to trace file."""
        trace_dir = self.memory_vault_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        
        trace_file = trace_dir / f"intent_trace_{user_id}.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "event": event,
            **intent_data
        }
        
        with open(trace_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_open_intents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all open intents for a user."""
        return self.open_intents.get(user_id, [])
    
    def bulk_close_session_intents(self, user_id: str, session_id: str):
        """
        Close all intents from a session (high-impact session end).
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        closed_count = 0
        intents_to_close = []
        
        # Find session intents
        for intent in self.open_intents.get(user_id, []):
            if intent.get("session_id") == session_id:
                intents_to_close.append(intent["id"])
        
        # Close them
        for intent_id in intents_to_close:
            if self.close_intent(user_id, intent_id, "session_ended", trigger_export=False):
                closed_count += 1
        
        # Trigger single export after bulk closure
        if closed_count > 0 and self.enable_live_export:
            logger.info(f"Closed {closed_count} intents from session {session_id}")
            try:
                from mesh_summary_exporter import get_global_exporter, ExportTrigger
                exporter = get_global_exporter()
                export_path = exporter.trigger_export(
                    user_id=user_id,
                    trigger=ExportTrigger.SESSION_END,
                    metadata={"session_id": session_id, "intents_closed": closed_count}
                )
                if export_path:
                    logger.info(f"Triggered export after session end: {export_path}")
            except Exception as e:
                logger.error(f"Failed to trigger session export: {e}")

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "IntentStatus",
    "IntentTraceWithExport"
]

# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Test intent lifecycle with export
    tracer = IntentTraceWithExport()
    
    # Open an intent
    user_id = "test_user"
    intent = tracer.open_intent(
        user_id=user_id,
        intent_id="test_intent_001",
        description="Optimize system performance",
        intent_type="optimization",
        priority="high"
    )
    
    print(f"Opened intent: {intent['id']}")
    
    # Simulate work...
    import time
    time.sleep(2)
    
    # Close intent (triggers export)
    success = tracer.close_intent(
        user_id=user_id,
        intent_id="test_intent_001",
        resolution="completed"
    )
    
    if success:
        print(f"Intent closed and export triggered!")
    
    # Check remaining open intents
    open_intents = tracer.get_open_intents(user_id)
    print(f"Remaining open intents: {len(open_intents)}")
