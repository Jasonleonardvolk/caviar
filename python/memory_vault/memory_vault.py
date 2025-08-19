#!/usr/bin/env python3
"""
Memory Vault - Persistent flat-file storage for TORI
Stores sessions, actions, intent traces, and cognitive events
No database required - pure JSON/JSONL storage
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

class MemoryVault:
    """
    Flat-file storage system for TORI's memory.
    Uses JSONL for append-only logs and JSON for state snapshots.
    """
    
    def __init__(self, 
                 base_dir: str = "memory_vault",
                 session_id: Optional[str] = None,
                 auto_create: bool = True):
        """
        Initialize Memory Vault.
        
        Args:
            base_dir: Base directory for storage
            session_id: Current session ID
            auto_create: Auto-create directories
        """
        self.base_dir = Path(base_dir)
        
        # Create directory structure
        if auto_create:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different data types
        self.sessions_dir = self.base_dir / "sessions"
        self.traces_dir = self.base_dir / "traces"
        self.actions_dir = self.base_dir / "actions"
        self.metrics_dir = self.base_dir / "metrics"
        self.snapshots_dir = self.base_dir / "snapshots"
        
        if auto_create:
            for dir_path in [self.sessions_dir, self.traces_dir, 
                           self.actions_dir, self.metrics_dir, self.snapshots_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set session
        if session_id:
            self.current_session = session_id
        else:
            # Generate session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = f"session_{timestamp}"
        
        # Session metadata
        self.session_metadata = {
            "session_id": self.current_session,
            "started_at": datetime.now().isoformat(),
            "events": 0,
            "intents_opened": 0,
            "intents_closed": 0
        }
        
        # Write session start
        self._log_session_event({
            "event": "session_start",
            "session_id": self.current_session,
            "timestamp": self.session_metadata["started_at"]
        })
        
        logger.info(f"MemoryVault initialized: {self.base_dir} (session: {self.current_session})")
    
    # ========================================================================
    # SESSION LOGGING
    # ========================================================================
    
    def log_conversation(self, role: str, text: str, metadata: Dict[str, Any] = None):
        """
        Log a conversation turn.
        
        Args:
            role: 'user' or 'assistant'
            text: Message text
            metadata: Optional metadata
        """
        event = {
            "event": "conversation",
            "role": role,
            "text": text,
            "turn": self.session_metadata["events"],
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            event["metadata"] = metadata
        
        self._log_session_event(event)
    
    def log_action(self, action_type: str, target: str = None, metadata: Dict[str, Any] = None):
        """
        Log a user or system action.
        
        Args:
            action_type: Type of action
            target: Action target
            metadata: Optional metadata
        """
        event = {
            "event": "action",
            "action_type": action_type,
            "target": target,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            event["metadata"] = metadata
        
        # Log to actions file
        file_path = self.actions_dir / f"{self.current_session}.jsonl"
        self._append_jsonl(file_path, event)
    
    # ========================================================================
    # INTENT TRACKING
    # ========================================================================
    
    def log_intent_opened(self, trace):
        """Log an intent being opened."""
        event = {
            "event": "intent_opened",
            "trace_id": trace.trace_id,
            "intent_type": trace.intent_type,
            "description": trace.description,
            "confidence": trace.confidence,
            "turn_opened": trace.turn_opened,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to traces file
        file_path = self.traces_dir / f"{self.current_session}.jsonl"
        self._append_jsonl(file_path, event)
        
        # Update session metadata
        self.session_metadata["intents_opened"] += 1
        
        logger.debug(f"Logged intent opened: {trace.trace_id}")
    
    def log_intent_closed(self, trace):
        """Log an intent being closed."""
        event = {
            "event": "intent_closed",
            "trace_id": trace.trace_id,
            "closure_state": trace.closure_state.value,
            "closure_trigger": trace.closure_trigger.value if trace.closure_trigger else None,
            "closure_confidence": trace.closure_confidence,
            "turn_closed": trace.turn_closed,
            "duration_seconds": trace.age_seconds(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to traces file
        file_path = self.traces_dir / f"{self.current_session}.jsonl"
        self._append_jsonl(file_path, event)
        
        # Update session metadata
        self.session_metadata["intents_closed"] += 1
        
        logger.debug(f"Logged intent closed: {trace.trace_id}")
    
    def log_nudge(self, trace_id: str, message: str, delivered: bool = False):
        """Log a nudge being generated or delivered."""
        event = {
            "event": "nudge",
            "trace_id": trace_id,
            "message": message,
            "delivered": delivered,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to traces file
        file_path = self.traces_dir / f"{self.current_session}.jsonl"
        self._append_jsonl(file_path, event)
        
        logger.debug(f"Logged nudge for: {trace_id}")
    
    # ========================================================================
    # METRICS AND ANALYTICS
    # ========================================================================
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log system metrics."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session,
            **metrics
        }
        
        # Log to metrics file (daily)
        date_str = datetime.now().strftime("%Y%m%d")
        file_path = self.metrics_dir / f"metrics_{date_str}.jsonl"
        self._append_jsonl(file_path, event)
    
    def save_snapshot(self, name: str, data: Any):
        """
        Save a snapshot of system state.
        
        Args:
            name: Snapshot name
            data: Data to snapshot
        """
        snapshot = {
            "name": name,
            "session_id": self.current_session,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Save as JSON
        file_path = self.snapshots_dir / f"{name}_{self.current_session}.json"
        self._write_json(file_path, snapshot)
        
        logger.info(f"Saved snapshot: {name}")
    
    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    
    def get_session_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if not session_id:
            session_id = self.current_session
        
        file_path = self.sessions_dir / f"{session_id}.jsonl"
        return self._read_jsonl(file_path)
    
    def get_intent_traces(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get intent traces for a session."""
        if not session_id:
            session_id = self.current_session
        
        file_path = self.traces_dir / f"{session_id}.jsonl"
        return self._read_jsonl(file_path)
    
    def get_open_intents_from_history(self) -> List[Dict[str, Any]]:
        """Analyze history to find unclosed intents."""
        traces = self.get_intent_traces()
        
        opened = {}
        closed = set()
        
        for event in traces:
            if event["event"] == "intent_opened":
                opened[event["trace_id"]] = event
            elif event["event"] == "intent_closed":
                closed.add(event["trace_id"])
        
        # Return opened but not closed
        return [trace for trace_id, trace in opened.items() if trace_id not in closed]
    
    def calculate_closure_metrics(self) -> Dict[str, Any]:
        """Calculate intent closure metrics."""
        traces = self.get_intent_traces()
        
        total_opened = 0
        total_closed = 0
        closure_states = {}
        closure_triggers = {}
        durations = []
        
        for event in traces:
            if event["event"] == "intent_opened":
                total_opened += 1
            elif event["event"] == "intent_closed":
                total_closed += 1
                
                # Track closure states
                state = event.get("closure_state", "unknown")
                closure_states[state] = closure_states.get(state, 0) + 1
                
                # Track triggers
                trigger = event.get("closure_trigger", "unknown")
                closure_triggers[trigger] = closure_triggers.get(trigger, 0) + 1
                
                # Track duration
                if "duration_seconds" in event:
                    durations.append(event["duration_seconds"])
        
        # Calculate metrics
        closure_rate = total_closed / total_opened if total_opened > 0 else 0.0
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_opened": total_opened,
            "total_closed": total_closed,
            "closure_rate": closure_rate,
            "closure_states": closure_states,
            "closure_triggers": closure_triggers,
            "average_duration_seconds": avg_duration,
            "unclosed_count": total_opened - total_closed
        }
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def end_session(self):
        """End current session and save metadata."""
        # Calculate final metrics
        metrics = self.calculate_closure_metrics()
        
        # Update session metadata
        self.session_metadata.update({
            "ended_at": datetime.now().isoformat(),
            "final_metrics": metrics
        })
        
        # Log session end
        self._log_session_event({
            "event": "session_end",
            "session_id": self.current_session,
            "timestamp": self.session_metadata["ended_at"],
            "metrics": metrics
        })
        
        # Save session metadata
        meta_path = self.sessions_dir / f"{self.current_session}_metadata.json"
        self._write_json(meta_path, self.session_metadata)
        
        logger.info(f"Session ended: {self.current_session}")
    
    def start_new_session(self, session_id: Optional[str] = None):
        """Start a new session."""
        # End current session
        self.end_session()
        
        # Start new session
        if session_id:
            self.current_session = session_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = f"session_{timestamp}"
        
        # Reset metadata
        self.session_metadata = {
            "session_id": self.current_session,
            "started_at": datetime.now().isoformat(),
            "events": 0,
            "intents_opened": 0,
            "intents_closed": 0
        }
        
        # Log session start
        self._log_session_event({
            "event": "session_start",
            "session_id": self.current_session,
            "timestamp": self.session_metadata["started_at"]
        })
        
        logger.info(f"New session started: {self.current_session}")
    
    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================
    
    def _log_session_event(self, event: Dict[str, Any]):
        """Log event to session file."""
        file_path = self.sessions_dir / f"{self.current_session}.jsonl"
        self._append_jsonl(file_path, event)
        self.session_metadata["events"] += 1
    
    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Append to JSONL file."""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to append to {file_path}: {e}")
    
    def _write_json(self, file_path: Path, data: Any):
        """Write JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
    
    def _read_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSONL file."""
        if not file_path.exists():
            return []
        
        events = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
        
        return events


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ["MemoryVault"]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

logger.info("MemoryVault module loaded successfully")
