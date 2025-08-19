# python/core/memory_vault.py
"""
Enhanced MemoryVault for comprehensive intent lifecycle logging and analytics.
Provides full accountability and data for retraining.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryVault:
    """
    Persistent storage for intent traces, conversation history, and analytics.
    Supports both JSONL file storage and SQLite database for querying.
    """
    
    def __init__(self, 
                 base_dir: str = "memory_vault",
                 session_id: Optional[str] = None,
                 use_database: bool = True):
        """
        Initialize the MemoryVault.
        
        Args:
            base_dir: Base directory for storage
            session_id: Optional session identifier
            use_database: Whether to use SQLite for structured queries
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.traces_dir = self.base_dir / "traces"
        self.sessions_dir = self.base_dir / "sessions"
        self.analytics_dir = self.base_dir / "analytics"
        
        for dir_path in [self.traces_dir, self.sessions_dir, self.analytics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Session management
        self.current_session = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Database setup
        self.use_database = use_database
        self.db_path = self.base_dir / "memory_vault.db"
        if self.use_database:
            self._initialize_database()
        
        logger.info(f"MemoryVault initialized for session {self.current_session}")
    
    def _initialize_database(self):
        """Initialize SQLite database schema."""
        with self._get_db() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS intent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    intent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    intent_name TEXT,
                    closure_state TEXT,
                    confidence REAL,
                    data JSON,
                    INDEX idx_session (session_id),
                    INDEX idx_intent (intent_id),
                    INDEX idx_timestamp (timestamp)
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME,
                    end_time DATETIME,
                    total_intents INTEGER,
                    closed_intents INTEGER,
                    summary JSON
                );
                
                CREATE TABLE IF NOT EXISTS action_trajectory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT,
                    intent_candidates JSON,
                    context JSON,
                    INDEX idx_session_turn (session_id, turn)
                );
            ''')
    
    @contextmanager
    def _get_db(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def log_intent_open(self, trace):
        """
        Log intent opening event.
        
        Args:
            trace: IntentTrace object
        """
        self._log_event("intent_opened", trace)
        
        if self.use_database:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO intent_events 
                    (session_id, intent_id, event_type, intent_name, closure_state, confidence, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_session,
                    trace.intent_id,
                    "opened",
                    trace.name,
                    trace.closure_state,
                    trace.confidence,
                    json.dumps(trace.to_dict())
                ))
    
    def log_intent_close(self, trace):
        """
        Log intent closure event.
        
        Args:
            trace: IntentTrace object
        """
        self._log_event("intent_closed", trace)
        
        if self.use_database:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO intent_events 
                    (session_id, intent_id, event_type, intent_name, closure_state, confidence, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_session,
                    trace.intent_id,
                    "closed",
                    trace.name,
                    trace.closure_state,
                    trace.confidence,
                    json.dumps(trace.to_dict())
                ))
    
    def log_intent_update(self, trace, update_type: str = "update"):
        """
        Log intent update event.
        
        Args:
            trace: IntentTrace object
            update_type: Type of update (e.g., "confidence_change", "migration")
        """
        self._log_event(f"intent_{update_type}", trace)
        
        if self.use_database:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO intent_events 
                    (session_id, intent_id, event_type, intent_name, closure_state, confidence, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_session,
                    trace.intent_id,
                    update_type,
                    trace.name,
                    trace.closure_state,
                    trace.confidence,
                    json.dumps(trace.to_dict())
                ))
    
    def log_action(self, turn: int, user_input: str, 
                  intent_candidates: List[Dict[str, Any]],
                  context: Optional[Dict[str, Any]] = None):
        """
        Log user action in trajectory.
        
        Args:
            turn: Turn number
            user_input: User input text
            intent_candidates: List of intent candidates
            context: Optional context
        """
        if self.use_database:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO action_trajectory 
                    (session_id, turn, user_input, intent_candidates, context)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    self.current_session,
                    turn,
                    user_input,
                    json.dumps(intent_candidates),
                    json.dumps(context) if context else None
                ))
    
    def _log_event(self, event_type: str, trace):
        """
        Log event to JSONL file.
        
        Args:
            event_type: Type of event
            trace: IntentTrace object
        """
        log_path = self.traces_dir / f"{self.current_session}.jsonl"
        
        entry = trace.to_dict()
        entry["event"] = event_type
        entry["session_id"] = self.current_session
        entry["timestamp"] = datetime.now().isoformat()
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.debug(f"Logged {event_type} for intent {trace.intent_id}")
    
    def log_session_end(self, summary: Dict[str, Any]):
        """
        Log session end with summary.
        
        Args:
            summary: Session summary dictionary
        """
        session_path = self.sessions_dir / f"{self.current_session}.json"
        
        session_data = {
            "session_id": self.current_session,
            "start_time": self.session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "summary": summary
        }
        
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        if self.use_database:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (session_id, start_time, end_time, total_intents, closed_intents, summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self.current_session,
                    self.session_start,
                    datetime.now(),
                    summary.get("active_intents", 0) + summary.get("closed_intents", 0),
                    summary.get("closed_intents", 0),
                    json.dumps(summary)
                ))
        
        logger.info(f"Session {self.current_session} ended and logged")
    
    def get_all_traces(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all traces for a session.
        
        Args:
            session_id: Optional session ID (defaults to current)
            
        Returns:
            List of trace dictionaries
        """
        session_id = session_id or self.current_session
        log_path = self.traces_dir / f"{session_id}.jsonl"
        
        traces = []
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    traces.append(json.loads(line))
        
        return traces
    
    def get_closure_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get closure statistics for a session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Dictionary of statistics
        """
        if self.use_database:
            session_id = session_id or self.current_session
            with self._get_db() as conn:
                # Get closure state distribution
                cursor = conn.execute('''
                    SELECT closure_state, COUNT(*) as count
                    FROM intent_events
                    WHERE session_id = ? AND event_type = 'closed'
                    GROUP BY closure_state
                ''', (session_id,))
                
                closure_dist = {row['closure_state']: row['count'] 
                               for row in cursor.fetchall()}
                
                # Get average confidence at closure
                cursor = conn.execute('''
                    SELECT AVG(confidence) as avg_confidence
                    FROM intent_events
                    WHERE session_id = ? AND event_type = 'closed'
                ''', (session_id,))
                
                avg_confidence = cursor.fetchone()['avg_confidence'] or 0
                
                # Get intent durations (in events, not time)
                cursor = conn.execute('''
                    SELECT 
                        i1.intent_id,
                        i1.intent_name,
                        COUNT(DISTINCT i2.id) as event_count
                    FROM intent_events i1
                    JOIN intent_events i2 ON i1.intent_id = i2.intent_id
                    WHERE i1.session_id = ? AND i1.event_type = 'opened'
                    GROUP BY i1.intent_id, i1.intent_name
                ''', (session_id,))
                
                durations = [
                    {"intent_id": row['intent_id'], 
                     "name": row['intent_name'],
                     "events": row['event_count']}
                    for row in cursor.fetchall()
                ]
                
                return {
                    "session_id": session_id,
                    "closure_distribution": closure_dist,
                    "average_confidence": round(avg_confidence, 3),
                    "intent_durations": durations
                }
        else:
            # Fallback to file-based analysis
            traces = self.get_all_traces(session_id)
            closure_dist = {}
            confidences = []
            
            for trace in traces:
                if trace.get("event") == "intent_closed":
                    state = trace.get("closure_state", "unknown")
                    closure_dist[state] = closure_dist.get(state, 0) + 1
                    confidences.append(trace.get("confidence", 0))
            
            return {
                "session_id": session_id or self.current_session,
                "closure_distribution": closure_dist,
                "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0
            }
    
    def export_session_data(self, session_id: Optional[str] = None,
                           output_path: Optional[str] = None) -> str:
        """
        Export all session data to a consolidated JSON file.
        
        Args:
            session_id: Session to export
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        session_id = session_id or self.current_session
        
        if not output_path:
            output_path = self.analytics_dir / f"export_{session_id}.json"
        else:
            output_path = Path(output_path)
        
        export_data = {
            "session_id": session_id,
            "export_time": datetime.now().isoformat(),
            "traces": self.get_all_traces(session_id),
            "statistics": self.get_closure_statistics(session_id)
        }
        
        # Add session summary if available
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                export_data["session_summary"] = json.load(f)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported session {session_id} to {output_path}")
        return str(output_path)
    
    def query_intents(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query on the intent database.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as list of dictionaries
        """
        if not self.use_database:
            logger.warning("Database not enabled, cannot execute query")
            return []
        
        with self._get_db() as conn:
            cursor = conn.execute(query)
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_intent_trajectory(self, intent_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete trajectory of a specific intent.
        
        Args:
            intent_id: Intent identifier
            
        Returns:
            List of all events for the intent
        """
        if self.use_database:
            with self._get_db() as conn:
                cursor = conn.execute('''
                    SELECT * FROM intent_events
                    WHERE intent_id = ?
                    ORDER BY timestamp
                ''', (intent_id,))
                
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:
            # Fallback to file search
            all_traces = self.get_all_traces()
            return [t for t in all_traces if t.get("intent_id") == intent_id]
