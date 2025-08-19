#!/usr/bin/env python3
"""
User Context Manager - Multi-User Session/Context Isolation
============================================================
Enforces per-user state isolation, prevents cross-user contamination,
manages session contexts, and provides helpers for API/session management.
"""

import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any, Set, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_SESSIONS_PER_USER = 5
SESSION_TIMEOUT_MINUTES = 30
CONTEXT_CACHE_SIZE = 100
AUDIT_LOG_DIR = Path("logs/user_context")

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UserSession:
    """Represents an active user session."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    adapter_path: Optional[str] = None
    mesh_context: Optional[Dict] = None
    domain: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    active: bool = True
    
    def is_expired(self, timeout_minutes: int = SESSION_TIMEOUT_MINUTES) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        return data

@dataclass
class UserContext:
    """Complete context for a user across all sessions."""
    user_id: str
    sessions: List[UserSession] = field(default_factory=list)
    active_adapter: Optional[str] = None
    active_domain: Optional[str] = None
    preferences: Dict = field(default_factory=dict)
    limits: Dict = field(default_factory=dict)
    last_inference: Optional[datetime] = None
    total_inferences: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_active_session(self) -> Optional[UserSession]:
        """Get most recent active session."""
        active_sessions = [s for s in self.sessions if s.active and not s.is_expired()]
        return active_sessions[-1] if active_sessions else None
    
    def add_session(self, session: UserSession) -> bool:
        """Add new session, enforcing limits."""
        # Remove expired sessions
        self.sessions = [s for s in self.sessions if not s.is_expired()]
        
        # Check session limit
        if len(self.sessions) >= MAX_SESSIONS_PER_USER:
            # Remove oldest session
            self.sessions.pop(0)
        
        self.sessions.append(session)
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "sessions": [s.to_dict() for s in self.sessions],
            "active_adapter": self.active_adapter,
            "active_domain": self.active_domain,
            "preferences": self.preferences,
            "limits": self.limits,
            "last_inference": self.last_inference.isoformat() if self.last_inference else None,
            "total_inferences": self.total_inferences,
            "created_at": self.created_at.isoformat()
        }

# ============================================================================
# USER CONTEXT MANAGER
# ============================================================================

class UserContextManager:
    """
    Manages multi-user contexts with complete isolation.
    Thread-safe singleton pattern for global access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._contexts: Dict[str, UserContext] = {}
            self._sessions: Dict[str, UserSession] = {}
            self._user_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
            self._global_lock = threading.Lock()
            self._audit_dir = AUDIT_LOG_DIR
            self._audit_dir.mkdir(parents=True, exist_ok=True)
            self._load_contexts()
            logger.info("UserContextManager initialized")
    
    def _load_contexts(self):
        """Load persisted contexts from disk."""
        context_file = self._audit_dir / "contexts.json"
        if context_file.exists():
            try:
                with open(context_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct contexts
                    for user_id, context_data in data.items():
                        context = UserContext(user_id=user_id)
                        # Restore fields
                        context.active_adapter = context_data.get('active_adapter')
                        context.active_domain = context_data.get('active_domain')
                        context.preferences = context_data.get('preferences', {})
                        context.limits = context_data.get('limits', {})
                        context.total_inferences = context_data.get('total_inferences', 0)
                        self._contexts[user_id] = context
                logger.info(f"Loaded {len(self._contexts)} user contexts")
            except Exception as e:
                logger.error(f"Failed to load contexts: {e}")
    
    def _save_contexts(self):
        """Persist contexts to disk."""
        context_file = self._audit_dir / "contexts.json"
        try:
            data = {
                user_id: context.to_dict()
                for user_id, context in self._contexts.items()
            }
            with open(context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save contexts: {e}")
    
    def get_or_create_context(self, user_id: str) -> UserContext:
        """Get or create user context with thread safety."""
        with self._user_locks[user_id]:
            if user_id not in self._contexts:
                self._contexts[user_id] = UserContext(user_id=user_id)
                self._audit_log("context_created", user_id)
            return self._contexts[user_id]
    
    def create_session(self, 
                      user_id: str,
                      adapter_path: Optional[str] = None,
                      domain: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> str:
        """
        Create new session for user.
        
        Args:
            user_id: User identifier
            adapter_path: Optional adapter path
            domain: Optional domain
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            adapter_path=adapter_path,
            domain=domain,
            metadata=metadata or {}
        )
        
        # Add to user context
        context = self.get_or_create_context(user_id)
        with self._user_locks[user_id]:
            context.add_session(session)
            self._sessions[session_id] = session
        
        self._audit_log("session_created", user_id, {"session_id": session_id})
        self._save_contexts()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            session.update_activity()
            return session
        return None
    
    def validate_session(self, session_id: str, user_id: str) -> bool:
        """
        Validate session belongs to user and is active.
        
        CRITICAL: Prevents cross-user access.
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        # CRITICAL: Check user ownership
        if session.user_id != user_id:
            self._audit_log("security_violation", user_id, {
                "attempted_session": session_id,
                "actual_owner": session.user_id
            })
            return False
        
        return session.active and not session.is_expired()
    
    def update_session_context(self,
                              session_id: str,
                              adapter_path: Optional[str] = None,
                              mesh_context: Optional[Dict] = None,
                              domain: Optional[str] = None) -> bool:
        """Update session context."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self._user_locks[session.user_id]:
            if adapter_path:
                session.adapter_path = adapter_path
            if mesh_context:
                session.mesh_context = mesh_context
            if domain:
                session.domain = domain
            session.update_activity()
        
        self._save_contexts()
        return True
    
    def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        with self._user_locks[session.user_id]:
            session.active = False
            if session_id in self._sessions:
                del self._sessions[session_id]
        
        self._audit_log("session_ended", session.user_id, {"session_id": session_id})
        self._save_contexts()
        return True
    
    def get_user_adapter(self, user_id: str) -> Optional[str]:
        """Get user's active adapter path."""
        context = self.get_or_create_context(user_id)
        
        # Check active session first
        active_session = context.get_active_session()
        if active_session and active_session.adapter_path:
            return active_session.adapter_path
        
        # Fall back to context default
        return context.active_adapter
    
    def get_user_domain(self, user_id: str) -> Optional[str]:
        """Get user's active domain."""
        context = self.get_or_create_context(user_id)
        
        # Check active session first
        active_session = context.get_active_session()
        if active_session and active_session.domain:
            return active_session.domain
        
        # Fall back to context default
        return context.active_domain
    
    def record_inference(self, 
                        user_id: str,
                        session_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> bool:
        """Record an inference event."""
        context = self.get_or_create_context(user_id)
        
        with self._user_locks[user_id]:
            context.last_inference = datetime.now()
            context.total_inferences += 1
            
            # Update session if provided
            if session_id:
                session = self.get_session(session_id)
                if session:
                    session.update_activity()
        
        self._audit_log("inference_recorded", user_id, metadata)
        self._save_contexts()
        return True
    
    def check_rate_limit(self, user_id: str, limit_per_minute: int = 10) -> bool:
        """Check if user is within rate limits."""
        context = self.get_or_create_context(user_id)
        
        # Simple rate limit check (enhance as needed)
        if context.last_inference:
            time_since = datetime.now() - context.last_inference
            if time_since < timedelta(seconds=60/limit_per_minute):
                self._audit_log("rate_limit_exceeded", user_id)
                return False
        
        return True
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions across all users."""
        expired_count = 0
        
        with self._global_lock:
            # Find expired sessions
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if session.is_expired()
            ]
            
            # Remove expired sessions
            for session_id in expired_ids:
                session = self._sessions[session_id]
                with self._user_locks[session.user_id]:
                    session.active = False
                    del self._sessions[session_id]
                    expired_count += 1
            
            # Clean up contexts
            for context in self._contexts.values():
                context.sessions = [s for s in context.sessions if not s.is_expired()]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
            self._save_contexts()
        
        return expired_count
    
    def get_all_active_users(self) -> List[str]:
        """Get list of users with active sessions."""
        active_users = []
        for user_id, context in self._contexts.items():
            if context.get_active_session():
                active_users.append(user_id)
        return active_users
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user."""
        context = self.get_or_create_context(user_id)
        active_session = context.get_active_session()
        
        return {
            "user_id": user_id,
            "total_sessions": len(context.sessions),
            "active_sessions": sum(1 for s in context.sessions if s.active),
            "total_inferences": context.total_inferences,
            "last_inference": context.last_inference.isoformat() if context.last_inference else None,
            "active_adapter": context.active_adapter,
            "active_domain": context.active_domain,
            "current_session": active_session.session_id if active_session else None
        }
    
    def _audit_log(self, event: str, user_id: str, metadata: Optional[Dict] = None):
        """Log audit event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "user_id": user_id,
            "metadata": metadata or {}
        }
        
        # Write to daily log file
        log_file = self._audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def enforce_isolation(self, user_id: str, resource_id: str, resource_type: str) -> bool:
        """
        Enforce user isolation for any resource access.
        
        CRITICAL: Central isolation enforcement point.
        
        Args:
            user_id: User requesting access
            resource_id: Resource identifier (adapter path, mesh path, etc.)
            resource_type: Type of resource (adapter, mesh, data)
            
        Returns:
            True if access allowed, False otherwise
        """
        # Check if resource belongs to user
        if resource_type in ["adapter", "mesh", "data"]:
            # Resource should contain user_id in path or name
            if f"user_{user_id}" in resource_id or resource_id.startswith("global"):
                return True
            
            # Log potential violation
            self._audit_log("isolation_violation_attempt", user_id, {
                "resource_id": resource_id,
                "resource_type": resource_type
            })
            return False
        
        return True

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_context_manager = UserContextManager()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_user_context(user_id: str) -> UserContext:
    """Get or create user context."""
    return _context_manager.get_or_create_context(user_id)

def create_user_session(user_id: str, **kwargs) -> str:
    """Create new user session."""
    return _context_manager.create_session(user_id, **kwargs)

def validate_user_session(session_id: str, user_id: str) -> bool:
    """Validate user owns session."""
    return _context_manager.validate_session(session_id, user_id)

def enforce_user_isolation(user_id: str, resource_id: str, resource_type: str) -> bool:
    """Enforce isolation for resource access."""
    return _context_manager.enforce_isolation(user_id, resource_id, resource_type)

def record_user_inference(user_id: str, session_id: Optional[str] = None, **kwargs) -> bool:
    """Record inference event."""
    return _context_manager.record_inference(user_id, session_id, kwargs)

def cleanup_sessions():
    """Clean up expired sessions."""
    return _context_manager.cleanup_expired_sessions()

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for user context management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="User Context Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get user statistics")
    stats_parser.add_argument("--user_id", required=True, help="User ID")
    
    # Active users command
    active_parser = subparsers.add_parser("active", help="List active users")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up expired sessions")
    
    # Create session command
    session_parser = subparsers.add_parser("session", help="Create session")
    session_parser.add_argument("--user_id", required=True, help="User ID")
    session_parser.add_argument("--domain", help="Domain")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == "stats":
        stats = _context_manager.get_user_statistics(args.user_id)
        print(f"\nUser Statistics for '{args.user_id}':")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == "active":
        users = _context_manager.get_all_active_users()
        print(f"\nActive Users ({len(users)}):")
        for user in users:
            print(f"  - {user}")
    
    elif args.command == "cleanup":
        count = _context_manager.cleanup_expired_sessions()
        print(f"Cleaned up {count} expired sessions")
    
    elif args.command == "session":
        session_id = _context_manager.create_session(
            args.user_id,
            domain=args.domain
        )
        print(f"Created session: {session_id}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
