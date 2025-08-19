#!/usr/bin/env python3
"""
Conversation Manager
====================
Orchestrates user inference, context injection, and vault updates.
Ensures proper isolation and logging of all conversations.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import threading
from collections import deque

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.saigon_inference_v5 import SaigonInference
from python.core.user_context import (
    UserContextManager,
    create_user_session,
    validate_user_session,
    record_user_inference
)
from python.core.concept_mesh_v5 import MeshManager
from python.core.adapter_loader_v5 import get_adapter_path_for_user

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONVERSATION_LENGTH = 100  # Max turns per conversation
CONVERSATION_LOG_DIR = Path("logs/conversations")
INTENT_VAULT_DIR = Path("data/intent_vault")
MAX_CONTEXT_WINDOW = 4096  # Token limit for context

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Message:
    """Single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class Conversation:
    """Complete conversation with history and context."""
    conversation_id: str
    user_id: str
    session_id: Optional[str]
    messages: List[Message] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    adapter_path: Optional[str] = None
    mesh_context: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Trim if too long
        if len(self.messages) > MAX_CONVERSATION_LENGTH:
            self.messages = self.messages[-MAX_CONVERSATION_LENGTH:]
    
    def get_context_window(self, max_turns: int = 5) -> str:
        """Get recent conversation context."""
        recent_messages = self.messages[-max_turns*2:] if self.messages else []
        
        context_parts = []
        for msg in recent_messages:
            prefix = "User:" if msg.role == "user" else "Assistant:"
            context_parts.append(f"{prefix} {msg.content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "adapter_path": self.adapter_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active": self.active
        }

@dataclass
class IntentGap:
    """Detected intent gap in conversation."""
    user_id: str
    conversation_id: str
    gap_type: str  # "knowledge", "capability", "context"
    description: str
    user_input: str
    assistant_response: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """
    Manages conversations with full context and isolation.
    """
    
    def __init__(self,
                 inference_engine: Optional[SaigonInference] = None,
                 mesh_manager: Optional[MeshManager] = None):
        """
        Initialize conversation manager.
        
        Args:
            inference_engine: Optional pre-configured inference engine
            mesh_manager: Optional pre-configured mesh manager
        """
        self.inference_engine = inference_engine or SaigonInference()
        self.mesh_manager = mesh_manager or MeshManager()
        self.context_manager = UserContextManager()
        
        # Conversation storage
        self.conversations: Dict[str, Conversation] = {}
        self.user_conversations: Dict[str, List[str]] = {}  # user_id -> conversation_ids
        
        # Thread safety
        self._lock = threading.Lock()
        self._user_locks = {}
        
        # Intent gaps queue
        self.intent_gaps = deque(maxlen=1000)
        
        # Ensure directories
        CONVERSATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        INTENT_VAULT_DIR.mkdir(parents=True, exist_ok=True)
        
        self._load_conversations()
    
    def _get_user_lock(self, user_id: str) -> threading.Lock:
        """Get or create lock for user."""
        if user_id not in self._user_locks:
            self._user_locks[user_id] = threading.Lock()
        return self._user_locks[user_id]
    
    def _load_conversations(self):
        """Load active conversations from disk."""
        conv_file = CONVERSATION_LOG_DIR / "active_conversations.json"
        if conv_file.exists():
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct conversations (simplified)
                    logger.info(f"Loaded {len(data)} active conversations")
            except Exception as e:
                logger.error(f"Failed to load conversations: {e}")
    
    def _save_conversations(self):
        """Save active conversations to disk."""
        conv_file = CONVERSATION_LOG_DIR / "active_conversations.json"
        try:
            # Save only active conversations
            active_convs = {
                cid: conv.to_dict()
                for cid, conv in self.conversations.items()
                if conv.active
            }
            
            with open(conv_file, 'w') as f:
                json.dump(active_convs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def start_conversation(self,
                          user_id: str,
                          session_id: Optional[str] = None,
                          domain: Optional[str] = None) -> str:
        """
        Start new conversation for user.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID
            domain: Optional domain context
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        # Create session if not provided
        if not session_id:
            session_id = create_user_session(user_id, domain=domain)
        
        # Validate session belongs to user
        if not validate_user_session(session_id, user_id):
            raise ValueError(f"Invalid session {session_id} for user {user_id}")
        
        # Get adapter and mesh
        adapter_path = get_adapter_path_for_user(user_id)
        mesh_context = self.mesh_manager.load_mesh(user_id)
        
        # Create conversation
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
            adapter_path=adapter_path,
            mesh_context=mesh_context,
            context={"domain": domain} if domain else {}
        )
        
        with self._get_user_lock(user_id):
            self.conversations[conversation_id] = conversation
            
            if user_id not in self.user_conversations:
                self.user_conversations[user_id] = []
            self.user_conversations[user_id].append(conversation_id)
        
        self._log_conversation_event("started", conversation_id, user_id)
        
        logger.info(f"Started conversation {conversation_id} for user {user_id}")
        return conversation_id
    
    def process_message(self,
                       conversation_id: str,
                       user_input: str,
                       use_history: bool = True,
                       max_tokens: int = 256,
                       temperature: float = 0.7) -> Dict[str, Any]:
        """
        Process user message and generate response.
        
        Args:
            conversation_id: Conversation ID
            user_input: User's input message
            use_history: Whether to include conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response dictionary with output and metadata
        """
        # Get conversation
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # CRITICAL: Verify user isolation
        user_id = conversation.user_id
        
        with self._get_user_lock(user_id):
            # Add user message
            conversation.add_message("user", user_input)
            
            # Build prompt with history
            if use_history:
                context = conversation.get_context_window()
                full_prompt = f"{context}\n\nUser: {user_input}\nAssistant:"
            else:
                full_prompt = user_input
            
            # Run inference
            result = self.inference_engine.run_inference(
                user_id=user_id,
                user_input=full_prompt,
                use_mesh_context=conversation.mesh_context is not None,
                max_new_tokens=max_tokens,
                temperature=temperature,
                log_inference=True
            )
            
            # Extract response
            assistant_response = result.get("output", "")
            
            # Add assistant message
            conversation.add_message("assistant", assistant_response, {
                "latency_ms": result.get("latency_ms"),
                "adapter_used": result.get("adapter_used")
            })
            
            # Detect intent gaps
            self._detect_intent_gaps(
                conversation_id,
                user_id,
                user_input,
                assistant_response
            )
            
            # Record inference
            record_user_inference(user_id, conversation.session_id, result)
            
            # Log to vault
            self._log_to_vault(conversation_id, user_id, user_input, assistant_response)
            
            # Save state
            self._save_conversations()
            
            return {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "input": user_input,
                "output": assistant_response,
                "latency_ms": result.get("latency_ms"),
                "adapter_used": result.get("adapter_used"),
                "message_count": len(conversation.messages)
            }
    
    def _detect_intent_gaps(self,
                           conversation_id: str,
                           user_id: str,
                           user_input: str,
                           assistant_response: str):
        """Detect potential intent gaps in conversation."""
        # Simple heuristics for gap detection
        gap_indicators = [
            ("I don't know", "knowledge"),
            ("I'm not sure", "knowledge"),
            ("I cannot", "capability"),
            ("I'm unable", "capability"),
            ("Could you clarify", "context"),
            ("I don't understand", "context")
        ]
        
        for indicator, gap_type in gap_indicators:
            if indicator.lower() in assistant_response.lower():
                gap = IntentGap(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    gap_type=gap_type,
                    description=f"Potential {gap_type} gap detected",
                    user_input=user_input,
                    assistant_response=assistant_response,
                    confidence=0.7
                )
                
                self.intent_gaps.append(gap)
                self._log_intent_gap(gap)
                
                logger.info(f"Detected {gap_type} gap in conversation {conversation_id}")
                break
    
    def _log_intent_gap(self, gap: IntentGap):
        """Log intent gap to vault."""
        gap_file = INTENT_VAULT_DIR / f"gaps_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(gap_file, 'a') as f:
                f.write(json.dumps(gap.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log intent gap: {e}")
    
    def _log_to_vault(self,
                     conversation_id: str,
                     user_id: str,
                     user_input: str,
                     assistant_response: str):
        """Log conversation turn to vault."""
        vault_file = INTENT_VAULT_DIR / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        
        try:
            with open(vault_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log to vault: {e}")
    
    def _log_conversation_event(self, event: str, conversation_id: str, user_id: str):
        """Log conversation event."""
        log_file = CONVERSATION_LOG_DIR / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "conversation_id": conversation_id,
            "user_id": user_id
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return False
        
        with self._get_user_lock(conversation.user_id):
            conversation.active = False
            self._log_conversation_event("ended", conversation_id, conversation.user_id)
            self._save_conversations()
        
        logger.info(f"Ended conversation {conversation_id}")
        return True
    
    def get_conversation_history(self,
                                conversation_id: str,
                                max_messages: Optional[int] = None) -> List[Dict]:
        """Get conversation history."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [m.to_dict() for m in messages]
    
    def get_user_conversations(self, user_id: str, active_only: bool = True) -> List[str]:
        """Get all conversation IDs for user."""
        conv_ids = self.user_conversations.get(user_id, [])
        
        if active_only:
            conv_ids = [
                cid for cid in conv_ids
                if cid in self.conversations and self.conversations[cid].active
            ]
        
        return conv_ids
    
    def get_intent_gaps(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get recent intent gaps."""
        gaps = list(self.intent_gaps)
        
        if user_id:
            gaps = [g for g in gaps if g.user_id == user_id]
        
        return [g.to_dict() for g in gaps]
    
    def clear_inactive_conversations(self, hours: int = 24) -> int:
        """Clear inactive conversations older than specified hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        cleared = 0
        
        with self._lock:
            to_remove = []
            for cid, conv in self.conversations.items():
                if conv.updated_at.timestamp() < cutoff and not conv.active:
                    to_remove.append(cid)
            
            for cid in to_remove:
                del self.conversations[cid]
                cleared += 1
        
        if cleared > 0:
            logger.info(f"Cleared {cleared} inactive conversations")
            self._save_conversations()
        
        return cleared

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_conversation_manager = None

def get_conversation_manager() -> ConversationManager:
    """Get global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager

def start_conversation(user_id: str, **kwargs) -> str:
    """Start new conversation."""
    manager = get_conversation_manager()
    return manager.start_conversation(user_id, **kwargs)

def chat(conversation_id: str, user_input: str, **kwargs) -> Dict[str, Any]:
    """Process chat message."""
    manager = get_conversation_manager()
    return manager.process_message(conversation_id, user_input, **kwargs)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for conversation management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conversation Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start conversation")
    start_parser.add_argument("--user_id", required=True, help="User ID")
    start_parser.add_argument("--domain", help="Domain context")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--user_id", required=True, help="User ID")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show history")
    history_parser.add_argument("--conversation_id", required=True, help="Conversation ID")
    history_parser.add_argument("--limit", type=int, help="Message limit")
    
    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Show intent gaps")
    gaps_parser.add_argument("--user_id", help="Filter by user")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = get_conversation_manager()
    
    if args.command == "start":
        conv_id = manager.start_conversation(args.user_id, domain=args.domain)
        print(f"Started conversation: {conv_id}")
    
    elif args.command == "chat":
        # Start conversation
        conv_id = manager.start_conversation(args.user_id)
        print(f"Started conversation: {conv_id}")
        print("Type 'quit' to exit\n")
        
        # Interactive chat loop
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                manager.end_conversation(conv_id)
                print("Conversation ended.")
                break
            
            if not user_input:
                continue
            
            result = manager.process_message(conv_id, user_input)
            print(f"\nAssistant: {result['output']}\n")
            print(f"[Latency: {result['latency_ms']:.2f}ms]\n")
    
    elif args.command == "history":
        history = manager.get_conversation_history(
            args.conversation_id,
            args.limit
        )
        
        print(f"\nConversation History ({len(history)} messages):")
        for msg in history:
            role = msg['role'].capitalize()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  [{role}]: {content}")
    
    elif args.command == "gaps":
        gaps = manager.get_intent_gaps(args.user_id)
        
        print(f"\nIntent Gaps ({len(gaps)} found):")
        for gap in gaps[:10]:  # Show last 10
            print(f"  [{gap['gap_type']}] {gap['description']}")
            print(f"    User: {gap['user_input'][:50]}...")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
