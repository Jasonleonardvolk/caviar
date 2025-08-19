# python/core/earl_intent_reasoner.py
"""
EARL (Early Adaptive Reasoning and Learning) Intent Reasoner
Handles auto-morph, supersession, migration, satisfaction, and intelligent closure detection.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from .intent_trace import IntentTrace, CLOSURE_STATES
from .memory_vault import MemoryVault

logger = logging.getLogger(__name__)


class EARLIntentReasoner:
    """
    Orchestrates intent lifecycle management with incremental hypothesis updating,
    semantic drift detection, and intelligent closure pathways.
    """
    
    def __init__(self, 
                 memory_vault: Optional[MemoryVault] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EARL Intent Reasoner.
        
        Args:
            memory_vault: Optional MemoryVault for persistence
            config: Optional configuration dictionary
        """
        self.memory_vault = memory_vault
        self.config = config or {}
        
        # Active intent traces
        self.active_traces: List[IntentTrace] = []
        self.closed_traces: List[IntentTrace] = []
        
        # Conversation state
        self.turn_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configuration parameters
        self.decay_rate = self.config.get("decay_rate", 0.9)
        self.min_confidence = self.config.get("min_confidence", 0.4)
        self.semantic_threshold = self.config.get("semantic_threshold", 0.4)
        self.supersession_threshold = self.config.get("supersession_threshold", 0.75)
        self.migration_threshold = self.config.get("migration_threshold", 0.85)
        self.abandonment_turns = self.config.get("abandonment_turns", 10)
        
        # Intent hypothesis tracking
        self.intent_hypotheses: Dict[str, float] = defaultdict(float)
        self.action_trajectory: List[Dict[str, Any]] = []
        
        logger.info(f"EARLIntentReasoner initialized for session {self.session_id}")
    
    def process_input(self, 
                     user_text: str, 
                     intent_candidates: List[Dict[str, Any]],
                     context: Optional[Dict[str, Any]] = None) -> List[IntentTrace]:
        """
        Process user input and update intent traces.
        
        Args:
            user_text: Current user input
            intent_candidates: List of potential intents from parser
            context: Optional context dictionary
            
        Returns:
            List of currently active intent traces
        """
        self.turn_count += 1
        current_turn = self.turn_count
        user_text = user_text.strip()
        
        # Record action in trajectory
        self.action_trajectory.append({
            "turn": current_turn,
            "text": user_text,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        # Phase 1: Decay and closure detection for existing intents
        self._process_decay_and_closure(user_text, current_turn)
        
        # Phase 2: Supersession and migration detection
        self._detect_supersession_and_migration(user_text, intent_candidates, current_turn)
        
        # Phase 3: Process new or continuing intents
        self._process_intent_candidates(user_text, intent_candidates, current_turn, context)
        
        # Phase 4: Check for implicit closures
        self._check_implicit_closures(user_text)
        
        # Phase 5: Abandonment check
        self._check_abandonment(current_turn)
        
        # Update hypothesis scores
        self._update_hypotheses(intent_candidates)
        
        return self.get_open_traces()
    
    def _process_decay_and_closure(self, user_text: str, current_turn: int):
        """Apply decay and detect diminished intents."""
        for trace in self.active_traces:
            if trace.is_active():
                sim = trace.update_confidence_and_decay(
                    user_text, 
                    current_turn,
                    self.decay_rate,
                    self.min_confidence
                )
                
                if not trace.is_active():
                    logger.info(f"Intent {trace.intent_id} ({trace.name}) closed due to decay: {trace.closure_state}")
                    self._log_closure(trace)
                    self.closed_traces.append(trace)
        
        # Remove closed traces from active list
        self.active_traces = [t for t in self.active_traces if t.is_active()]
    
    def _detect_supersession_and_migration(self, 
                                          user_text: str,
                                          intent_candidates: List[Dict[str, Any]],
                                          current_turn: int):
        """Detect when intents supersede or migrate to new ones."""
        for candidate in intent_candidates:
            candidate_text = candidate.get("text", user_text)
            candidate_name = candidate["name"]
            candidate_id = candidate.get("intent_id", candidate_name)
            
            for trace in self.active_traces:
                if not trace.is_active():
                    continue
                
                # Compute similarity between existing intent and candidate
                sim = trace.compute_similarity(candidate_text)
                
                # Migration: Very high similarity but different name
                if sim > self.migration_threshold and candidate_name != trace.name:
                    trace.migrate(
                        candidate_id,
                        reason=f"Migrated to more specific intent: {candidate_name}"
                    )
                    logger.info(f"Intent {trace.intent_id} migrated to {candidate_id}")
                    self._log_closure(trace)
                    self.closed_traces.append(trace)
                
                # Supersession: High similarity and broader scope
                elif sim > self.supersession_threshold and candidate_name != trace.name:
                    if self._is_broader_intent(candidate, trace):
                        trace.supersede(
                            candidate_id,
                            reason=f"Superseded by broader intent: {candidate_name}"
                        )
                        logger.info(f"Intent {trace.intent_id} superseded by {candidate_id}")
                        self._log_closure(trace)
                        self.closed_traces.append(trace)
        
        # Remove closed traces
        self.active_traces = [t for t in self.active_traces if t.is_active()]
    
    def _process_intent_candidates(self, 
                                  user_text: str,
                                  intent_candidates: List[Dict[str, Any]],
                                  current_turn: int,
                                  context: Optional[Dict[str, Any]]):
        """Process new or continuing intent candidates."""
        for candidate in intent_candidates:
            intent_name = candidate["name"]
            description = candidate.get("text", user_text)
            confidence = candidate.get("confidence", 0.8)
            
            # Check if intent already exists and is active
            existing = self._find_active_intent(intent_name)
            
            if existing:
                # Reinforce existing intent
                existing.last_active_turn = current_turn
                existing.confidence = min(1.0, existing.confidence + 0.1)
                logger.info(f"Continuing intent {intent_name} ({existing.intent_id})")
            else:
                # Create new intent trace
                trace = IntentTrace(
                    name=intent_name,
                    description=description,
                    turn_opened=current_turn,
                    last_active_turn=current_turn,
                    confidence=confidence,
                    context_metadata=context or {}
                )
                self.active_traces.append(trace)
                self._log_opening(trace)
                logger.info(f"Opened new intent {intent_name} ({trace.intent_id})")
    
    def _check_implicit_closures(self, user_text: str):
        """Check for implicit closure triggers in user text."""
        # Acknowledgment patterns
        acknowledgment_triggers = [
            "thank", "thanks", "thank you", "perfect", "great",
            "solved", "fixed", "done", "that helps", "that's it",
            "got it", "understood", "makes sense", "all set"
        ]
        
        lower_text = user_text.lower()
        
        # Check for acknowledgment
        if any(trigger in lower_text for trigger in acknowledgment_triggers):
            self.acknowledge_closure(user_text)
        
        # Check for satisfaction patterns
        satisfaction_patterns = [
            ("that's what i needed", 0.9),
            ("exactly what i was looking for", 0.95),
            ("answers my question", 0.85),
            ("that solves it", 0.9)
        ]
        
        for pattern, confidence_boost in satisfaction_patterns:
            if pattern in lower_text:
                # Close most recent high-confidence intent
                recent_intents = sorted(
                    [t for t in self.active_traces if t.is_active()],
                    key=lambda t: t.last_active_turn,
                    reverse=True
                )
                if recent_intents and recent_intents[0].confidence > 0.7:
                    recent_intents[0].mark_closed(
                        state="confirmed",
                        reason=f"User satisfaction pattern: '{pattern}'"
                    )
                    self._log_closure(recent_intents[0])
                    logger.info(f"Intent {recent_intents[0].intent_id} closed via satisfaction pattern")
    
    def _check_abandonment(self, current_turn: int):
        """Check for abandoned intents based on inactivity."""
        for trace in self.active_traces:
            if not trace.is_active():
                continue
            
            inactivity = trace.get_inactivity_in_turns(current_turn)
            if inactivity > self.abandonment_turns:
                trace.abandon(
                    reason=f"No activity for {inactivity} turns"
                )
                self._log_closure(trace)
                self.closed_traces.append(trace)
                logger.info(f"Intent {trace.intent_id} abandoned after {inactivity} turns")
        
        # Remove abandoned traces
        self.active_traces = [t for t in self.active_traces if t.is_active()]
    
    def acknowledge_closure(self, text: str, intent_name: Optional[str] = None):
        """
        Acknowledge user closure signal.
        
        Args:
            text: User text containing closure signal
            intent_name: Optional specific intent to close
        """
        if intent_name:
            # Close specific intent
            trace = self._find_active_intent(intent_name)
            if trace:
                trace.mark_closed(state="confirmed", reason="User acknowledgment")
                self._log_closure(trace)
                self.closed_traces.append(trace)
                logger.info(f"User acknowledged closure for {trace.intent_id} ({trace.name})")
        else:
            # Close most recent active intent
            unresolved = [t for t in self.active_traces if t.is_active()]
            if unresolved:
                to_close = max(unresolved, key=lambda t: t.last_active_turn)
                to_close.mark_closed(state="confirmed", reason="User acknowledgment")
                self._log_closure(to_close)
                self.closed_traces.append(to_close)
                logger.info(f"User acknowledged closure for {to_close.intent_id} ({to_close.name})")
        
        # Clean up
        self.active_traces = [t for t in self.active_traces if t.is_active()]
    
    def satisfy_intent_elsewhere(self, intent_name: str, action_id: str, reason: Optional[str] = None):
        """
        Mark an intent as satisfied by an external action.
        
        Args:
            intent_name: Name of the intent to satisfy
            action_id: ID of the satisfying action
            reason: Optional reason for satisfaction
        """
        trace = self._find_active_intent(intent_name)
        if trace:
            trace.satisfy_elsewhere(action_id, reason)
            self._log_closure(trace)
            self.closed_traces.append(trace)
            self.active_traces = [t for t in self.active_traces if t.is_active()]
            logger.info(f"Intent {trace.intent_id} satisfied by action {action_id}")
    
    def _find_active_intent(self, intent_name: str) -> Optional[IntentTrace]:
        """Find an active intent by name."""
        for trace in self.active_traces:
            if trace.name == intent_name and trace.is_active():
                return trace
        return None
    
    def _is_broader_intent(self, candidate: Dict[str, Any], trace: IntentTrace) -> bool:
        """
        Determine if a candidate intent is broader than an existing trace.
        
        Args:
            candidate: Candidate intent dictionary
            trace: Existing intent trace
            
        Returns:
            True if candidate is broader in scope
        """
        # Simple heuristic: check if candidate description contains trace description
        candidate_text = candidate.get("text", "").lower()
        trace_text = trace.description.lower()
        
        # Check for containment or generalization patterns
        if trace_text in candidate_text:
            return True
        
        # Check for known generalization patterns
        generalizations = {
            "setup": ["configure", "initialize", "prepare"],
            "deploy": ["release", "publish", "launch"],
            "build": ["create", "make", "develop"],
            "analyze": ["examine", "investigate", "study"]
        }
        
        for general, specifics in generalizations.items():
            if general in candidate_text and any(s in trace_text for s in specifics):
                return True
        
        return False
    
    def _update_hypotheses(self, intent_candidates: List[Dict[str, Any]]):
        """Update intent hypothesis scores based on new evidence."""
        for candidate in intent_candidates:
            name = candidate["name"]
            confidence = candidate.get("confidence", 0.5)
            
            # Bayesian-style update
            self.intent_hypotheses[name] = (
                self.intent_hypotheses[name] * 0.7 + confidence * 0.3
            )
        
        # Decay unused hypotheses
        for name in list(self.intent_hypotheses.keys()):
            if name not in [c["name"] for c in intent_candidates]:
                self.intent_hypotheses[name] *= 0.9
                if self.intent_hypotheses[name] < 0.1:
                    del self.intent_hypotheses[name]
    
    def get_open_traces(self) -> List[IntentTrace]:
        """Get all currently open intent traces."""
        return [t for t in self.active_traces if t.is_active()]
    
    def get_closed_traces(self) -> List[IntentTrace]:
        """Get all closed intent traces."""
        return self.closed_traces
    
    def get_intent_summary(self) -> Dict[str, Any]:
        """Get a summary of current intent states."""
        open_traces = self.get_open_traces()
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "active_intents": len(open_traces),
            "closed_intents": len(self.closed_traces),
            "top_hypotheses": dict(sorted(
                self.intent_hypotheses.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "active_details": [
                {
                    "name": t.name,
                    "confidence": round(t.confidence, 2),
                    "age_turns": t.get_age_in_turns(self.turn_count),
                    "inactive_turns": t.get_inactivity_in_turns(self.turn_count)
                }
                for t in open_traces
            ]
        }
    
    def _log_opening(self, trace: IntentTrace):
        """Log intent opening to MemoryVault."""
        if self.memory_vault:
            self.memory_vault.log_intent_open(trace)
    
    def _log_closure(self, trace: IntentTrace):
        """Log intent closure to MemoryVault."""
        if self.memory_vault:
            self.memory_vault.log_intent_close(trace)
    
    def end_session(self):
        """End the current session and close all open intents."""
        logger.info(f"Ending session {self.session_id}")
        
        # Mark all open intents as abandoned
        for trace in self.get_open_traces():
            trace.abandon(reason="Session ended")
            self._log_closure(trace)
            self.closed_traces.append(trace)
        
        self.active_traces = []
        
        # Log session summary
        summary = self.get_intent_summary()
        logger.info(f"Session summary: {summary}")
        
        if self.memory_vault:
            self.memory_vault.log_session_end(summary)
