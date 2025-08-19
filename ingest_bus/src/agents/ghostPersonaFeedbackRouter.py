"""
Ghost Persona Feedback Router for TORI

This module enables Ghost personas to dynamically refine or retract hypotheses
based on trust layer signals, hallucination flags, verified concept score drops,
or user feedback. It creates a feedback loop where Ghosts can update their
reflections when new trust information becomes available.

Key Features:
- Monitors concepts in memory linked to each Ghost persona
- Detects trust downgrades via LoopRecord or verification UI
- Allows Ghosts to reflect again with updated trust signals
- Enables multi-agent review for flagged concepts
- Maintains audit trail of Ghost reflection updates
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Configure logging
logger = logging.getLogger("tori.ghost_feedback")

class TrustSignalType(str, Enum):
    """Types of trust signals that can trigger Ghost feedback."""
    HALLUCINATION_DETECTED = "hallucination_detected"
    VERIFICATION_FAILED = "verification_failed"
    USER_CORRECTION = "user_correction"
    CONCEPT_DOWNGRADED = "concept_downgraded"
    INTEGRITY_VIOLATION = "integrity_violation"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    SOURCE_RETRACTED = "source_retracted"

class GhostResponseType(str, Enum):
    """Types of Ghost responses to trust signals."""
    REFLECTION_UPDATE = "reflection_update"
    HYPOTHESIS_RETRACTION = "hypothesis_retraction"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    MULTI_AGENT_REVIEW = "multi_agent_review"
    CONCEPT_FLAGGING = "concept_flagging"
    REANALYSIS_REQUEST = "reanalysis_request"

@dataclass
class TrustSignal:
    """Trust signal that triggers Ghost feedback."""
    signal_id: str
    signal_type: TrustSignalType
    concept_id: str
    source_id: str  # video_id, document_id, etc.
    severity: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    timestamp: datetime
    
    # Context information
    original_confidence: float
    new_confidence: float
    affected_segments: List[str]
    user_feedback: Optional[str] = None

@dataclass
class GhostFeedbackResponse:
    """Ghost persona response to trust signal."""
    response_id: str
    ghost_persona: str
    signal_id: str
    response_type: GhostResponseType
    
    # Updated reflection content
    original_reflection: str
    updated_reflection: str
    confidence_change: float
    
    # Reasoning and metadata
    reasoning: str
    additional_context: str
    timestamp: datetime
    
    # Actions taken
    concepts_flagged: List[str]
    review_requested: bool
    retraction_complete: bool

class GhostPersonaFeedbackRouter:
    """
    Routes trust signals to appropriate Ghost personas and manages their
    feedback responses, creating a dynamic reflection system that adapts
    to new trust information.
    """
    
    def __init__(self):
        """Initialize the Ghost feedback router."""
        self.ghost_registry = {}  # persona_name -> GhostPersona
        self.concept_ghost_links = {}  # concept_id -> [ghost_personas]
        self.trust_signals = {}  # signal_id -> TrustSignal
        self.ghost_responses = {}  # response_id -> GhostFeedbackResponse
        self.feedback_bus = {}  # event_type -> [callback_functions]
        
        # Configuration
        self.trust_threshold = 0.7  # Threshold for triggering feedback
        self.response_timeout = 30.0  # Seconds to wait for Ghost responses
        self.auto_retraction_threshold = 0.3  # Auto-retract below this confidence
        
        logger.info("🤖 Ghost Persona Feedback Router initialized")
    
    def register_ghost_persona(
        self,
        persona_name: str,
        reflection_function: Callable,
        specialties: List[str] = None
    ):
        """
        Register a Ghost persona for feedback routing.
        
        Args:
            persona_name: Name of the Ghost persona
            reflection_function: Function to call for re-reflection
            specialties: Areas of expertise for this Ghost
        """
        self.ghost_registry[persona_name] = {
            "name": persona_name,
            "reflection_function": reflection_function,
            "specialties": specialties or [],
            "active_concepts": set(),
            "reflection_history": [],
            "trust_score": 1.0
        }
        
        logger.info(f"👻 Registered Ghost persona: {persona_name}")
    
    def link_concept_to_ghost(self, concept_id: str, ghost_persona: str):
        """Link a concept to a Ghost persona for monitoring."""
        if concept_id not in self.concept_ghost_links:
            self.concept_ghost_links[concept_id] = []
        
        if ghost_persona not in self.concept_ghost_links[concept_id]:
            self.concept_ghost_links[concept_id].append(ghost_persona)
            
            # Update Ghost's active concepts
            if ghost_persona in self.ghost_registry:
                self.ghost_registry[ghost_persona]["active_concepts"].add(concept_id)
        
        logger.debug(f"🔗 Linked concept {concept_id} to Ghost {ghost_persona}")
    
    def emit_trust_signal(
        self,
        concept_id: str,
        signal_type: TrustSignalType,
        source_id: str,
        original_confidence: float,
        new_confidence: float,
        evidence: Dict[str, Any] = None,
        user_feedback: str = None
    ) -> str:
        """
        Emit a trust signal that may trigger Ghost feedback.
        
        Args:
            concept_id: ID of the affected concept
            signal_type: Type of trust signal
            source_id: Source that triggered the signal
            original_confidence: Original confidence score
            new_confidence: Updated confidence score
            evidence: Supporting evidence for the signal
            user_feedback: Optional user feedback text
            
        Returns:
            Signal ID for tracking
        """
        signal_id = str(uuid.uuid4())
        
        # Calculate severity based on confidence drop
        confidence_drop = original_confidence - new_confidence
        severity = min(confidence_drop / original_confidence, 1.0) if original_confidence > 0 else 1.0
        
        trust_signal = TrustSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            concept_id=concept_id,
            source_id=source_id,
            severity=severity,
            evidence=evidence or {},
            timestamp=datetime.now(timezone.utc),
            original_confidence=original_confidence,
            new_confidence=new_confidence,
            affected_segments=[],  # Could be populated with segment IDs
            user_feedback=user_feedback
        )
        
        self.trust_signals[signal_id] = trust_signal
        
        logger.info(f"🚨 Trust signal emitted: {signal_type} for concept {concept_id} (severity: {severity:.2f})")
        
        # Route to affected Ghosts
        asyncio.create_task(self._route_trust_signal(trust_signal))
        
        return signal_id
    
    async def _route_trust_signal(self, trust_signal: TrustSignal):
        """Route trust signal to appropriate Ghost personas."""
        try:
            concept_id = trust_signal.concept_id
            
            # Find Ghosts linked to this concept
            affected_ghosts = self.concept_ghost_links.get(concept_id, [])
            
            if not affected_ghosts:
                logger.info(f"ℹ️  No Ghosts linked to concept {concept_id}")
                return
            
            logger.info(f"📡 Routing trust signal to {len(affected_ghosts)} Ghost(s): {affected_ghosts}")
            
            # Process each Ghost's response
            response_tasks = []
            for ghost_name in affected_ghosts:
                if ghost_name in self.ghost_registry:
                    task = asyncio.create_task(
                        self._get_ghost_response(ghost_name, trust_signal)
                    )
                    response_tasks.append(task)
            
            # Wait for all Ghost responses (with timeout)
            if response_tasks:
                responses = await asyncio.gather(*response_tasks, return_exceptions=True)
                
                # Process responses
                for response in responses:
                    if isinstance(response, GhostFeedbackResponse):
                        await self._process_ghost_response(response)
                    elif isinstance(response, Exception):
                        logger.error(f"❌ Ghost response failed: {response}")
            
        except Exception as e:
            logger.error(f"❌ Trust signal routing failed: {str(e)}")
    
    async def _get_ghost_response(
        self,
        ghost_name: str,
        trust_signal: TrustSignal
    ) -> GhostFeedbackResponse:
        """Get response from a specific Ghost persona."""
        try:
            ghost_data = self.ghost_registry[ghost_name]
            
            # Prepare context for Ghost
            context = {
                "concept_id": trust_signal.concept_id,
                "signal_type": trust_signal.signal_type,
                "severity": trust_signal.severity,
                "confidence_drop": trust_signal.original_confidence - trust_signal.new_confidence,
                "evidence": trust_signal.evidence,
                "user_feedback": trust_signal.user_feedback,
                "ghost_specialties": ghost_data["specialties"]
            }
            
            # Get original reflection (would typically come from memory)
            original_reflection = self._get_ghost_original_reflection(ghost_name, trust_signal.concept_id)
            
            # Generate Ghost response based on trust signal
            updated_reflection, reasoning, response_type, confidence_change = await self._generate_ghost_feedback(
                ghost_name, context, original_reflection
            )
            
            # Create response object
            response = GhostFeedbackResponse(
                response_id=str(uuid.uuid4()),
                ghost_persona=ghost_name,
                signal_id=trust_signal.signal_id,
                response_type=response_type,
                original_reflection=original_reflection,
                updated_reflection=updated_reflection,
                confidence_change=confidence_change,
                reasoning=reasoning,
                additional_context=json.dumps(context),
                timestamp=datetime.now(timezone.utc),
                concepts_flagged=[trust_signal.concept_id] if response_type == GhostResponseType.CONCEPT_FLAGGING else [],
                review_requested=response_type == GhostResponseType.MULTI_AGENT_REVIEW,
                retraction_complete=response_type == GhostResponseType.HYPOTHESIS_RETRACTION
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Ghost {ghost_name} response failed: {str(e)}")
            raise
    
    async def _generate_ghost_feedback(
        self,
        ghost_name: str,
        context: Dict[str, Any],
        original_reflection: str
    ) -> tuple[str, str, GhostResponseType, float]:
        """Generate Ghost feedback based on trust signal context."""
        
        # Determine response type based on severity and signal type
        severity = context["severity"]
        signal_type = context["signal_type"]
        confidence_drop = context["confidence_drop"]
        
        if severity > 0.8 or signal_type == TrustSignalType.HALLUCINATION_DETECTED:
            response_type = GhostResponseType.HYPOTHESIS_RETRACTION
            confidence_change = -0.9
            
            updated_reflection = f"Upon reflection with new trust information, I need to retract my earlier hypothesis. The evidence suggests I may have misinterpreted the source material. Original assessment: '{original_reflection[:100]}...' requires significant revision."
            
            reasoning = f"High severity trust signal ({severity:.2f}) indicates significant reliability issues. Retracting hypothesis to maintain system integrity."
            
        elif severity > 0.6 or confidence_drop > 0.3:
            response_type = GhostResponseType.REFLECTION_UPDATE
            confidence_change = -confidence_drop * 0.8
            
            updated_reflection = f"I'm revising my earlier reflection based on new trust signals. While my core insight about '{original_reflection[:50]}...' may still hold, I'm less confident about specific details. The evidence warrants a more cautious interpretation."
            
            reasoning = f"Moderate trust concerns (severity: {severity:.2f}) suggest refinement needed rather than full retraction."
            
        elif severity > 0.4:
            response_type = GhostResponseType.CONFIDENCE_ADJUSTMENT
            confidence_change = -confidence_drop * 0.5
            
            updated_reflection = f"Noting some uncertainty in my previous reflection: '{original_reflection}' - maintaining core assessment but with adjusted confidence based on new verification data."
            
            reasoning = f"Minor trust signal suggests confidence adjustment rather than content change."
            
        elif signal_type == TrustSignalType.USER_CORRECTION:
            response_type = GhostResponseType.MULTI_AGENT_REVIEW
            confidence_change = -0.2
            
            updated_reflection = f"User feedback has highlighted potential issues with my reflection: '{original_reflection}' - requesting multi-agent review to validate or correct my interpretation."
            
            reasoning = "User correction received - involving other agents for collaborative validation."
            
        else:
            response_type = GhostResponseType.CONCEPT_FLAGGING
            confidence_change = -0.1
            
            updated_reflection = f"Flagging concept for attention while maintaining reflection: '{original_reflection}' - monitoring for additional trust signals."
            
            reasoning = f"Low-level trust signal - flagging for monitoring without immediate changes."
        
        # Personalize based on Ghost type
        if ghost_name == "Ghost Collective":
            updated_reflection = f"🔮 {updated_reflection}"
        elif ghost_name == "Scholar":
            updated_reflection = f"📚 From an analytical perspective: {updated_reflection}"
        elif ghost_name == "Creator":
            updated_reflection = f"💡 Creative reassessment: {updated_reflection}"
        elif ghost_name == "Critic":
            updated_reflection = f"🔍 Critical review indicates: {updated_reflection}"
        
        return updated_reflection, reasoning, response_type, confidence_change
    
    def _get_ghost_original_reflection(self, ghost_name: str, concept_id: str) -> str:
        """Retrieve Ghost's original reflection for a concept."""
        # In a real implementation, this would query the memory system
        # For now, return a placeholder
        ghost_data = self.ghost_registry.get(ghost_name, {})
        reflection_history = ghost_data.get("reflection_history", [])
        
        # Find most recent reflection for this concept
        for reflection in reversed(reflection_history):
            if reflection.get("concept_id") == concept_id:
                return reflection.get("content", "No previous reflection found")
        
        return f"Previous reflection by {ghost_name} on concept {concept_id}"
    
    async def _process_ghost_response(self, response: GhostFeedbackResponse):
        """Process and store Ghost feedback response."""
        try:
            # Store response
            self.ghost_responses[response.response_id] = response
            
            # Update Ghost's reflection history
            ghost_name = response.ghost_persona
            if ghost_name in self.ghost_registry:
                self.ghost_registry[ghost_name]["reflection_history"].append({
                    "response_id": response.response_id,
                    "concept_id": self.trust_signals[response.signal_id].concept_id,
                    "content": response.updated_reflection,
                    "timestamp": response.timestamp.isoformat(),
                    "confidence_change": response.confidence_change
                })
            
            # Handle specific response types
            if response.response_type == GhostResponseType.MULTI_AGENT_REVIEW:
                await self._trigger_multi_agent_review(response)
            
            elif response.response_type == GhostResponseType.HYPOTHESIS_RETRACTION:
                await self._process_hypothesis_retraction(response)
            
            elif response.response_type == GhostResponseType.CONCEPT_FLAGGING:
                await self._flag_concept_for_attention(response)
            
            # Log the response
            logger.info(
                f"👻 {ghost_name} responded to trust signal: "
                f"{response.response_type} (confidence change: {response.confidence_change:+.2f})"
            )
            
            # Emit feedback processed event
            await self._emit_feedback_event("ghost_feedback_processed", response)
            
        except Exception as e:
            logger.error(f"❌ Failed to process Ghost response: {str(e)}")
    
    async def _trigger_multi_agent_review(self, response: GhostFeedbackResponse):
        """Trigger multi-agent review for a concept."""
        concept_id = self.trust_signals[response.signal_id].concept_id
        
        logger.info(f"🤝 Triggering multi-agent review for concept: {concept_id}")
        
        # Get all Ghosts except the one that requested review
        all_ghosts = [name for name in self.ghost_registry.keys() if name != response.ghost_persona]
        
        # Create review context
        review_context = {
            "requesting_ghost": response.ghost_persona,
            "concept_id": concept_id,
            "original_reflection": response.original_reflection,
            "updated_reflection": response.updated_reflection,
            "reasoning": response.reasoning
        }
        
        # Notify other Ghosts for collaborative review
        for ghost_name in all_ghosts:
            await self._request_collaborative_input(ghost_name, review_context)
    
    async def _process_hypothesis_retraction(self, response: GhostFeedbackResponse):
        """Process hypothesis retraction."""
        concept_id = self.trust_signals[response.signal_id].concept_id
        
        logger.info(f"🔄 Processing hypothesis retraction for concept: {concept_id}")
        
        # In a real system, this would:
        # 1. Remove or flag the concept in memory
        # 2. Update ConceptMesh relationships
        # 3. Log the retraction in LoopRecord
        # 4. Notify other systems of the change
        
        # For now, just log the action
        retraction_log = {
            "action": "hypothesis_retraction",
            "concept_id": concept_id,
            "ghost_persona": response.ghost_persona,
            "retraction_reason": response.reasoning,
            "timestamp": response.timestamp.isoformat()
        }
        
        logger.info(f"📝 Hypothesis retraction logged: {json.dumps(retraction_log)}")
    
    async def _flag_concept_for_attention(self, response: GhostFeedbackResponse):
        """Flag concept for attention without immediate action."""
        concept_id = self.trust_signals[response.signal_id].concept_id
        
        logger.info(f"🚩 Flagging concept for attention: {concept_id}")
        
        # Add to attention queue for human review or further monitoring
        flag_data = {
            "concept_id": concept_id,
            "flagged_by": response.ghost_persona,
            "reason": response.reasoning,
            "severity": "low",
            "timestamp": response.timestamp.isoformat()
        }
        
        # In real implementation, this would integrate with admin UI
        logger.info(f"🏷️  Concept flagged: {json.dumps(flag_data)}")
    
    async def _request_collaborative_input(self, ghost_name: str, review_context: Dict[str, Any]):
        """Request collaborative input from a Ghost for multi-agent review."""
        if ghost_name not in self.ghost_registry:
            return
        
        logger.info(f"🤝 Requesting collaborative input from {ghost_name}")
        
        # In a real implementation, this would call the Ghost's reflection function
        # with the review context to get their input on the disputed concept
    
    async def _emit_feedback_event(self, event_type: str, data: Any):
        """Emit feedback event to registered listeners."""
        if event_type in self.feedback_bus:
            for callback in self.feedback_bus[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Feedback event callback failed: {str(e)}")
    
    def subscribe_to_feedback_events(self, event_type: str, callback: Callable):
        """Subscribe to feedback events."""
        if event_type not in self.feedback_bus:
            self.feedback_bus[event_type] = []
        self.feedback_bus[event_type].append(callback)
    
    def get_ghost_status(self, ghost_name: str) -> Dict[str, Any]:
        """Get status and statistics for a Ghost persona."""
        if ghost_name not in self.ghost_registry:
            return {"error": "Ghost not found"}
        
        ghost_data = self.ghost_registry[ghost_name]
        
        return {
            "name": ghost_name,
            "specialties": ghost_data["specialties"],
            "active_concepts": len(ghost_data["active_concepts"]),
            "total_reflections": len(ghost_data["reflection_history"]),
            "trust_score": ghost_data["trust_score"],
            "recent_responses": len([
                r for r in self.ghost_responses.values()
                if r.ghost_persona == ghost_name and 
                (datetime.now(timezone.utc) - r.timestamp).total_seconds() < 3600
            ])
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            "registered_ghosts": len(self.ghost_registry),
            "monitored_concepts": len(self.concept_ghost_links),
            "total_trust_signals": len(self.trust_signals),
            "total_responses": len(self.ghost_responses),
            "recent_signals": len([
                s for s in self.trust_signals.values()
                if (datetime.now(timezone.utc) - s.timestamp).total_seconds() < 3600
            ]),
            "response_types": {
                response_type.value: len([
                    r for r in self.ghost_responses.values()
                    if r.response_type == response_type
                ])
                for response_type in GhostResponseType
            }
        }

# Global instance
ghost_feedback_router = GhostPersonaFeedbackRouter()

# Example event handler registration
async def handle_trust_signal_update(concept_id: str, status: Dict[str, Any]):
    """Example handler for trust signal updates."""
    if status.get("integrity_score", 1.0) < 0.7:
        ghost_feedback_router.emit_trust_signal(
            concept_id=concept_id,
            signal_type=TrustSignalType.VERIFICATION_FAILED,
            source_id=status.get("source_id", "unknown"),
            original_confidence=status.get("original_confidence", 1.0),
            new_confidence=status.get("integrity_score", 0.5),
            evidence=status
        )

# Register the handler
ghost_feedback_router.subscribe_to_feedback_events("trustSignalUpdate", handle_trust_signal_update)
