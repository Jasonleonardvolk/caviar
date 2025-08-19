"""
🔄 ACTIVATION FEEDBACK LOOP
Back-propagates adjustments from soliton/concept activation to phase encoder

This creates a closed-loop system where:
- Failed bindings trigger re-encoding
- Successful resonances strengthen phase coupling
- Oscillatory feedback nudges sampling parameters
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger("feedback.oscillator")


@dataclass
class FeedbackEvent:
    """Single feedback event from activation"""
    concept_id: str
    timestamp: datetime
    event_type: str  # 'binding_failed', 'resonance_success', 'phase_drift', etc.
    success_state: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackState:
    """Current state of feedback system"""
    concept_id: str
    requires_reencoding: bool = False
    suggested_region: Optional[Dict[str, np.ndarray]] = None
    suggested_resolution: Optional[int] = None
    phase_shift_suggestion: float = 0.0
    amplitude_scaling: float = 1.0
    confidence: float = 0.5
    history: List[FeedbackEvent] = field(default_factory=list)


class OscillatorFeedbackLoop:
    """
    🧠 Intelligent feedback system for phase encoding optimization
    
    Features:
    - Monitors soliton binding success/failure
    - Detects phase drift and instabilities
    - Suggests re-encoding parameters
    - Learns from historical patterns
    """
    
    def __init__(self, 
                 history_size: int = 100,
                 learning_rate: float = 0.1):
        self.feedback_states: Dict[str, FeedbackState] = {}
        self.global_history = deque(maxlen=history_size)
        self.learning_rate = learning_rate
        
        # Success metrics
        self.success_rates: Dict[str, float] = {}
        self.phase_stability: Dict[str, float] = {}
        
        # Callbacks for encoder hooks
        self.encoder_callbacks: List[Callable] = []
        
        logger.info("🔄 Oscillator Feedback Loop initialized")
    
    def record_activation_event(self,
                               concept_id: str,
                               event_type: str,
                               success: bool,
                               metadata: Optional[Dict[str, Any]] = None) -> FeedbackEvent:
        """
        📝 Record an activation event
        
        Event types:
        - 'binding_failed': Soliton failed to bind
        - 'binding_success': Successful binding
        - 'resonance_detected': Phase resonance found
        - 'phase_drift': Phase has drifted significantly
        - 'vortex_formed': Phase vortex detected
        - 'amplitude_decay': Amplitude below threshold
        """
        event = FeedbackEvent(
            concept_id=concept_id,
            timestamp=datetime.now(),
            event_type=event_type,
            success_state=success,
            metadata=metadata or {}
        )
        
        # Update feedback state
        if concept_id not in self.feedback_states:
            self.feedback_states[concept_id] = FeedbackState(concept_id=concept_id)
        
        state = self.feedback_states[concept_id]
        state.history.append(event)
        
        # Global history
        self.global_history.append(event)
        
        # Analyze event and update suggestions
        self._analyze_event(event, state)
        
        # Update success rate
        self._update_success_metrics(concept_id, success)
        
        logger.info(f"📝 Recorded {event_type} for concept '{concept_id}' (success={success})")
        
        return event
    
    def _analyze_event(self, event: FeedbackEvent, state: FeedbackState):
        """Analyze event and update feedback state"""
        
        if event.event_type == 'binding_failed':
            # Soliton binding failed - suggest re-encoding
            state.requires_reencoding = True
            state.confidence *= 0.8  # Reduce confidence
            
            # Suggest phase adjustment
            if 'phase_mismatch' in event.metadata:
                mismatch = event.metadata['phase_mismatch']
                state.phase_shift_suggestion += self.learning_rate * mismatch
            
            # Suggest amplitude boost
            if 'amplitude' in event.metadata and event.metadata['amplitude'] < 0.3:
                state.amplitude_scaling *= 1.2
            
            # Suggest region expansion
            if state.suggested_resolution:
                state.suggested_resolution = min(200, int(state.suggested_resolution * 1.2))
            else:
                state.suggested_resolution = 100
                
        elif event.event_type == 'binding_success':
            # Success - reinforce current parameters
            state.confidence = min(1.0, state.confidence * 1.1)
            
            # Reduce need for re-encoding
            if state.requires_reencoding and state.confidence > 0.7:
                state.requires_reencoding = False
                
        elif event.event_type == 'phase_drift':
            # Phase has drifted - suggest correction
            if 'drift_amount' in event.metadata:
                drift = event.metadata['drift_amount']
                state.phase_shift_suggestion -= drift * 0.5  # Compensate
            
            state.requires_reencoding = True
            
        elif event.event_type == 'vortex_formed':
            # Vortex detected - need smoother encoding
            state.requires_reencoding = True
            
            # Suggest finer resolution to resolve vortex
            if state.suggested_resolution:
                state.suggested_resolution = int(state.suggested_resolution * 1.5)
            else:
                state.suggested_resolution = 150
                
        elif event.event_type == 'amplitude_decay':
            # Amplitude too low - boost it
            state.amplitude_scaling *= 1.5
            state.requires_reencoding = True
            
        elif event.event_type == 'resonance_detected':
            # Good resonance - remember these parameters
            state.confidence = min(1.0, state.confidence * 1.2)
            
            # Store successful parameters
            if 'phase' in event.metadata:
                event.suggested_adjustments['optimal_phase'] = event.metadata['phase']
            if 'amplitude' in event.metadata:
                event.suggested_adjustments['optimal_amplitude'] = event.metadata['amplitude']
    
    def _update_success_metrics(self, concept_id: str, success: bool):
        """Update running success rate"""
        if concept_id not in self.success_rates:
            self.success_rates[concept_id] = 0.5  # Initial estimate
        
        # Exponential moving average
        alpha = 0.1
        self.success_rates[concept_id] = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * self.success_rates[concept_id]
        )
    
    def get_feedback(self, concept_id: str) -> FeedbackState:
        """
        🎯 Get current feedback state for concept
        
        Returns FeedbackState with:
        - requires_reencoding: Whether phase encoding should be redone
        - suggested_region: New sampling region bounds
        - phase_shift_suggestion: Phase offset to apply
        - amplitude_scaling: Amplitude multiplier
        """
        if concept_id not in self.feedback_states:
            return FeedbackState(concept_id=concept_id)
        
        return self.feedback_states[concept_id]
    
    def suggest_encoding_parameters(self, 
                                   concept_id: str,
                                   current_region: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        🔮 Suggest optimized encoding parameters
        
        Based on feedback history, suggests:
        - Modified region sampling
        - Phase corrections
        - Amplitude adjustments
        - Resolution changes
        """
        state = self.get_feedback(concept_id)
        
        suggestions = {
            'requires_reencoding': state.requires_reencoding,
            'confidence': state.confidence
        }
        
        # Region modifications
        if state.requires_reencoding:
            new_region = {}
            
            for coord, values in current_region.items():
                if state.suggested_resolution:
                    # Increase resolution
                    new_region[coord] = np.linspace(
                        values[0], values[-1], 
                        state.suggested_resolution
                    )
                else:
                    new_region[coord] = values
                
                # Expand bounds slightly if many failures
                if state.confidence < 0.3:
                    span = values[-1] - values[0]
                    expansion = 0.2 * span
                    new_region[coord] = np.linspace(
                        values[0] - expansion/2,
                        values[-1] + expansion/2,
                        len(values)
                    )
            
            suggestions['suggested_region'] = new_region
        
        # Phase adjustments
        if abs(state.phase_shift_suggestion) > 0.1:
            suggestions['phase_offset'] = state.phase_shift_suggestion
        
        # Amplitude adjustments
        if state.amplitude_scaling != 1.0:
            suggestions['amplitude_scale'] = state.amplitude_scaling
        
        # Resolution suggestion
        if state.suggested_resolution:
            suggestions['resolution'] = state.suggested_resolution
        
        # Historical optimal parameters
        recent_successes = [
            e for e in state.history[-10:]
            if e.success_state and e.suggested_adjustments
        ]
        
        if recent_successes:
            # Average successful parameters
            optimal_phases = [
                e.suggested_adjustments.get('optimal_phase', 0)
                for e in recent_successes
                if 'optimal_phase' in e.suggested_adjustments
            ]
            if optimal_phases:
                suggestions['target_phase'] = np.mean(optimal_phases)
        
        logger.info(f"🔮 Suggestions for '{concept_id}': re-encode={state.requires_reencoding}, confidence={state.confidence:.2f}")
        
        return suggestions
    
    def register_encoder_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register callback for encoder integration"""
        self.encoder_callbacks.append(callback)
        logger.info(f"📎 Registered encoder callback: {callback.__name__}")
    
    def trigger_reencoding(self, concept_id: str):
        """
        🔄 Trigger re-encoding through callbacks
        
        Notifies registered encoders to re-process concept
        """
        state = self.get_feedback(concept_id)
        
        if not state.requires_reencoding:
            logger.info(f"ℹ️ Concept '{concept_id}' doesn't require re-encoding")
            return
        
        # Get suggestions
        current_region = getattr(state, 'current_region', {
            'r': np.linspace(2, 10, 50)
        })
        
        suggestions = self.suggest_encoding_parameters(concept_id, current_region)
        
        # Notify callbacks
        for callback in self.encoder_callbacks:
            try:
                callback(concept_id, suggestions)
                logger.info(f"✅ Triggered re-encoding for '{concept_id}' via {callback.__name__}")
            except Exception as e:
                logger.error(f"❌ Callback failed: {e}")
        
        # Reset state after triggering
        state.requires_reencoding = False
        state.phase_shift_suggestion *= 0.5  # Decay suggestions
        state.amplitude_scaling = 1.0 + (state.amplitude_scaling - 1.0) * 0.5
    
    def compute_phase_stability(self, concept_id: str, window_size: int = 10) -> float:
        """
        📊 Compute phase stability metric
        
        Measures how stable the phase has been over recent events
        """
        if concept_id not in self.feedback_states:
            return 1.0  # Assume stable if no data
        
        state = self.feedback_states[concept_id]
        recent_events = state.history[-window_size:]
        
        if len(recent_events) < 2:
            return 1.0
        
        # Count phase-related issues
        phase_issues = sum(
            1 for e in recent_events
            if e.event_type in ['phase_drift', 'vortex_formed', 'binding_failed']
        )
        
        stability = 1.0 - (phase_issues / len(recent_events))
        
        # Cache result
        self.phase_stability[concept_id] = stability
        
        return stability
    
    def export_feedback_report(self, output_path: str):
        """
        📊 Export comprehensive feedback report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'concepts': {},
            'global_metrics': {
                'total_events': len(self.global_history),
                'average_success_rate': np.mean(list(self.success_rates.values())) if self.success_rates else 0,
                'concepts_needing_reencoding': sum(
                    1 for s in self.feedback_states.values() 
                    if s.requires_reencoding
                )
            }
        }
        
        # Per-concept analysis
        for concept_id, state in self.feedback_states.items():
            concept_report = {
                'requires_reencoding': state.requires_reencoding,
                'confidence': state.confidence,
                'success_rate': self.success_rates.get(concept_id, 0),
                'phase_stability': self.phase_stability.get(concept_id, 1.0),
                'total_events': len(state.history),
                'recent_events': [
                    {
                        'type': e.event_type,
                        'success': e.success_state,
                        'timestamp': e.timestamp.isoformat()
                    }
                    for e in state.history[-5:]  # Last 5 events
                ],
                'suggestions': {
                    'phase_shift': state.phase_shift_suggestion,
                    'amplitude_scale': state.amplitude_scaling,
                    'resolution': state.suggested_resolution
                }
            }
            
            report['concepts'][concept_id] = concept_report
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Exported feedback report to {output_path}")
    
    def create_phase_encoder_hook(self):
        """
        🔗 Create hook for phase encoder integration
        
        Returns a function that can be called from phase_encode.py
        """
        def encoder_hook(concept_id: str, encoding_params: Dict[str, Any]) -> Dict[str, Any]:
            """
            Hook to be called before phase encoding
            
            Modifies encoding parameters based on feedback
            """
            suggestions = self.suggest_encoding_parameters(
                concept_id,
                encoding_params.get('region_sample', {})
            )
            
            # Apply suggestions to parameters
            if suggestions.get('requires_reencoding'):
                if 'suggested_region' in suggestions:
                    encoding_params['region_sample'] = suggestions['suggested_region']
                
                if 'phase_offset' in suggestions:
                    encoding_params['phase_offset'] = suggestions['phase_offset']
                
                if 'amplitude_scale' in suggestions:
                    encoding_params['amplitude_scale'] = suggestions['amplitude_scale']
                
                logger.info(f"🔗 Applied feedback adjustments to encoding for '{concept_id}'")
            
            return encoding_params
        
        return encoder_hook


# Global feedback instance
feedback_loop = OscillatorFeedbackLoop()


# Example integration with phase encoder
def integrate_with_encoder():
    """Example of how to integrate with phase_encode.py"""
    
    # Add this to phase_encode.py:
    from feedback.oscillator_feedback import feedback_loop
    
    def encode_curvature_to_phase_with_feedback(
        concept_id: str,
        kretschmann_scalar,
        coords,
        region_sample,
        **kwargs
    ):
        # Get feedback adjustments
        hook = feedback_loop.create_phase_encoder_hook()
        params = {
            'region_sample': region_sample,
            **kwargs
        }
        
        # Apply feedback
        adjusted_params = hook(concept_id, params)
        
        # Call original encoder with adjusted parameters
        from phase_encode import encode_curvature_to_phase
        result = encode_curvature_to_phase(
            kretschmann_scalar=kretschmann_scalar,
            coords=coords,
            region_sample=adjusted_params['region_sample'],
            **adjusted_params
        )
        
        return result


# Example usage
if __name__ == "__main__":
    # Simulate feedback events
    
    # Failed binding
    feedback_loop.record_activation_event(
        concept_id="BH_Singularity",
        event_type="binding_failed",
        success=False,
        metadata={
            'phase_mismatch': 0.5,
            'amplitude': 0.2
        }
    )
    
    # Phase drift
    feedback_loop.record_activation_event(
        concept_id="BH_Singularity",
        event_type="phase_drift",
        success=False,
        metadata={
            'drift_amount': 0.3
        }
    )
    
    # Get feedback
    feedback = feedback_loop.get_feedback("BH_Singularity")
    print(f"Requires re-encoding: {feedback.requires_reencoding}")
    print(f"Confidence: {feedback.confidence:.2f}")
    
    # Get suggestions
    current_region = {'r': np.linspace(2, 10, 50)}
    suggestions = feedback_loop.suggest_encoding_parameters("BH_Singularity", current_region)
    print(f"Suggestions: {suggestions}")
    
    # Successful resonance
    feedback_loop.record_activation_event(
        concept_id="Event_Horizon",
        event_type="resonance_detected",
        success=True,
        metadata={
            'phase': 1.57,
            'amplitude': 0.8
        }
    )
    
    # Compute stability
    stability = feedback_loop.compute_phase_stability("BH_Singularity")
    print(f"Phase stability: {stability:.2f}")
    
    # Export report
    feedback_loop.export_feedback_report("feedback_report.json")
