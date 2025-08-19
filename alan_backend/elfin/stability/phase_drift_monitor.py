"""
Phase Drift Monitor for ELFIN Stability Framework.

This module provides real-time monitoring of phase drift between ELFIN concepts
and the ψ-Sync oscillator phases, with adaptive actions when drift exceeds thresholds.
"""

import os
import logging
import time
import math
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Callable, Optional, Any, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftThresholdType(Enum):
    """Types of threshold specifications for phase drift."""
    
    RADIANS = auto()       # Absolute radian value
    PI_RATIO = auto()      # Ratio of π (e.g., π/4)
    PERCENTAGE = auto()    # Percentage of 2π
    STANDARD_DEV = auto()  # Standard deviations from mean


class AdaptiveActionType(Enum):
    """Types of adaptive actions to trigger on drift threshold violation."""
    
    NOTIFY = auto()        # Notification only
    ADAPT_PLAN = auto()    # Adapt execution plan
    EXECUTE_AGENT = auto() # Execute specific agent
    CUSTOM_ACTION = auto() # Custom action function


class PhaseDriftMonitor:
    """
    Monitor and react to phase drift between concepts and oscillators.
    
    This class provides real-time monitoring of phase synchronization
    between ELFIN concepts and the underlying phase oscillators, with
    the ability to trigger adaptive actions when drift exceeds thresholds.
    """
    
    def __init__(
        self,
        concept_to_psi_map: Dict[str, int],
        thresholds: Optional[Dict[str, float]] = None,
        banksy_monitor = None
    ):
        """
        Initialize phase drift monitor.
        
        Args:
            concept_to_psi_map: Mapping from concept IDs to oscillator indices
            thresholds: Default thresholds for each concept
            banksy_monitor: Optional Banksy PsiSyncMonitor instance
        """
        self.concept_to_psi = concept_to_psi_map
        self.thresholds = thresholds or {}
        self.banksy_monitor = banksy_monitor
        
        # Phase reference values
        self.reference_phases = {}
        
        # Adaptive actions (concept_id -> list of actions)
        self.adaptive_actions = {}
        
        # Stats
        self.drift_history = {}
        for concept_id in concept_to_psi_map:
            self.drift_history[concept_id] = []
        
        # For minimal implementation
        self.minimal_phase_state = {}
        for concept_id in concept_to_psi_map:
            self.minimal_phase_state[concept_id] = 0.0
            
        logger.info("Using minimal PhaseDriftMonitor implementation")
    
    def set_reference_phase(self, concept_id: str, phase: float):
        """
        Set reference phase for a concept.
        
        Args:
            concept_id: ID of the concept
            phase: Reference phase value
        """
        if concept_id in self.concept_to_psi:
            self.reference_phases[concept_id] = phase
            logger.info(f"Set reference phase for {concept_id}: {phase}")
        else:
            logger.warning(f"Unknown concept ID: {concept_id}")
    
    def measure_drift(self, concept_id: str, current_phase: float) -> float:
        """
        Measure phase drift for a concept.
        
        Args:
            concept_id: ID of the concept
            current_phase: Current phase value
            
        Returns:
            Phase drift in radians
        """
        if concept_id not in self.concept_to_psi:
            logger.warning(f"Unknown concept ID: {concept_id}")
            return 0.0
            
        if concept_id not in self.reference_phases:
            # Set current phase as reference if not set
            self.set_reference_phase(concept_id, current_phase)
            return 0.0
            
        # Calculate minimum circular distance
        reference = self.reference_phases[concept_id]
        drift = abs((current_phase - reference + math.pi) % (2 * math.pi) - math.pi)
        
        # Store in history
        self.drift_history[concept_id].append(drift)
        
        # Update minimal phase state
        self.minimal_phase_state[concept_id] = current_phase
        
        return drift
    
    def register_adaptive_action(
        self,
        concept_id: str,
        threshold: float,
        threshold_type: DriftThresholdType = DriftThresholdType.RADIANS,
        action_type: AdaptiveActionType = AdaptiveActionType.NOTIFY,
        action_fn: Optional[Callable] = None,
        description: str = ""
    ):
        """
        Register an adaptive action to trigger when drift exceeds threshold.
        
        Args:
            concept_id: ID of the concept to monitor
            threshold: Threshold value
            threshold_type: Type of threshold
            action_type: Type of action to trigger
            action_fn: Custom action function
            description: Description of the action
        """
        if concept_id not in self.concept_to_psi:
            logger.warning(f"Unknown concept ID: {concept_id}")
            return
            
        # Convert threshold to radians
        if threshold_type == DriftThresholdType.PI_RATIO:
            threshold_radians = threshold * math.pi
        elif threshold_type == DriftThresholdType.PERCENTAGE:
            threshold_radians = threshold * 2 * math.pi
        elif threshold_type == DriftThresholdType.STANDARD_DEV:
            # Will be computed dynamically based on history
            threshold_radians = None
        else:
            threshold_radians = threshold
        
        # Create action entry
        action = {
            'concept_id': concept_id,
            'threshold': threshold,
            'threshold_type': threshold_type,
            'threshold_radians': threshold_radians,
            'action_type': action_type,
            'action_fn': action_fn,
            'description': description,
            'last_triggered': 0.0
        }
        
        # Add to actions list
        if concept_id not in self.adaptive_actions:
            self.adaptive_actions[concept_id] = []
            
        self.adaptive_actions[concept_id].append(action)
        
        logger.info(f"Registered adaptive action for {concept_id}")
    
    def check_and_trigger_actions(self) -> List[Dict]:
        """
        Check all actions and trigger those whose thresholds are exceeded.
        
        Returns:
            List of triggered action results
        """
        results = []
        
        for concept_id, actions in self.adaptive_actions.items():
            if concept_id not in self.concept_to_psi:
                continue
                
            # Get current phase
            if self.banksy_monitor:
                # Use Banksy monitor if available
                psi_idx = self.concept_to_psi[concept_id]
                current_phase = self.banksy_monitor.get_phase(psi_idx)
            else:
                # Use minimal implementation
                current_phase = self.minimal_phase_state.get(concept_id, 0.0)
                
            # Measure drift
            drift = self.measure_drift(concept_id, current_phase)
            
            # Check actions
            for action in actions:
                # Skip actions with no action function
                if action['action_fn'] is None and action['action_type'] not in [
                    AdaptiveActionType.NOTIFY
                ]:
                    continue
                    
                # Compute threshold in radians
                if action['threshold_type'] == DriftThresholdType.STANDARD_DEV:
                    if len(self.drift_history[concept_id]) < 10:
                        continue  # Not enough history
                        
                    # Compute dynamic threshold based on history
                    mean = np.mean(self.drift_history[concept_id])
                    std = np.std(self.drift_history[concept_id])
                    threshold_radians = mean + action['threshold'] * std
                else:
                    threshold_radians = action['threshold_radians']
                
                # Check if threshold is exceeded
                if drift > threshold_radians:
                    # Check cooldown (don't trigger too often)
                    if time.time() - action['last_triggered'] < 1.0:
                        continue
                        
                    # Trigger action
                    if action['action_type'] == AdaptiveActionType.NOTIFY:
                        logger.info(f"Phase drift notification for {concept_id}: {drift}")
                        result = {
                            'action': str(action['action_type']),
                            'concept': concept_id,
                            'drift': drift,
                            'threshold': threshold_radians,
                            'status': 'success'
                        }
                    else:
                        # Call custom action function
                        try:
                            result = action['action_fn'](
                                concept_id, drift, threshold_radians, action['action_type']
                            )
                        except Exception as e:
                            logger.error(f"Error in adaptive action: {e}")
                            result = {
                                'action': str(action['action_type']),
                                'concept': concept_id,
                                'drift': drift,
                                'threshold': threshold_radians,
                                'status': 'error',
                                'error': str(e)
                            }
                    
                    # Update last triggered
                    action['last_triggered'] = time.time()
                    
                    # Add to results
                    results.append(result)
        
        return results
    
    def get_phase_statistics(self, concept_id: str) -> Dict[str, float]:
        """
        Get statistics about phase drift for a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary of statistics
        """
        if concept_id not in self.concept_to_psi:
            logger.warning(f"Unknown concept ID: {concept_id}")
            return {}
            
        if not self.drift_history[concept_id]:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'current': 0.0
            }
            
        history = self.drift_history[concept_id]
        
        return {
            'mean': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'current': float(history[-1]) if history else 0.0
        }
    
    def create_lyapunov_predicate(self, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Create a Lyapunov predicate for a set of concepts.
        
        This creates a symbolic Lyapunov function that can be used to
        verify stability of the concept synchronization.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            Lyapunov predicate dictionary
        """
        # Check that all concepts are valid
        for concept_id in concept_ids:
            if concept_id not in self.concept_to_psi:
                logger.warning(f"Unknown concept ID: {concept_id}")
                return {}
        
        # In a real implementation, this would create a Lyapunov function
        # based on phase synchronization properties. For now, we just
        # return a placeholder.
        
        # Symbolic representation of the Lyapunov function
        symbolic_form = "V(x) = x^T I x"
        
        # Create predicate
        predicate = {
            'type': 'lyapunov',
            'concepts': concept_ids,
            'symbolic_form': symbolic_form,
            'description': f"Lyapunov function for concepts: {', '.join(concept_ids)}"
        }
        
        return predicate
