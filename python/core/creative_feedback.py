#!/usr/bin/env python3
"""
Creative-Singularity Feedback Loop - Entropy injection for creative exploration
Manages the delicate balance between stability and creative chaos
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
import logging
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class CreativeMode(Enum):
    """Creative exploration modes"""
    STABLE = "stable"           # Normal operation
    EXPLORING = "exploring"     # Active entropy injection
    CONSOLIDATING = "consolidating"  # Post-exploration stabilization
    EMERGENCY = "emergency"     # Emergency damping

@dataclass
class EntropyInjection:
    """Record of entropy injection event"""
    timestamp: datetime
    mode: CreativeMode
    lambda_factor: float
    duration_steps: int
    trigger_novelty: float
    trigger_reason: str
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'mode': self.mode.value,
            'lambda_factor': self.lambda_factor,
            'duration_steps': self.duration_steps,
            'trigger_novelty': self.trigger_novelty,
            'trigger_reason': self.trigger_reason,
            'outcomes': self.outcomes
        }

class AestheticRegularizer:
    """Maintains aesthetic constraints during creative exploration"""
    
    def __init__(self):
        self.style_metrics = {
            'coherence': 0.5,      # Preference for coherent outputs
            'diversity': 0.3,      # Encourage diverse explorations
            'stability': 0.2       # Maintain some stability
        }
        self.baseline_performance = {}
        
    def evaluate(self, state: Dict[str, Any]) -> float:
        """Evaluate aesthetic quality of current state"""
        score = 0.0
        
        # Coherence metric
        if 'coherence_state' in state:
            coherence_score = {
                'local': 0.3,
                'global': 0.7,
                'critical': 0.5  # Critical can be good for creativity
            }.get(state['coherence_state'], 0.5)
            score += self.style_metrics['coherence'] * coherence_score
        
        # Diversity metric (based on novelty)
        if 'novelty_score' in state:
            # Sweet spot around 0.6 novelty
            novelty = state['novelty_score']
            diversity_score = 1.0 - abs(0.6 - novelty)
            score += self.style_metrics['diversity'] * diversity_score
        
        # Stability metric
        if 'lambda_max' in state:
            # Prefer moderate eigenvalues
            lambda_max = state['lambda_max']
            if lambda_max < 0.05:
                stability_score = 1.0 - lambda_max / 0.05
            else:
                stability_score = 0.0
            score += self.style_metrics['stability'] * stability_score
        
        return score
    
    def should_prune(self, aesthetic_score: float, 
                    baseline_score: float,
                    tolerance: float = 0.8) -> bool:
        """Determine if exploration should be pruned"""
        return aesthetic_score < baseline_score * tolerance

class CreativeSingularityFeedback:
    """
    Manages creative exploration through controlled entropy injection
    """
    
    def __init__(self, 
                 novelty_threshold_high: float = 0.7,
                 novelty_threshold_low: float = 0.2,
                 max_exploration_steps: int = 1000):
        
        self.novelty_threshold_high = novelty_threshold_high
        self.novelty_threshold_low = novelty_threshold_low
        self.max_exploration_steps = max_exploration_steps
        
        # Current state
        self.mode = CreativeMode.STABLE
        self.current_injection: Optional[EntropyInjection] = None
        self.steps_in_mode = 0
        
        # History
        self.injection_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        
        # Aesthetic regularizer
        self.regularizer = AestheticRegularizer()
        
        # Safety limits
        self.max_lambda_allowed = 0.1  # Hard safety limit
        self.emergency_threshold = 0.08
        
        # Metrics
        self.metrics = {
            'total_injections': 0,
            'successful_explorations': 0,
            'pruned_explorations': 0,
            'emergency_stops': 0,
            'total_creative_gain': 0.0
        }
        
        logger.info("Creative-Singularity Feedback initialized")
    
    def evaluate_creative_potential(self, 
                                  current_state: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate whether to inject entropy for creative exploration
        Returns (should_inject, lambda_factor, reason)
        """
        novelty = current_state.get('novelty_score', 0.0)
        lambda_max = current_state.get('lambda_max', 0.0)
        
        # Emergency check first
        if lambda_max > self.emergency_threshold:
            return False, 0.5, "emergency_damping_required"
        
        # Check if already exploring
        if self.mode == CreativeMode.EXPLORING:
            # Check if we should continue or stop
            if self.steps_in_mode > self.max_exploration_steps:
                return False, 1.0, "exploration_timeout"
            
            # Check aesthetic constraints
            aesthetic_score = self.regularizer.evaluate(current_state)
            baseline = self.regularizer.baseline_performance.get('score', 0.5)
            
            if self.regularizer.should_prune(aesthetic_score, baseline):
                return False, 0.8, "aesthetic_constraint_violated"
            
            # Continue exploration with adaptive factor
            if novelty > 0.8:
                return True, 1.2, "high_novelty_continuation"
            else:
                return True, 1.1, "exploration_continuation"
        
        # Not currently exploring - check if we should start
        if novelty > self.novelty_threshold_high:
            # High novelty - inject entropy for exploration
            factor = 1.5 + (novelty - self.novelty_threshold_high) * 0.5
            return True, min(factor, 2.0), "high_novelty_trigger"
        
        elif novelty < self.novelty_threshold_low:
            # Low novelty - mild tightening to encourage variation
            return True, 0.8, "low_novelty_trigger"
        
        # Moderate novelty - no change needed
        return False, 1.0, "stable_novelty"
    
    def inject_entropy(self, 
                      lambda_factor: float,
                      duration_steps: int,
                      trigger_novelty: float,
                      trigger_reason: str) -> EntropyInjection:
        """Start entropy injection"""
        injection = EntropyInjection(
            timestamp=datetime.now(timezone.utc),
            mode=CreativeMode.EXPLORING,
            lambda_factor=lambda_factor,
            duration_steps=duration_steps,
            trigger_novelty=trigger_novelty,
            trigger_reason=trigger_reason
        )
        
        self.current_injection = injection
        self.mode = CreativeMode.EXPLORING
        self.steps_in_mode = 0
        self.metrics['total_injections'] += 1
        
        logger.info(f"🎲 Entropy injection started: λ_factor={lambda_factor:.2f}, "
                   f"duration={duration_steps}, reason={trigger_reason}")
        
        return injection
    
    def update(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update creative feedback loop
        Returns action dictionary
        """
        self.steps_in_mode += 1
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'mode': self.mode.value,
            'novelty': current_state.get('novelty_score', 0.0),
            'lambda_max': current_state.get('lambda_max', 0.0),
            'aesthetic_score': self.regularizer.evaluate(current_state)
        })
        
        # Evaluate creative potential
        should_inject, lambda_factor, reason = self.evaluate_creative_potential(current_state)
        
        action = {
            'mode': self.mode.value,
            'action': 'none',
            'lambda_factor': 1.0,
            'reason': reason
        }
        
        # Handle mode transitions
        if self.mode == CreativeMode.STABLE:
            if should_inject:
                # Start exploration
                duration = min(self.max_exploration_steps, 100 + int(current_state.get('novelty_score', 0.5) * 200))
                self.inject_entropy(lambda_factor, duration, 
                                  current_state.get('novelty_score', 0.0), reason)
                action['action'] = 'inject_entropy'
                action['lambda_factor'] = lambda_factor
                
        elif self.mode == CreativeMode.EXPLORING:
            if not should_inject or self.steps_in_mode >= self.current_injection.duration_steps:
                # End exploration
                self._end_exploration(current_state)
                action['action'] = 'end_exploration'
                action['mode'] = CreativeMode.CONSOLIDATING.value
            else:
                # Continue exploration
                action['action'] = 'continue_exploration'
                action['lambda_factor'] = lambda_factor
                
        elif self.mode == CreativeMode.CONSOLIDATING:
            # Consolidation period after exploration
            if self.steps_in_mode > 50:  # Fixed consolidation period
                self.mode = CreativeMode.STABLE
                self.steps_in_mode = 0
                logger.info("Returned to stable mode after consolidation")
            action['action'] = 'consolidating'
            action['lambda_factor'] = 0.9  # Mild damping during consolidation
            
        elif self.mode == CreativeMode.EMERGENCY:
            # Emergency mode
            if current_state.get('lambda_max', 0.0) < self.emergency_threshold * 0.5:
                self.mode = CreativeMode.STABLE
                self.steps_in_mode = 0
                logger.info("Exited emergency mode")
            action['action'] = 'emergency_damping'
            action['lambda_factor'] = 0.5
        
        # Check for emergency condition
        if current_state.get('lambda_max', 0.0) > self.emergency_threshold:
            self.mode = CreativeMode.EMERGENCY
            self.steps_in_mode = 0
            self.metrics['emergency_stops'] += 1
            action['action'] = 'emergency_damping'
            action['lambda_factor'] = 0.5
            logger.warning(f"⚠️ Emergency mode activated! λ_max={current_state.get('lambda_max', 0.0):.3f}")
        
        return action
    
    def _end_exploration(self, final_state: Dict[str, Any]):
        """End current exploration and evaluate outcomes"""
        if not self.current_injection:
            return
        
        # Calculate outcomes
        perf_during = [p for p in self.performance_history 
                      if p['mode'] == CreativeMode.EXPLORING.value]
        
        if perf_during:
            avg_novelty = np.mean([p['novelty'] for p in perf_during])
            max_novelty = np.max([p['novelty'] for p in perf_during])
            avg_aesthetic = np.mean([p['aesthetic_score'] for p in perf_during])
            
            creative_gain = max_novelty - self.current_injection.trigger_novelty
            
            self.current_injection.outcomes = {
                'final_novelty': final_state.get('novelty_score', 0.0),
                'avg_novelty': float(avg_novelty),
                'max_novelty': float(max_novelty),
                'avg_aesthetic': float(avg_aesthetic),
                'creative_gain': float(creative_gain),
                'duration_actual': self.steps_in_mode
            }
            
            # Update metrics
            if creative_gain > 0.1:
                self.metrics['successful_explorations'] += 1
                self.metrics['total_creative_gain'] += creative_gain
            elif avg_aesthetic < 0.4:
                self.metrics['pruned_explorations'] += 1
        
        # Store in history
        self.injection_history.append(self.current_injection)
        self.current_injection = None
        
        # Transition to consolidation
        self.mode = CreativeMode.CONSOLIDATING
        self.steps_in_mode = 0
        
        logger.info("Exploration ended, entering consolidation phase")
    
    def get_creative_metrics(self) -> Dict[str, Any]:
        """Get comprehensive creative metrics"""
        recent_injections = list(self.injection_history)[-10:]
        
        metrics = self.metrics.copy()
        
        # Add recent statistics
        if recent_injections:
            recent_gains = [inj.outcomes.get('creative_gain', 0.0) 
                          for inj in recent_injections]
            metrics['recent_avg_gain'] = float(np.mean(recent_gains))
            metrics['recent_max_gain'] = float(np.max(recent_gains))
        
        # Success rate
        if metrics['total_injections'] > 0:
            metrics['success_rate'] = metrics['successful_explorations'] / metrics['total_injections']
        else:
            metrics['success_rate'] = 0.0
        
        # Current state
        metrics['current_mode'] = self.mode.value
        metrics['steps_in_mode'] = self.steps_in_mode
        
        return metrics
    
    def get_exploration_report(self) -> Dict[str, Any]:
        """Get detailed exploration report"""
        return {
            'metrics': self.get_creative_metrics(),
            'recent_injections': [inj.to_dict() for inj in list(self.injection_history)[-5:]],
            'performance_trend': self._analyze_performance_trend(),
            'regularizer_weights': self.regularizer.style_metrics,
            'safety_thresholds': {
                'novelty_high': self.novelty_threshold_high,
                'novelty_low': self.novelty_threshold_low,
                'max_lambda': self.max_lambda_allowed,
                'emergency': self.emergency_threshold
            }
        }
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze recent performance trends"""
        recent = list(self.performance_history)[-100:]
        
        if len(recent) < 10:
            return {'insufficient_data': True}
        
        # Extract time series
        novelties = [p['novelty'] for p in recent]
        aesthetics = [p['aesthetic_score'] for p in recent]
        
        # Compute trends
        novelty_trend = np.polyfit(range(len(novelties)), novelties, 1)[0]
        aesthetic_trend = np.polyfit(range(len(aesthetics)), aesthetics, 1)[0]
        
        return {
            'novelty_trend': float(novelty_trend),
            'aesthetic_trend': float(aesthetic_trend),
            'avg_novelty': float(np.mean(novelties)),
            'avg_aesthetic': float(np.mean(aesthetics)),
            'interpretation': self._interpret_trends(novelty_trend, aesthetic_trend)
        }
    
    def _interpret_trends(self, novelty_trend: float, aesthetic_trend: float) -> str:
        """Interpret performance trends"""
        if novelty_trend > 0.001 and aesthetic_trend > 0:
            return "healthy_creative_growth"
        elif novelty_trend > 0.001 and aesthetic_trend < -0.001:
            return "chaotic_exploration"
        elif novelty_trend < -0.001 and aesthetic_trend > 0:
            return "stabilizing_refinement"
        elif novelty_trend < -0.001 and aesthetic_trend < -0.001:
            return "stagnation_warning"
        else:
            return "stable_equilibrium"

# Global instance
_feedback_loop = None

def get_creative_feedback() -> CreativeSingularityFeedback:
    """Get or create global creative feedback loop"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = CreativeSingularityFeedback()
    return _feedback_loop

if __name__ == "__main__":
    # Test the creative feedback loop
    feedback = get_creative_feedback()
    
    # Simulate system evolution
    for i in range(200):
        # Generate synthetic state
        phase = i / 50  # 4 phases
        
        if phase < 1:  # Low novelty phase
            novelty = 0.1 + np.random.random() * 0.1
        elif phase < 2:  # Rising novelty
            novelty = 0.2 + (phase - 1) * 0.5 + np.random.random() * 0.1
        elif phase < 3:  # High novelty
            novelty = 0.7 + np.random.random() * 0.2
        else:  # Declining
            novelty = 0.9 - (phase - 3) * 0.5 + np.random.random() * 0.1
        
        state = {
            'novelty_score': novelty,
            'lambda_max': 0.02 + novelty * 0.05,
            'coherence_state': ['local', 'global', 'critical'][int(novelty * 3) % 3]
        }
        
        # Update feedback loop
        action = feedback.update(state)
        
        if action['action'] != 'none':
            print(f"Step {i}: {action['action']} (λ_factor={action['lambda_factor']:.2f})")
    
    # Get final report
    report = feedback.get_exploration_report()
    print("\nExploration Report:")
    print(json.dumps(report, indent=2))
