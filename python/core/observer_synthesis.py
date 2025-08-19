#!/usr/bin/env python3
"""
Observer-Synthesis - Metacognitive token generation
Implements observer-observed feedback loops
"""

import hashlib
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import deque
import logging

# Import from appropriate archive
try:
    from python.core.psi_archive import get_psi_archive
except ImportError:
    # Fallback if psi_archive not available
    class MockArchive:
        def log_event(self, event_type: str, data: Dict): pass
    
    def get_psi_archive():
        return MockArchive()

logger = logging.getLogger(__name__)

class ObserverSynthesis:
    """
    Manages observer-observed synthesis through metacognitive tokens
    Each measurement generates a token that becomes part of reasoning context
    """
    
    def __init__(self, max_tokens: int = 1000):
        """
        Initialize observer synthesis
        
        Args:
            max_tokens: Maximum tokens to retain in history
        """
        self.token_history = deque(maxlen=max_tokens)
        self.active_tokens = set()  # Currently in reasoning context
        self.token_metadata = {}    # token -> measurement data
        self.psi_archive = get_psi_archive()
        
        # Metrics
        self.metrics = {
            'tokens_generated': 0,
            'tokens_in_context': 0,
            'reflexive_loops': 0,
            'measurement_frequency': deque(maxlen=100)  # Track timing
        }
        
    def emit_token(self, measurement: Dict[str, Any]) -> str:
        """
        Generate metacognitive token from measurement
        
        Args:
            measurement: Dict containing measurement data
                        (e.g., lambda_max, betti numbers, curvature)
                        
        Returns:
            Generated token hash
        """
        # Add timestamp if not present
        if 'timestamp' not in measurement:
            measurement['timestamp'] = datetime.now(timezone.utc).isoformat()
            
        # Serialize deterministically
        blob = json.dumps(measurement, sort_keys=True).encode('utf-8')
        
        # Generate token using BLAKE2b (fast, secure)
        token = hashlib.blake2b(blob, digest_size=16).hexdigest()
        
        # Record token
        self.token_history.append(token)
        self.token_metadata[token] = measurement.copy()
        
        # Log to archive
        self.psi_archive.log_event("meta_token", {
            "t": time.time(),
            "token": token,
            "measurement_type": measurement.get('type', 'unknown'),
            "source": measurement.get('source', 'unknown')
        })
        
        # Update metrics
        self.metrics['tokens_generated'] += 1
        self.metrics['measurement_frequency'].append(time.time())
        
        logger.debug(f"Generated metacognitive token: {token[:8]}...")
        
        return token
    
    def add_to_context(self, token: str) -> bool:
        """
        Add token to active reasoning context
        
        Args:
            token: Token to activate
            
        Returns:
            True if successfully added
        """
        if token not in self.token_metadata:
            logger.warning(f"Unknown token: {token[:8]}...")
            return False
            
        self.active_tokens.add(token)
        self.metrics['tokens_in_context'] = len(self.active_tokens)
        
        # Check for reflexive loops
        if self._detect_reflexive_loop(token):
            self.metrics['reflexive_loops'] += 1
            logger.info(f"Reflexive loop detected with token {token[:8]}...")
            
        return True
    
    def remove_from_context(self, token: str):
        """Remove token from active context"""
        self.active_tokens.discard(token)
        self.metrics['tokens_in_context'] = len(self.active_tokens)
    
    def get_context_measurements(self) -> List[Dict[str, Any]]:
        """
        Get all measurements currently in context
        
        Returns:
            List of measurement dicts
        """
        return [
            self.token_metadata[token]
            for token in self.active_tokens
            if token in self.token_metadata
        ]
    
    def _detect_reflexive_loop(self, new_token: str) -> bool:
        """
        Detect if adding token creates reflexive measurement loop
        
        A reflexive loop occurs when:
        - System measures its own measurement
        - Measurement of measurement exceeds threshold
        """
        if len(self.active_tokens) < 2:
            return False
            
        new_measurement = self.token_metadata[new_token]
        
        # Check if this measurement references other tokens
        if 'references' in new_measurement:
            referenced = set(new_measurement['references'])
            if referenced & self.active_tokens:
                return True
                
        # Check for measurement-of-measurement pattern
        if new_measurement.get('type') == 'meta_measurement':
            return True
            
        return False
    
    def synthesize_context(self) -> Dict[str, Any]:
        """
        Synthesize current context into summary
        
        Returns:
            Context summary with key metrics
        """
        if not self.active_tokens:
            return {
                'empty': True,
                'summary': {},
                'token_count': 0
            }
            
        measurements = self.get_context_measurements()
        
        # Aggregate measurements by type
        by_type = {}
        for m in measurements:
            m_type = m.get('type', 'unknown')
            if m_type not in by_type:
                by_type[m_type] = []
            by_type[m_type].append(m)
            
        # Compute summary statistics
        summary = {}
        
        # Lambda max trajectory
        lambda_values = [
            m.get('lambda_max', 0.0) 
            for m in measurements 
            if 'lambda_max' in m
        ]
        if lambda_values:
            summary['lambda_trajectory'] = lambda_values[-5:]  # Last 5
            summary['lambda_trend'] = 'increasing' if len(lambda_values) > 1 and lambda_values[-1] > lambda_values[0] else 'stable'
            
        # Betti evolution
        betti_measurements = [
            m for m in measurements 
            if 'betti' in m or 'b0' in m
        ]
        if betti_measurements:
            summary['topology_changes'] = len(betti_measurements)
            
        # Curvature summary
        curvature_values = [
            m.get('mean_curvature', 0.0)
            for m in measurements
            if 'mean_curvature' in m
        ]
        if curvature_values:
            summary['curvature_max'] = max(curvature_values)
            
        return {
            'empty': False,
            'summary': summary,
            'token_count': len(self.active_tokens),
            'measurement_types': list(by_type.keys()),
            'reflexive': self.metrics['reflexive_loops'] > 0
        }
    
    def create_reasoning_prompt(self) -> str:
        """
        Create reasoning prompt that includes metacognitive context
        
        Returns:
            Formatted prompt string
        """
        synthesis = self.synthesize_context()
        
        if synthesis['empty']:
            return ""
            
        prompt_parts = ["[Metacognitive Context]"]
        
        summary = synthesis['summary']
        
        if 'lambda_trajectory' in summary:
            prompt_parts.append(
                f"Spectral evolution: {summary['lambda_trajectory']} ({summary['lambda_trend']})"
            )
            
        if 'topology_changes' in summary:
            prompt_parts.append(
                f"Topological transitions: {summary['topology_changes']}"
            )
            
        if 'curvature_max' in summary:
            prompt_parts.append(
                f"Peak curvature: {summary['curvature_max']:.3f}"
            )
            
        if synthesis['reflexive']:
            prompt_parts.append("Note: Reflexive measurement detected")
            
        return "\n".join(prompt_parts)
    
    def get_measurement_rate(self) -> float:
        """
        Get current measurement rate (measurements per second)
        
        Returns:
            Rate in Hz
        """
        if len(self.metrics['measurement_frequency']) < 2:
            return 0.0
            
        times = list(self.metrics['measurement_frequency'])
        duration = times[-1] - times[0]
        
        if duration > 0:
            return len(times) / duration
        return 0.0
    
    def should_throttle(self, max_rate: float = 10.0) -> bool:
        """
        Check if measurement rate should be throttled
        
        Args:
            max_rate: Maximum measurements per second
            
        Returns:
            True if throttling needed
        """
        current_rate = self.get_measurement_rate()
        return current_rate > max_rate
    
    def clear_context(self):
        """Clear all tokens from active context"""
        self.active_tokens.clear()
        self.metrics['tokens_in_context'] = 0
        logger.info("Cleared metacognitive context")

# Global instance
_observer_synthesis = None

def get_observer_synthesis() -> ObserverSynthesis:
    """Get or create global ObserverSynthesis instance"""
    global _observer_synthesis
    if _observer_synthesis is None:
        _observer_synthesis = ObserverSynthesis()
    return _observer_synthesis

# Convenience function for modules
def emit_token(measurement: Dict[str, Any]) -> str:
    """
    Quick helper to emit token from any module
    
    Usage:
        token = emit_token({
            "type": "spectral",
            "lambda_max": 0.042,
            "source": "eigensentry"
        })
    """
    return get_observer_synthesis().emit_token(measurement)

# Integration helpers for existing modules
def integrate_with_eigensentry(sentry_instance):
    """
    Patch EigenSentry to emit tokens
    
    Args:
        sentry_instance: CurvatureAwareGuard instance
    """
    original_check = sentry_instance.check_eigenvalues
    synthesis = get_observer_synthesis()
    
    def wrapped_check(eigenvalues, state):
        # Original check
        result = original_check(eigenvalues, state)
        
        # Emit token
        token = synthesis.emit_token({
            "type": "curvature",
            "source": "eigensentry",
            "lambda_max": result['max_eigenvalue'],
            "mean_curvature": result['curvature'],
            "threshold": result['threshold'],
            "damping_active": result['action'] != 'none'
        })
        
        # Add to context if significant
        if result['action'] != 'none' or result['max_eigenvalue'] > 0.8 * result['threshold']:
            synthesis.add_to_context(token)
            logger.info(f"Added EigenSentry token to context: {token[:8]}...")
            
        return result
        
    sentry_instance.check_eigenvalues = wrapped_check
    logger.info("EigenSentry integrated with observer synthesis")

def integrate_with_origin_sentry(origin_instance):
    """
    Patch OriginSentry to emit tokens
    
    Args:
        origin_instance: OriginSentry instance
    """
    original_classify = origin_instance.classify
    synthesis = get_observer_synthesis()
    
    def wrapped_classify(eigenvalues, betti_numbers=None):
        # Original classification
        result = original_classify(eigenvalues, betti_numbers)
        
        # Emit token
        token = synthesis.emit_token({
            "type": "spectral",
            "source": "origin_sentry",
            "lambda_max": result['lambda_max'],
            "coherence": result['coherence'],
            "novelty_score": result['novelty_score'],
            "dim_expansion": result['dim_expansion'],
            "betti": betti_numbers if betti_numbers else []
        })
        
        # Add to context if novel
        if result['novelty_score'] > 0.5 or result['dim_expansion']:
            synthesis.add_to_context(token)
            logger.info(f"Added OriginSentry token to context: {token[:8]}...")
            
        return result
        
    origin_instance.classify = wrapped_classify
    logger.info("OriginSentry integrated with observer synthesis")
