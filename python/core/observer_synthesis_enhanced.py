#!/usr/bin/env python3
"""
Observer-Observed Synthesis - Self-measurement and reflexive cognition.
ENHANCED VERSION with correctness and safety fixes.

The system observes its own spectral state and feeds it back into reasoning
through metacognitive tokens and reflexive patterns.
"""

import numpy as np
import hashlib
import json
import time
import threading
import warnings
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
import logging
from pathlib import Path
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)

# Safety constants
DEFAULT_REFLEX_BUDGET = 60  # Per hour
MEASUREMENT_COOLDOWN_MS = 100  # Minimum time between measurements
MAX_MEASUREMENT_HISTORY = 10000  # Cap for memory safety
MAX_REFLEX_WINDOW = 3600  # Maximum reflex window entries (1 per second for 1 hour)
MAX_EIGENVALUE_DIM = 1000  # Maximum eigenvalue array size
VALID_COHERENCE_STATES = {'local', 'global', 'critical'}
FORCE_MEASUREMENT_LIMIT = 10  # Max forced measurements per hour

# Performance constants
HASH_CACHE_SIZE = 1000  # Cache for recent spectral hashes


@dataclass
class SelfMeasurement:
    """Record of a self-measurement event."""
    timestamp: datetime
    spectral_hash: str
    eigenvalues: List[float]
    coherence_state: str
    novelty_score: float
    measurement_operator: str
    metacognitive_tokens: List[str]
    feedback_applied: bool = False
    error_state: Optional[str] = None  # Track measurement errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert measurement to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'spectral_hash': self.spectral_hash,
            'eigenvalues': self.eigenvalues,
            'coherence_state': self.coherence_state,
            'novelty_score': self.novelty_score,
            'measurement_operator': self.measurement_operator,
            'metacognitive_tokens': self.metacognitive_tokens,
            'feedback_applied': self.feedback_applied,
            'error_state': self.error_state
        }


class MeasurementError(Exception):
    """Custom exception for measurement errors."""
    pass


class RefexBudgetExhausted(Exception):
    """Raised when reflex budget is exhausted."""
    pass


class ObserverObservedSynthesis:
    """
    Implements self-measurement operators and reflexive feedback.
    
    Thread-safe implementation with reflex budget management and
    oscillation detection to prevent reflexive overload.
    
    Enhanced with comprehensive safety checks and error handling.
    """
    
    def __init__(self, reflex_budget: int = DEFAULT_REFLEX_BUDGET):
        # Validate initialization parameters
        if not isinstance(reflex_budget, int) or reflex_budget <= 0:
            raise ValueError("reflex_budget must be a positive integer")
            
        self.reflex_budget = min(reflex_budget, 3600)  # Cap at 1 per second
        self.measurements = deque(maxlen=1000)
        self.measurement_history = deque(maxlen=MAX_MEASUREMENT_HISTORY)
        
        # Thread safety with reentrant lock
        self._lock = threading.RLock()  # Use RLock for reentrant locking
        
        # Track reflex usage with bounded deque
        self.reflex_window = deque(maxlen=MAX_REFLEX_WINDOW)
        self.forced_measurements = deque(maxlen=FORCE_MEASUREMENT_LIMIT)
        self.last_measurement_time = time.monotonic() * 1000  # milliseconds
        
        # Measurement operators with error handling
        self.operators: Dict[str, Callable] = {
            'spectral_hash': self._spectral_hash_operator,
            'coherence_map': self._coherence_map_operator,
            'novelty_trace': self._novelty_trace_operator,
            'eigenmode_signature': self._eigenmode_signature_operator
        }
        
        # Metacognitive token vocabulary
        self.token_vocab = self._init_token_vocab()
        
        # Reflexive state with enhanced tracking
        self.reflexive_mode = False
        self.oscillation_detector = deque(maxlen=10)  # Track more history
        self.oscillation_count = 0
        self.last_oscillation_check = datetime.now(timezone.utc)
        
        # Performance tracking
        self._measurement_times = deque(maxlen=100)
        self._hash_cache = {}  # Limited size hash cache
        
        # Health metrics
        self.total_measurements = 0
        self.failed_measurements = 0
        self.last_error = None
        
        logger.info(f"Observer-Observed Synthesis initialized with budget: {self.reflex_budget}/hour")
    
    def _init_token_vocab(self) -> Dict[str, str]:
        """Initialize metacognitive token vocabulary."""
        return {
            # Spectral states
            'λ_low': 'STABLE_EIGEN',
            'λ_med': 'ACTIVE_EIGEN', 
            'λ_high': 'CRITICAL_EIGEN',
            'λ_critical': 'CRITICAL_LAMBDA',
            
            # Coherence states
            'local': 'LOCAL_COHERENCE',
            'global': 'GLOBAL_COHERENCE',
            'critical': 'CRITICAL_COHERENCE',
            
            # Dynamics
            'expanding': 'DIM_EXPANSION',
            'contracting': 'DIM_CONTRACTION',
            'oscillating': 'OSCILLATORY_MODE',
            'stable': 'STABLE_DYNAMICS',
            
            # Self-reference
            'observing': 'SELF_OBSERVE',
            'reflecting': 'META_REFLECT',
            'modifying': 'SELF_MODIFY',
            'monitoring': 'SELF_MONITOR',
            
            # Transitions
            'coherence_transition': 'COHERENCE_TRANSITION',
            'degenerate': 'DEGENERATE_MODES',
            'spectral_gap': 'SPECTRAL_GAP',
            
            # Error states
            'error': 'MEASUREMENT_ERROR',
            'overload': 'REFLEX_OVERLOAD',
            'unknown': 'UNKNOWN_TOKEN'
        }
    
    def _validate_inputs(self, 
                        eigenvalues: np.ndarray,
                        coherence_state: str,
                        novelty_score: float) -> None:
        """
        Validate measurement inputs with comprehensive checks.
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate eigenvalues
        if eigenvalues is None:
            raise ValueError("eigenvalues cannot be None")
        if not isinstance(eigenvalues, np.ndarray):
            raise ValueError("eigenvalues must be a numpy array")
        if eigenvalues.size == 0:
            raise ValueError("eigenvalues cannot be empty")
        if eigenvalues.size > MAX_EIGENVALUE_DIM:
            raise ValueError(f"eigenvalues size {eigenvalues.size} exceeds maximum {MAX_EIGENVALUE_DIM}")
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("eigenvalues must contain only finite values")
        if eigenvalues.dtype not in [np.float32, np.float64]:
            eigenvalues = eigenvalues.astype(np.float64)
            
        # Validate coherence state
        if not isinstance(coherence_state, str):
            raise ValueError("coherence_state must be a string")
        if coherence_state not in VALID_COHERENCE_STATES:
            raise ValueError(f"coherence_state must be one of {VALID_COHERENCE_STATES}")
            
        # Validate novelty score
        if not isinstance(novelty_score, (int, float)):
            raise ValueError("novelty_score must be numeric")
        if not np.isfinite(novelty_score):
            raise ValueError("novelty_score must be finite")
        if not 0 <= novelty_score <= 1:
            raise ValueError("novelty_score must be between 0 and 1")
    
    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except Exception as e:
            self.failed_measurements += 1
            self.last_error = {
                'operation': operation,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            logger.error(f"Error in {operation}: {str(e)}", exc_info=True)
            raise MeasurementError(f"Failed in {operation}: {str(e)}") from e
    
    def register_operator(self, name: str, operator_func: Callable) -> None:
        """
        Register a custom measurement operator with validation.
        
        Args:
            name: Operator name (alphanumeric + underscore only)
            operator_func: Function that takes (eigenvalues, coherence_state, novelty_score)
                          and returns SelfMeasurement
        """
        with self._lock:
            # Validate operator name
            if not name or not name.replace('_', '').isalnum():
                raise ValueError("Operator name must be alphanumeric with underscores")
            if len(name) > 50:
                raise ValueError("Operator name too long (max 50 chars)")
                
            # Validate operator function
            if not callable(operator_func):
                raise ValueError("operator_func must be callable")
                
            self.operators[name] = operator_func
            logger.info(f"Registered measurement operator: {name}")
    
    def measure(self, 
                eigenvalues: np.ndarray, 
                coherence_state: str,
                novelty_score: float,
                operator: str = 'spectral_hash',
                force: bool = False) -> Optional[SelfMeasurement]:
        """
        Perform self-measurement with comprehensive safety checks.
        
        Args:
            eigenvalues: Current eigenvalue spectrum
            coherence_state: Current coherence state
            novelty_score: Current novelty score (0-1)
            operator: Measurement operator to use
            force: Bypass budget checks (rate-limited)
            
        Returns:
            SelfMeasurement if performed, None if blocked
            
        Raises:
            MeasurementError: If measurement fails
            ValueError: If inputs are invalid
        """
        with self._lock:
            start_time = time.perf_counter()
            
            try:
                # Validate inputs first
                self._validate_inputs(eigenvalues, coherence_state, novelty_score)
                
                # Check cooldown
                now_ms = time.monotonic() * 1000
                if not force and (now_ms - self.last_measurement_time) < MEASUREMENT_COOLDOWN_MS:
                    logger.debug("Measurement blocked by cooldown")
                    return None
                
                # Check forced measurement limit
                if force:
                    self._check_forced_limit()
                
                # Check reflex budget
                if not force and not self._check_reflex_budget():
                    logger.debug("Reflex budget exhausted")
                    return None
                
                # Select and validate operator
                if operator not in self.operators:
                    logger.warning(f"Unknown operator {operator}, using spectral_hash")
                    operator = 'spectral_hash'
                
                measurement_func = self.operators[operator]
                
                # Perform measurement with error handling
                with self._error_handler(f"operator_{operator}"):
                    measurement = measurement_func(eigenvalues.copy(), 
                                                 coherence_state, 
                                                 novelty_score)
                
                # Validate measurement result
                if not isinstance(measurement, SelfMeasurement):
                    raise MeasurementError(f"Operator {operator} returned invalid type")
                
                # Update tracking
                self.measurements.append(measurement)
                self.measurement_history.append(measurement)
                self.last_measurement_time = now_ms
                self._update_reflex_window()
                self.total_measurements += 1
                
                # Track performance
                elapsed = time.perf_counter() - start_time
                self._measurement_times.append(elapsed)
                
                # Check for reflexive oscillation
                self._check_oscillation(measurement)
                
                logger.debug(f"Measurement completed in {elapsed*1000:.2f}ms: {measurement.spectral_hash[:8]}")
                
                return measurement
                
            except Exception as e:
                # Ensure we track the error
                if not isinstance(e, MeasurementError):
                    self.failed_measurements += 1
                    self.last_error = {
                        'operation': 'measure',
                        'error': str(e),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                raise
    
    def _check_forced_limit(self) -> None:
        """Check if forced measurements are within limit."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        
        # Clean old forced measurements
        while self.forced_measurements and self.forced_measurements[0] < cutoff:
            self.forced_measurements.popleft()
        
        if len(self.forced_measurements) >= FORCE_MEASUREMENT_LIMIT:
            raise RefexBudgetExhausted(
                f"Forced measurement limit ({FORCE_MEASUREMENT_LIMIT}/hour) exceeded"
            )
        
        self.forced_measurements.append(now)
    
    def _spectral_hash_operator(self, eigenvalues: np.ndarray,
                               coherence_state: str,
                               novelty_score: float) -> SelfMeasurement:
        """Basic spectral hash measurement with caching."""
        # Create cache key
        cache_key = (
            eigenvalues.tobytes(),
            coherence_state,
            round(novelty_score, 3)
        )
        
        # Check cache
        if cache_key in self._hash_cache:
            spectral_hash = self._hash_cache[cache_key]
        else:
            # Compute hash
            spectral_data = {
                'eigenvalues': np.round(eigenvalues, 6).tolist(),
                'coherence': coherence_state,
                'novelty': round(novelty_score, 3)
            }
            spectral_bytes = json.dumps(spectral_data, sort_keys=True).encode('utf-8')
            spectral_hash = hashlib.sha256(spectral_bytes).hexdigest()
            
            # Update cache with size limit
            if len(self._hash_cache) >= HASH_CACHE_SIZE:
                # Remove oldest entries
                for _ in range(HASH_CACHE_SIZE // 10):
                    self._hash_cache.pop(next(iter(self._hash_cache)))
            self._hash_cache[cache_key] = spectral_hash
        
        # Generate metacognitive tokens
        tokens = []
        
        # Eigenvalue magnitude token with safety
        if len(eigenvalues) > 0:
            lambda_max = np.max(np.abs(eigenvalues))
            if lambda_max < 0.01:
                tokens.append(self.token_vocab['λ_low'])
            elif lambda_max < 0.04:
                tokens.append(self.token_vocab['λ_med'])
            elif lambda_max < 0.08:
                tokens.append(self.token_vocab['λ_high'])
            else:
                tokens.append(self.token_vocab['λ_critical'])
        
        # Coherence token
        tokens.append(self.token_vocab.get(coherence_state, self.token_vocab['unknown']))
        
        # Add self-observation token
        tokens.append(self.token_vocab['observing'])
        
        return SelfMeasurement(
            timestamp=datetime.now(timezone.utc),
            spectral_hash=spectral_hash,
            eigenvalues=eigenvalues.tolist(),
            coherence_state=coherence_state,
            novelty_score=novelty_score,
            measurement_operator='spectral_hash',
            metacognitive_tokens=tokens
        )
    
    def _coherence_map_operator(self, eigenvalues: np.ndarray,
                               coherence_state: str,
                               novelty_score: float) -> SelfMeasurement:
        """Coherence-focused measurement."""
        measurement = self._spectral_hash_operator(eigenvalues, coherence_state, novelty_score)
        measurement.measurement_operator = 'coherence_map'
        
        # Add coherence transition detection
        if len(self.measurements) > 0:
            prev_coherence = self.measurements[-1].coherence_state
            if prev_coherence != coherence_state:
                measurement.metacognitive_tokens.append(self.token_vocab['coherence_transition'])
        
        return measurement
    
    def _novelty_trace_operator(self, eigenvalues: np.ndarray,
                              coherence_state: str,
                              novelty_score: float) -> SelfMeasurement:
        """Novelty-focused measurement."""
        measurement = self._spectral_hash_operator(eigenvalues, coherence_state, novelty_score)
        measurement.measurement_operator = 'novelty_trace'
        
        # Classify novelty dynamics with safety
        if len(self.measurements) > 0:
            prev_novelty = self.measurements[-1].novelty_score
            if prev_novelty > 0:  # Avoid division by zero
                ratio = novelty_score / prev_novelty
                if ratio > 1.2:
                    measurement.metacognitive_tokens.append(self.token_vocab['expanding'])
                elif ratio < 0.8:
                    measurement.metacognitive_tokens.append(self.token_vocab['contracting'])
                else:
                    measurement.metacognitive_tokens.append(self.token_vocab['stable'])
        
        return measurement
    
    def _eigenmode_signature_operator(self, eigenvalues: np.ndarray,
                                    coherence_state: str,
                                    novelty_score: float) -> SelfMeasurement:
        """Eigenmode pattern measurement."""
        measurement = self._spectral_hash_operator(eigenvalues, coherence_state, novelty_score)
        measurement.measurement_operator = 'eigenmode_signature'
        
        # Analyze eigenvalue distribution safely
        if len(eigenvalues) > 2:
            # Check for degenerate modes
            sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
            
            # Safe difference calculation
            if len(sorted_eigs) > 1:
                diffs = np.abs(np.diff(sorted_eigs))
                degeneracy = np.sum(diffs < 1e-6)
                
                if degeneracy > 0:
                    measurement.metacognitive_tokens.append(self.token_vocab['degenerate'])
                
                # Check for spectral gaps
                if np.max(diffs) > 0.02:
                    measurement.metacognitive_tokens.append(self.token_vocab['spectral_gap'])
        
        return measurement
    
    def generate_metacognitive_context(self, 
                                     recent_k: int = 5) -> Dict[str, Any]:
        """
        Generate metacognitive context from recent measurements.
        
        Thread-safe implementation with comprehensive error handling.
        
        Args:
            recent_k: Number of recent measurements to consider
            
        Returns:
            Dictionary containing metacognitive context
        """
        with self._lock:
            # Validate input
            recent_k = max(1, min(recent_k, 20))  # Reasonable bounds
            
            recent = list(self.measurements)[-recent_k:]
            
            if not recent:
                return {
                    'has_self_observations': False,
                    'metacognitive_tokens': [],
                    'token_set': set(),
                    'spectral_trajectory': [],
                    'health': self.get_health_status()
                }
            
            # Aggregate tokens safely
            all_tokens = []
            for m in recent:
                if m.metacognitive_tokens:  # Check for None/empty
                    all_tokens.extend(m.metacognitive_tokens)
            
            # Count token frequencies
            token_freq = {}
            for token in all_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            # Build spectral trajectory with error handling
            trajectory = []
            for m in recent:
                try:
                    trajectory.append({
                        'hash': m.spectral_hash[:8] if m.spectral_hash else 'unknown',
                        'coherence': m.coherence_state,
                        'novelty': round(m.novelty_score, 3),
                        'time': m.timestamp.isoformat(),
                        'operator': m.measurement_operator
                    })
                except Exception as e:
                    logger.error(f"Error building trajectory: {e}")
            
            # Detect patterns
            patterns = self._detect_reflexive_patterns(recent)
            
            context = {
                'has_self_observations': True,
                'metacognitive_tokens': all_tokens,
                'token_set': list(set(all_tokens)),
                'token_frequencies': token_freq,
                'spectral_trajectory': trajectory,
                'reflexive_patterns': patterns,
                'measurement_count': len(self.measurements),
                'reflex_budget_remaining': self._get_reflex_budget_remaining(),
                'reflexive_mode': self.reflexive_mode,
                'oscillation_count': self.oscillation_count,
                'health': self.get_health_status()
            }
            
            # Add warnings
            warnings = []
            if self.reflexive_mode:
                warnings.append('REFLEXIVE_OSCILLATION_DETECTED')
                context['metacognitive_tokens'].append(self.token_vocab['oscillating'])
            
            if self._get_reflex_budget_remaining() < 10:
                warnings.append('LOW_REFLEX_BUDGET')
                
            if self.failed_measurements > 10:
                warnings.append('HIGH_ERROR_RATE')
                
            if warnings:
                context['warnings'] = warnings
            
            return context
    
    def _detect_reflexive_patterns(self, 
                                  measurements: List[SelfMeasurement]) -> List[str]:
        """Detect patterns in self-measurements with enhanced safety."""
        patterns = []
        
        if len(measurements) < 2:
            return patterns
        
        try:
            # Check for rapid state changes
            state_changes = 0
            for i in range(1, len(measurements)):
                if measurements[i].coherence_state != measurements[i-1].coherence_state:
                    state_changes += 1
            
            if state_changes > len(measurements) / 2:
                patterns.append('RAPID_STATE_CHANGES')
            
            # Check for novelty trends safely
            novelties = [m.novelty_score for m in measurements]
            if len(set(novelties)) > 1:  # Non-constant values
                # Simple trend detection
                avg_first_half = np.mean(novelties[:len(novelties)//2])
                avg_second_half = np.mean(novelties[len(novelties)//2:])
                
                trend = avg_second_half - avg_first_half
                if trend > 0.1:
                    patterns.append('INCREASING_NOVELTY')
                elif trend < -0.1:
                    patterns.append('DECREASING_NOVELTY')
                else:
                    patterns.append('STABLE_NOVELTY')
            
            # Check for hash cycles
            hashes = [m.spectral_hash for m in measurements]
            unique_hashes = len(set(hashes))
            if unique_hashes < len(hashes) * 0.8:
                patterns.append('STATE_CYCLING')
            
            # Check measurement intervals
            if len(measurements) > 2:
                intervals = []
                for i in range(1, len(measurements)):
                    delta = measurements[i].timestamp - measurements[i-1].timestamp
                    intervals.append(delta.total_seconds())
                
                avg_interval = np.mean(intervals)
                if avg_interval < 1.0:  # Less than 1 second average
                    patterns.append('RAPID_MEASUREMENT')
                    
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            patterns.append('PATTERN_DETECTION_ERROR')
        
        return patterns
    
    def _check_oscillation(self, measurement: SelfMeasurement) -> None:
        """Enhanced oscillation detection."""
        self.oscillation_detector.append(measurement.spectral_hash)
        
        # Check if enough history
        if len(self.oscillation_detector) < 4:
            return
        
        hashes = list(self.oscillation_detector)
        
        # Check for various oscillation patterns
        # A-B-A-B pattern
        if (len(set(hashes[-4:])) == 2 and 
            hashes[-4] == hashes[-2] and 
            hashes[-3] == hashes[-1]):
            self.reflexive_mode = True
            self.oscillation_count += 1
            logger.warning(f"Reflexive oscillation detected! Count: {self.oscillation_count}")
            return
        
        # A-B-C-A-B-C pattern (3-cycle)
        if len(self.oscillation_detector) >= 6:
            recent_6 = hashes[-6:]
            if (recent_6[0] == recent_6[3] and 
                recent_6[1] == recent_6[4] and 
                recent_6[2] == recent_6[5]):
                self.reflexive_mode = True
                self.oscillation_count += 1
                logger.warning(f"3-cycle oscillation detected! Count: {self.oscillation_count}")
                return
        
        # Check if oscillation has stopped
        now = datetime.now(timezone.utc)
        if (self.reflexive_mode and 
            (now - self.last_oscillation_check).total_seconds() > 60):
            # If no oscillation for 60 seconds, reset
            unique_recent = len(set(hashes[-4:]))
            if unique_recent >= 3:  # Mostly unique recent hashes
                self.reflexive_mode = False
                logger.info("Reflexive oscillation resolved")
        
        self.last_oscillation_check = now
    
    def _check_reflex_budget(self) -> bool:
        """Check if reflex budget allows measurement."""
        remaining = self._get_reflex_budget_remaining()
        if remaining <= 0:
            raise RefexBudgetExhausted("Reflex budget exhausted")
        return True
    
    def _get_reflex_budget_remaining(self) -> int:
        """Get remaining reflex budget with proper cleanup."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        
        # Efficient cleanup using deque properties
        while self.reflex_window and self.reflex_window[0] < cutoff:
            self.reflex_window.popleft()
        
        return max(0, self.reflex_budget - len(self.reflex_window))
    
    def _update_reflex_window(self) -> None:
        """Update reflex usage window."""
        self.reflex_window.append(datetime.now(timezone.utc))
    
    def apply_stochastic_measurement(self, 
                                   eigenvalues: np.ndarray,
                                   coherence_state: str,
                                   novelty_score: float,
                                   base_probability: float = 0.1) -> Optional[SelfMeasurement]:
        """
        Apply stochastic self-measurement with safety limits.
        
        Args:
            eigenvalues: Current eigenvalue spectrum
            coherence_state: Current coherence state
            novelty_score: Current novelty score (0-1)
            base_probability: Base measurement probability (0-1)
            
        Returns:
            SelfMeasurement if performed, None otherwise
        """
        with self._lock:
            try:
                # Validate base probability
                base_probability = max(0.0, min(1.0, base_probability))
                
                # Calculate adjusted probability
                prob = base_probability * (1 + novelty_score)
                prob = min(1.0, prob)  # Cap at 1.0
                
                # Adjust for coherence transitions
                if len(self.measurements) > 0:
                    if self.measurements[-1].coherence_state != coherence_state:
                        prob = min(1.0, prob * 1.5)  # 50% boost for transitions
                
                # Reduce during oscillation
                if self.reflexive_mode:
                    prob *= 0.1
                
                # Reduce if budget is low
                budget_ratio = self._get_reflex_budget_remaining() / self.reflex_budget
                if budget_ratio < 0.2:
                    prob *= 0.5
                
                # Stochastic decision
                if np.random.random() < prob:
                    # Random operator selection
                    operator = np.random.choice(list(self.operators.keys()))
                    return self.measure(eigenvalues, coherence_state, 
                                      novelty_score, operator)
                
                return None
                
            except Exception as e:
                logger.error(f"Error in stochastic measurement: {e}")
                return None
    
    def get_measurement_history(self, 
                              window_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get measurement history within time window.
        
        Args:
            window_minutes: Time window in minutes (max 1440 = 24 hours)
            
        Returns:
            List of measurement dictionaries
        """
        with self._lock:
            # Bound window size
            window_minutes = max(1, min(window_minutes, 1440))
            
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            
            history = []
            for m in self.measurement_history:
                if m.timestamp > cutoff:
                    try:
                        history.append(m.to_dict())
                    except Exception as e:
                        logger.error(f"Error serializing measurement: {e}")
            
            return history
    
    def save_measurements(self, path: Path) -> None:
        """
        Save measurement history to file with error handling.
        
        Args:
            path: Path to save JSON file
            
        Raises:
            IOError: If save fails
        """
        with self._lock:
            try:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Prepare data with error handling
                measurements = []
                for m in self.measurement_history:
                    try:
                        measurements.append(m.to_dict())
                    except Exception as e:
                        logger.error(f"Error serializing measurement: {e}")
                
                data = {
                    'measurements': measurements,
                    'metadata': {
                        'reflex_budget': self.reflex_budget,
                        'total_measurements': self.total_measurements,
                        'failed_measurements': self.failed_measurements,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'version': '2.0'
                    }
                }
                
                # Write with atomic operation
                temp_path = path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Atomic rename
                temp_path.replace(path)
                
                logger.info(f"Saved {len(measurements)} measurements to {path}")
                
            except Exception as e:
                logger.error(f"Failed to save measurements: {e}")
                raise IOError(f"Failed to save measurements: {e}") from e
    
    def load_measurements(self, path: Path) -> None:
        """
        Load measurement history from file.
        
        Args:
            path: Path to JSON file
            
        Raises:
            IOError: If load fails
        """
        with self._lock:
            try:
                if not path.exists():
                    raise FileNotFoundError(f"Measurement file not found: {path}")
                
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Clear existing measurements
                self.measurement_history.clear()
                
                # Load measurements with validation
                measurements = data.get('measurements', [])
                for m_dict in measurements:
                    try:
                        # Reconstruct measurement
                        measurement = SelfMeasurement(
                            timestamp=datetime.fromisoformat(m_dict['timestamp']),
                            spectral_hash=m_dict['spectral_hash'],
                            eigenvalues=m_dict['eigenvalues'],
                            coherence_state=m_dict['coherence_state'],
                            novelty_score=m_dict['novelty_score'],
                            measurement_operator=m_dict['measurement_operator'],
                            metacognitive_tokens=m_dict['metacognitive_tokens'],
                            feedback_applied=m_dict.get('feedback_applied', False),
                            error_state=m_dict.get('error_state')
                        )
                        self.measurement_history.append(measurement)
                    except Exception as e:
                        logger.error(f"Error loading measurement: {e}")
                
                logger.info(f"Loaded {len(self.measurement_history)} measurements from {path}")
                
            except Exception as e:
                logger.error(f"Failed to load measurements: {e}")
                raise IOError(f"Failed to load measurements: {e}") from e
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the synthesis system.
        
        Returns:
            Dictionary with health metrics
        """
        with self._lock:
            # Calculate averages safely
            avg_measurement_time = 0.0
            if self._measurement_times:
                avg_measurement_time = np.mean(list(self._measurement_times)) * 1000  # ms
            
            error_rate = 0.0
            if self.total_measurements > 0:
                error_rate = self.failed_measurements / self.total_measurements
            
            return {
                'status': 'healthy' if error_rate < 0.1 else 'degraded',
                'total_measurements': self.total_measurements,
                'failed_measurements': self.failed_measurements,
                'error_rate': round(error_rate, 3),
                'reflex_budget_remaining': self._get_reflex_budget_remaining(),
                'reflex_budget_total': self.reflex_budget,
                'reflexive_mode': self.reflexive_mode,
                'oscillation_count': self.oscillation_count,
                'avg_measurement_time_ms': round(avg_measurement_time, 2),
                'measurement_history_size': len(self.measurement_history),
                'last_error': self.last_error
            }
    
    def reset_oscillation_state(self) -> None:
        """Reset oscillation detection state."""
        with self._lock:
            self.reflexive_mode = False
            self.oscillation_count = 0
            self.oscillation_detector.clear()
            logger.info("Oscillation state reset")


# Global instance with thread-safe initialization
_synthesis = None
_synthesis_lock = threading.Lock()


def get_observer_synthesis() -> ObserverObservedSynthesis:
    """Get or create global observer synthesis instance (thread-safe)."""
    global _synthesis
    
    # Double-checked locking pattern
    if _synthesis is None:
        with _synthesis_lock:
            if _synthesis is None:
                _synthesis = ObserverObservedSynthesis()
    
    return _synthesis


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the module.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('observer_synthesis.log', mode='a')
        ]
    )


if __name__ == "__main__":
    # Configure logging
    configure_logging(logging.DEBUG)
    
    # Test the enhanced observer-observed synthesis
    synthesis = get_observer_synthesis()
    
    # Test input validation
    try:
        synthesis.measure(None, 'local', 0.5)
    except ValueError as e:
        print(f"✓ Validation caught null input: {e}")
    
    try:
        synthesis.measure(np.array([]), 'local', 0.5)
    except ValueError as e:
        print(f"✓ Validation caught empty array: {e}")
    
    try:
        synthesis.measure(np.array([1, 2, 3]), 'invalid', 0.5)
    except ValueError as e:
        print(f"✓ Validation caught invalid coherence state: {e}")
    
    try:
        synthesis.measure(np.array([1, 2, 3]), 'local', 1.5)
    except ValueError as e:
        print(f"✓ Validation caught out-of-range novelty: {e}")
    
    # Register a custom operator
    def custom_operator(eigenvalues, coherence_state, novelty_score):
        """Example custom measurement operator."""
        measurement = synthesis._spectral_hash_operator(
            eigenvalues, coherence_state, novelty_score
        )
        measurement.measurement_operator = 'custom'
        measurement.metacognitive_tokens.append('CUSTOM_MEASUREMENT')
        return measurement
    
    synthesis.register_operator('custom', custom_operator)
    
    # Simulate measurements
    print("\n--- Testing measurements ---")
    for i in range(20):
        eigenvalues = np.random.randn(5) * 0.02
        coherence = ['local', 'global', 'critical'][i % 3]
        novelty = min(1.0, 0.1 + (i % 5) * 0.2)  # Ensure valid range
        
        # Try stochastic measurement
        measurement = synthesis.apply_stochastic_measurement(
            eigenvalues, coherence, novelty, base_probability=0.5
        )
        
        if measurement:
            print(f"Measurement {i}: {measurement.spectral_hash[:8]}, "
                  f"tokens: {measurement.metacognitive_tokens}")
        
        time.sleep(0.1)
    
    # Generate metacognitive context
    context = synthesis.generate_metacognitive_context()
    print("\nMetacognitive Context:")
    print(json.dumps(context, indent=2))
    
    # Get health status
    health = synthesis.get_health_status()
    print("\nHealth Status:")
    print(json.dumps(health, indent=2))
    
    # Test save/load
    save_path = Path("test_measurements.json")
    synthesis.save_measurements(save_path)
    print(f"\nSaved measurements to {save_path}")
    
    # Create new instance and load
    new_synthesis = ObserverObservedSynthesis()
    new_synthesis.load_measurements(save_path)
    print(f"Loaded {len(new_synthesis.measurement_history)} measurements")
    
    # Cleanup
    save_path.unlink()
