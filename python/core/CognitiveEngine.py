"""
TORI/KHA Cognitive Engine - Production Implementation
Core cognitive processing with stability monitoring and eigenvalue analysis
"""

import numpy as np
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable, Protocol, Awaitable
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
import hashlib
import os
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingState(Enum):
    """Cognitive processing states"""
    IDLE = "idle"
    PROCESSING = "processing"
    STABILIZING = "stabilizing"
    ERROR = "error"
    HALTED = "halted"

@dataclass
class CognitiveState:
    """State representation for cognitive processing"""
    thought_vector: np.ndarray
    confidence: float
    stability_score: float
    coherence: float
    contradiction_level: float
    phase: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'thought_vector': self.thought_vector.tolist(),
            'confidence': self.confidence,
            'stability_score': self.stability_score,
            'coherence': self.coherence,
            'contradiction_level': self.contradiction_level,
            'phase': self.phase,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveState':
        """Create from dictionary"""
        data['thought_vector'] = np.array(data['thought_vector'])
        return cls(**data)

@dataclass
class ProcessingResult:
    """Result of cognitive processing"""
    success: bool
    output: Any
    state: CognitiveState
    trace: List[Dict[str, Any]]
    metrics: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class StabilityCallback(Protocol):
    """Protocol for stability callbacks"""
    def __call__(self, eigvals: np.ndarray, max_val: float) -> Awaitable[None]: ...

class CognitiveEngine:
    """
    Production-ready cognitive engine with full stability monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cognitive engine with configuration"""
        self.config = config or {}
        
        # Core parameters
        self.vector_dim = self.config.get('vector_dim', 512)
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.stability_threshold = self.config.get('stability_threshold', 0.1)
        self.coherence_threshold = self.config.get('coherence_threshold', 0.7)
        self.contradiction_threshold = self.config.get('contradiction_threshold', 0.3)
        self.convergence_epsilon = self.config.get('convergence_epsilon', 0.001)
        self.convergence_steps = self.config.get('convergence_steps', 5)
        
        # File-based storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/cognitive'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.current_state = self._initialize_state()
        self.state_history = deque(maxlen=self.config.get('history_size', 10000))
        self.processing_state = ProcessingState.IDLE
        
        # Stability monitoring
        self.eigenvalue_history = deque(maxlen=1000)
        self.stability_callbacks: List[StabilityCallback] = []
        
        # Processing components
        self.thought_matrix = self._initialize_thought_matrix()
        self.phase_transitions = self._load_phase_transitions()
        
        # Cache eigenvalues for performance
        self._cached_max_eigenvalue = self._compute_max_eigenvalue()
        self._matrix_modified = False
        
        # Async support
        self.processing_lock = asyncio.Lock()
        
        # Load saved state if exists
        self._load_checkpoint()
        
        logger.info(f"CognitiveEngine initialized with vector_dim={self.vector_dim}")
    
    def _initialize_state(self) -> CognitiveState:
        """Initialize default cognitive state"""
        return CognitiveState(
            thought_vector=np.zeros(self.vector_dim),
            confidence=1.0,
            stability_score=1.0,
            coherence=1.0,
            contradiction_level=0.0,
            phase="initialization",
            timestamp=time.time()
        )
    
    def _initialize_thought_matrix(self) -> np.ndarray:
        """Initialize thought transformation matrix"""
        # Create a stable transformation matrix with controlled eigenvalues
        # This ensures system stability from the start
        
        # Random orthogonal matrix (preserves norms)
        Q, _ = np.linalg.qr(np.random.randn(self.vector_dim, self.vector_dim))
        
        # Diagonal matrix with eigenvalues < 1 for stability
        eigenvalues = np.random.uniform(0.5, 0.95, self.vector_dim)
        D = np.diag(eigenvalues)
        
        # Construct matrix with known eigenvalues
        matrix = Q @ D @ Q.T
        
        # Add small noise for realism
        matrix += np.random.randn(self.vector_dim, self.vector_dim) * 0.01
        
        # Clip eigenspectrum to maintain stability guarantee
        u, s, vt = np.linalg.svd(matrix)
        s = np.minimum(s, 0.99)  # Ensure all eigenvalues < 1
        matrix = u @ np.diag(s) @ vt
        
        return matrix
    
    def _compute_max_eigenvalue(self) -> float:
        """Compute maximum eigenvalue of thought matrix"""
        eigenvalues = np.linalg.eigvals(self.thought_matrix)
        return float(np.max(np.abs(eigenvalues)))
    
    def _load_phase_transitions(self) -> Dict[str, Dict[str, float]]:
        """Load or create phase transition probabilities"""
        transitions_file = self.storage_path / "phase_transitions.json"
        
        if transitions_file.exists():
            with open(transitions_file, 'r') as f:
                return json.load(f)
        
        # Default phase transitions
        transitions = {
            "initialization": {"processing": 0.9, "error": 0.1},
            "processing": {"stabilizing": 0.7, "processing": 0.2, "error": 0.1},
            "stabilizing": {"processing": 0.3, "idle": 0.6, "error": 0.1},
            "idle": {"processing": 0.8, "idle": 0.2},
            "error": {"initialization": 0.5, "halted": 0.5}
        }
        
        # Save for future use
        with open(transitions_file, 'w') as f:
            json.dump(transitions, f, indent=2)
        
        return transitions
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Main cognitive processing method with stability monitoring
        """
        async with self.processing_lock:
            self.processing_state = ProcessingState.PROCESSING
            
            trace = []
            errors = []
            warnings = []
            start_time = time.time()
            
            try:
                # Input encoding
                encoded_input = await self._encode_input(input_data, context)
                trace.append({
                    'phase': 'encoding',
                    'timestamp': time.time(),
                    'input_shape': encoded_input.shape,
                    'input_norm': float(np.linalg.norm(encoded_input))
                })
                
                # Cognitive processing loop
                current_vector = encoded_input
                iteration = 0
                
                # Track convergence
                consecutive_stable_steps = 0
                last_change_rate = 1.0
                
                while iteration < self.max_iterations:
                    # Apply thought transformation - offloaded to executor
                    loop = asyncio.get_running_loop()
                    next_vector = await loop.run_in_executor(
                        None, self._apply_thought_transform_sync, current_vector, iteration
                    )
                    
                    # Check stability - offloaded to executor
                    stability_info = await loop.run_in_executor(
                        None, self._check_stability_sync, current_vector, next_vector
                    )
                    
                    # Log less frequently to reduce log volume
                    if iteration % 10 == 0 or iteration < 5:
                        logger.debug(f"Iteration {iteration}: stability={stability_info['stability_score']:.3f}, "
                                     f"change={stability_info['change_rate']:.3f}")
                    
                    trace.append({
                        'iteration': iteration,
                        'phase': self.current_state.phase,
                        'stability': stability_info,
                        'vector_norm': float(np.linalg.norm(next_vector)),
                        'timestamp': time.time()
                    })
                    
                    # Update state
                    self.current_state = CognitiveState(
                        thought_vector=next_vector,
                        confidence=stability_info['confidence'],
                        stability_score=stability_info['stability_score'],
                        coherence=stability_info['coherence'],
                        contradiction_level=stability_info['contradiction'],
                        phase=await self._determine_phase(stability_info),
                        timestamp=time.time(),
                        metadata={'iteration': iteration}
                    )
                    
                    # Add to history
                    self.state_history.append(self.current_state)
                    
                    # Adaptive convergence check
                    change_rate = stability_info['change_rate']
                    if abs(change_rate - last_change_rate) < self.convergence_epsilon:
                        consecutive_stable_steps += 1
                    else:
                        consecutive_stable_steps = 0
                    
                    last_change_rate = change_rate
                    
                    # Check convergence
                    if (consecutive_stable_steps >= self.convergence_steps) or \
                       (await self._check_convergence(current_vector, next_vector, stability_info)):
                        break
                    
                    # Check for instability
                    if stability_info['stability_score'] < self.stability_threshold:
                        warnings.append(f"Low stability at iteration {iteration}: {stability_info['stability_score']:.3f}")
                        
                        # Apply stabilization - offload to executor
                        stabilized_vector = await loop.run_in_executor(
                            None, self._apply_stabilization_sync, next_vector, stability_info
                        )
                        next_vector = stabilized_vector
                        self.processing_state = ProcessingState.STABILIZING
                    
                    current_vector = next_vector
                    iteration += 1
                    
                    # Prevent infinite loops
                    if iteration >= self.max_iterations:
                        warnings.append(f"Maximum iterations ({self.max_iterations}) reached")
                        break
                
                # Generate output
                output = await self._decode_output(current_vector, trace)
                
                # Calculate metrics
                metrics = {
                    'processing_time': time.time() - start_time,
                    'iterations': iteration,
                    'final_stability': self.current_state.stability_score,
                    'final_coherence': self.current_state.coherence,
                    'final_contradiction': self.current_state.contradiction_level,
                    'convergence_rate': self._calculate_convergence_rate(trace)
                }
                
                # Save checkpoint
                await self._save_checkpoint()
                
                self.processing_state = ProcessingState.IDLE
                
                return ProcessingResult(
                    success=True,
                    output=output,
                    state=self.current_state,
                    trace=trace,
                    metrics=metrics,
                    errors=errors,
                    warnings=warnings
                )
                
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
                self.processing_state = ProcessingState.ERROR
                errors.append(str(e))
                
                return ProcessingResult(
                    success=False,
                    output=None,
                    state=self.current_state,
                    trace=trace,
                    metrics={'processing_time': time.time() - start_time},
                    errors=errors,
                    warnings=warnings
                )
    
    async def _encode_input(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Encode input data into thought vector"""
        # Use executor for CPU-bound operations
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._encode_input_sync, input_data, context
        )
    
    def _encode_input_sync(self, input_data: Any, context: Optional[Dict[str, Any]]) -> np.ndarray:
        """Synchronous implementation of input encoding"""
        if isinstance(input_data, str):
            # Text encoding - replace with better semantic embedding
            from ingest_pdf.extraction.concept_extraction import extract_concepts_from_text
            
            try:
                # Try to use ConceptExtractor for semantic vectors
                concepts = extract_concepts_from_text(input_data)
                if concepts:
                    # Create a weighted average of concept vectors
                    vector = np.zeros(self.vector_dim)
                    for concept in concepts:
                        # Extract word vector or use a hash-based approach
                        concept_name = concept.name
                        concept_score = concept.score
                        
                        # Hash-based fallback if needed
                        concept_hash = hashlib.sha256(concept_name.encode()).digest()
                        hash_vector = np.frombuffer(concept_hash, dtype=np.float32)
                        
                        # Modulo addressing to fill the vector
                        for i in range(self.vector_dim):
                            vector[i] += hash_vector[i % len(hash_vector)] * concept_score
                    
                    # Add context if provided
                    if context:
                        context_str = json.dumps(context, sort_keys=True)
                        context_hash = hashlib.sha256(context_str.encode()).digest()
                        context_vector = np.frombuffer(context_hash, dtype=np.float32)
                        
                        # Modulo addressing for context
                        for i in range(self.vector_dim):
                            vector[i] += context_vector[i % len(context_vector)] * 0.5
                    
                    # Normalize
                    return vector / (np.linalg.norm(vector) + 1e-8)
            except Exception as e:
                logger.warning(f"Concept extraction failed, using fallback: {e}")
            
            # Fallback to improved hash-based embedding
            hash_obj = hashlib.sha256(input_data.encode())
            hash_bytes = hash_obj.digest()
            
            # Use modulo addressing instead of tiling
            vector = np.zeros(self.vector_dim, dtype=np.float32)
            for i in range(self.vector_dim):
                vector[i] = hash_bytes[i % len(hash_bytes)] / 255.0  # Normalize to [0,1]
            
            # Add context if provided
            if context:
                context_str = json.dumps(context, sort_keys=True)
                context_hash = hashlib.sha256(context_str.encode()).digest()
                context_vector = np.zeros(self.vector_dim, dtype=np.float32)
                for i in range(self.vector_dim):
                    context_vector[i] = context_hash[i % len(context_hash)] / 255.0
                vector = (vector + context_vector) / 2
            
            # Normalize
            return vector / (np.linalg.norm(vector) + 1e-8)
            
        elif isinstance(input_data, dict):
            # Dictionary encoding
            json_str = json.dumps(input_data, sort_keys=True)
            return self._encode_input_sync(json_str, context)
            
        elif isinstance(input_data, (list, np.ndarray)):
            # Array encoding
            array = np.asarray(input_data).flatten()
            
            if len(array) >= self.vector_dim:
                return array[:self.vector_dim] / (np.linalg.norm(array[:self.vector_dim]) + 1e-8)
            else:
                # Pad with zeros
                padded = np.zeros(self.vector_dim)
                padded[:len(array)] = array
                return padded / (np.linalg.norm(padded) + 1e-8)
        
        else:
            # Generic encoding
            return self._encode_input_sync(str(input_data), context)
    
    def _apply_thought_transform_sync(self, vector: np.ndarray, iteration: int) -> np.ndarray:
        """
        Synchronous implementation of thought transformation
        Can be safely called from an executor without blocking the event loop
        """
        # Linear transformation
        transformed = self.thought_matrix @ vector
        
        # Non-linear activation (tanh for bounded output)
        activated = np.tanh(transformed)
        
        # Residual connection for stability
        output = 0.7 * activated + 0.3 * vector
        
        # Monitor eigenvalues less frequently (every 10 iterations)
        # Use cached eigenvalues when possible to avoid recomputation
        if iteration % 10 == 0:
            if self._matrix_modified:
                max_eigenvalue = self._compute_max_eigenvalue()
                self._cached_max_eigenvalue = max_eigenvalue
                self._matrix_modified = False
                
                # Record history - but don't compute full eigenvalues again
                self.eigenvalue_history.append({
                    'timestamp': time.time(),
                    'max_eigenvalue': max_eigenvalue,
                    'eigenvalues': None  # Don't store full eigenvalues to save space
                })
            else:
                max_eigenvalue = self._cached_max_eigenvalue
        
        return output / (np.linalg.norm(output) + 1e-8)
    
    def _check_stability_sync(self, prev_vector: np.ndarray, curr_vector: np.ndarray) -> Dict[str, float]:
        """
        Synchronous implementation of stability checking
        Can be safely called from an executor without blocking the event loop
        """
        # Vector similarity (should be high for stability)
        similarity = np.dot(prev_vector, curr_vector)
        
        # Rate of change
        change_rate = np.linalg.norm(curr_vector - prev_vector)
        
        # Eigenvalue-based stability (using cached value)
        eigenvalue_stability = 1.0 / (1.0 + self._cached_max_eigenvalue)
        
        # Phase space analysis
        if len(self.state_history) >= 10:
            recent_states = list(self.state_history)[-10:]
            trajectories = [s.thought_vector for s in recent_states]
            
            # Calculate trajectory divergence
            divergence = np.std([np.linalg.norm(t - trajectories[-1]) for t in trajectories[:-1]])
            trajectory_stability = 1.0 / (1.0 + divergence)
        else:
            trajectory_stability = 1.0
        
        # Coherence (based on vector structure)
        coherence = 1.0 - np.std(curr_vector)
        
        # Contradiction detection (oscillation in signs)
        sign_changes = np.sum(np.diff(np.sign(curr_vector)) != 0)
        contradiction = sign_changes / len(curr_vector)
        
        # Combined stability score
        stability_score = (
            0.3 * similarity +
            0.2 * (1.0 - change_rate) +
            0.2 * eigenvalue_stability +
            0.2 * trajectory_stability +
            0.1 * coherence
        )
        
        # Confidence based on history
        confidence = min(1.0, len(self.state_history) / 100.0) * stability_score
        
        return {
            'stability_score': float(np.clip(stability_score, 0, 1)),
            'confidence': float(np.clip(confidence, 0, 1)),
            'coherence': float(np.clip(coherence, 0, 1)),
            'contradiction': float(np.clip(contradiction, 0, 1)),
            'change_rate': float(change_rate),
            'eigenvalue_stability': float(eigenvalue_stability),
            'trajectory_stability': float(trajectory_stability)
        }
    
    async def _determine_phase(self, stability_info: Dict[str, float]) -> str:
        """Determine cognitive phase based on stability metrics"""
        current_phase = self.current_state.phase
        
        # Get transition probabilities
        transitions = self.phase_transitions.get(current_phase, {})
        
        # Determine next phase based on stability
        if stability_info['stability_score'] > 0.8 and stability_info['change_rate'] < 0.1:
            # High stability, low change -> idle
            if 'idle' in transitions:
                return 'idle'
        elif stability_info['stability_score'] < 0.3:
            # Low stability -> error
            if 'error' in transitions:
                return 'error'
        elif stability_info['change_rate'] > 0.5:
            # High change rate -> processing
            if 'processing' in transitions:
                return 'processing'
        elif 0.3 <= stability_info['stability_score'] <= 0.7:
            # Medium stability -> stabilizing
            if 'stabilizing' in transitions:
                return 'stabilizing'
        
        # Default to current phase
        return current_phase
    
    async def _check_convergence(self, prev_vector: np.ndarray, curr_vector: np.ndarray, 
                                stability_info: Dict[str, float]) -> bool:
        """Check if processing has converged"""
        # Multiple convergence criteria
        
        # 1. Low change rate
        if stability_info['change_rate'] < 0.01:
            return True
        
        # 2. High stability for multiple iterations
        if len(self.state_history) >= 5:
            recent_stabilities = [s.stability_score for s in list(self.state_history)[-5:]]
            if all(s > 0.9 for s in recent_stabilities):
                return True
        
        # 3. Reached stable phase
        if self.current_state.phase == 'idle':
            return True
        
        # 4. Contradiction resolved
        if (stability_info['contradiction'] < self.contradiction_threshold and
            stability_info['coherence'] > self.coherence_threshold):
            return True
        
        return False
    
    def _apply_stabilization_sync(self, vector: np.ndarray, stability_info: Dict[str, float]) -> np.ndarray:
        """
        Synchronous implementation of stabilization
        Can be safely called from an executor without blocking the event loop
        """
        logger.warning(f"Applying stabilization: stability_score={stability_info['stability_score']:.3f}")
        
        # Multiple stabilization strategies
        
        # 1. Damping
        damping_factor = 0.5 + 0.5 * stability_info['stability_score']
        stabilized = vector * damping_factor
        
        # 2. Pull towards stable subspace
        if len(self.state_history) >= 10:
            # Find most stable historical state
            stable_states = sorted(
                list(self.state_history)[-50:],
                key=lambda s: s.stability_score,
                reverse=True
            )[:5]
            
            # Average of stable states
            stable_center = np.mean([s.thought_vector for s in stable_states], axis=0)
            
            # Blend with stable center
            blend_factor = 1.0 - stability_info['stability_score']
            stabilized = (1 - blend_factor) * stabilized + blend_factor * stable_center
        
        # 3. Reduce high-frequency components
        if stability_info['contradiction'] > 0.5:
            # Apply low-pass filter in frequency domain
            fft = np.fft.fft(stabilized)
            frequencies = np.fft.fftfreq(len(stabilized))
            
            # Suppress high frequencies
            cutoff = 0.3
            fft[np.abs(frequencies) > cutoff] *= 0.1
            
            stabilized = np.real(np.fft.ifft(fft))
        
        # 4. Eigenvalue correction
        if stability_info['eigenvalue_stability'] < 0.5:
            # Reduce influence of unstable eigendirections
            eigenvalues, eigenvectors = np.linalg.eig(self.thought_matrix)
            
            # Project onto stable eigenvectors
            stable_mask = np.abs(eigenvalues) < 0.9
            if np.any(stable_mask):
                stable_eigenvectors = eigenvectors[:, stable_mask]
                projection = stable_eigenvectors @ (stable_eigenvectors.T @ stabilized)
                stabilized = 0.7 * projection + 0.3 * stabilized
            
            # Matrix has been modified, recompute eigenvalues
            self._matrix_modified = True
        
        # Normalize
        return stabilized / (np.linalg.norm(stabilized) + 1e-8)
    
    async def _decode_output(self, final_vector: np.ndarray, trace: List[Dict[str, Any]]) -> Any:
        """Decode final thought vector into output"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._decode_output_sync, final_vector, trace)
    
    def _decode_output_sync(self, final_vector: np.ndarray, trace: List[Dict[str, Any]]) -> Any:
        """Synchronous implementation of output decoding"""
        output = {
            'summary': {
                'final_phase': self.current_state.phase,
                'stability': self.current_state.stability_score,
                'coherence': self.current_state.coherence,
                'confidence': self.current_state.confidence,
                'contradiction': self.current_state.contradiction_level
            },
            'vector_analysis': {
                'magnitude': float(np.linalg.norm(final_vector)),
                'mean': float(np.mean(final_vector)),
                'std': float(np.std(final_vector)),
                'dominant_components': self._get_dominant_components(final_vector)
            },
            'processing_trace': {
                'total_iterations': len(trace),
                'phase_transitions': self._extract_phase_transitions(trace),
                'stability_profile': self._extract_stability_profile(trace)
            },
            'interpretation': self._interpret_result(final_vector, trace)
        }
        
        return output
    
    def _get_dominant_components(self, vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get dominant components of vector"""
        indices = np.argsort(np.abs(vector))[-top_k:][::-1]
        
        return [
            {
                'index': int(idx),
                'value': float(vector[idx]),
                'magnitude': float(np.abs(vector[idx]))
            }
            for idx in indices
        ]
    
    def _extract_phase_transitions(self, trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract phase transitions from trace"""
        transitions = []
        prev_phase = None
        
        for entry in trace:
            if 'phase' in entry and entry['phase'] != prev_phase:
                transitions.append({
                    'iteration': entry.get('iteration', 0),
                    'from_phase': prev_phase,
                    'to_phase': entry['phase'],
                    'timestamp': entry['timestamp']
                })
                prev_phase = entry['phase']
        
        return transitions
    
    def _extract_stability_profile(self, trace: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract stability metrics over time"""
        profile = {
            'stability_scores': [],
            'coherence': [],
            'contradiction': [],
            'change_rates': []
        }
        
        for entry in trace:
            if 'stability' in entry:
                stability = entry['stability']
                profile['stability_scores'].append(stability.get('stability_score', 0))
                profile['coherence'].append(stability.get('coherence', 0))
                profile['contradiction'].append(stability.get('contradiction', 0))
                profile['change_rates'].append(stability.get('change_rate', 0))
        
        return profile
    
    def _interpret_result(self, final_vector: np.ndarray, trace: List[Dict[str, Any]]) -> str:
        """Generate human-readable interpretation"""
        stability = self.current_state.stability_score
        coherence = self.current_state.coherence
        contradiction = self.current_state.contradiction_level
        
        if stability > 0.8 and coherence > 0.8:
            interpretation = "Processing converged to a highly stable and coherent state."
        elif stability > 0.6:
            interpretation = "Processing reached a moderately stable state."
        elif contradiction > 0.5:
            interpretation = "Processing revealed significant contradictions in the input."
        elif stability < 0.3:
            interpretation = "Processing encountered instability - results may be unreliable."
        else:
            interpretation = "Processing completed with mixed stability indicators."
        
        # Add phase information
        interpretation += f" Final phase: {self.current_state.phase}."
        
        # Add convergence information
        if len(trace) < 10:
            interpretation += " Rapid convergence achieved."
        elif len(trace) > 50:
            interpretation += " Extended processing was required."
        
        return interpretation
    
    def _calculate_convergence_rate(self, trace: List[Dict[str, Any]]) -> float:
        """Calculate convergence rate from trace"""
        if len(trace) < 2:
            return 0.0
        
        # Extract change rates
        change_rates = []
        for entry in trace:
            if 'stability' in entry and 'change_rate' in entry['stability']:
                change_rates.append(entry['stability']['change_rate'])
        
        if len(change_rates) < 2:
            return 0.0
        
        # Calculate rate of decrease in change rate
        rates_array = np.array(change_rates)
        if len(rates_array) > 1:
            # Fit exponential decay
            x = np.arange(len(rates_array))
            log_rates = np.log(rates_array + 1e-10)
            
            # Linear regression on log scale
            slope, _ = np.polyfit(x, log_rates, 1)
            
            # Convergence rate is negative of slope
            return float(-slope)
        
        return 0.0
    
    async def _save_checkpoint(self):
        """Save current state to disk"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_checkpoint_sync)
    
    def _save_checkpoint_sync(self):
        """Synchronous implementation of checkpoint saving"""
        # Create compact checkpoint data
        matrix_hash = hashlib.sha256(self.thought_matrix.tobytes()).hexdigest()
        
        # Serialize the matrix to a compressed numpy file
        matrix_file = self.storage_path / f"thought_matrix_{matrix_hash[:8]}.npz"
        if not matrix_file.exists():
            np.savez_compressed(matrix_file, matrix=self.thought_matrix)
        
        checkpoint = {
            'current_state': self.current_state.to_dict(),
            'thought_matrix_hash': matrix_hash,
            'thought_matrix_file': matrix_file.name,
            'state_history': [s.to_dict() for s in list(self.state_history)[-100:]],
            'max_eigenvalue': self._cached_max_eigenvalue,
            'timestamp': time.time()
        }
        
        checkpoint_file = self.storage_path / "checkpoint.json"
        temp_file = checkpoint_file.with_suffix('.tmp')
        
        # Write to temp file first
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Atomic rename using os.replace for cross-device safety
        try:
            os.replace(str(temp_file), str(checkpoint_file))
        except OSError:
            # Fallback if cross-device move
            import shutil
            shutil.copy2(str(temp_file), str(checkpoint_file))
            os.unlink(str(temp_file))
        
        logger.debug(f"Checkpoint saved to {checkpoint_file}")
    
    def _load_checkpoint(self):
        """Load saved state from disk"""
        checkpoint_file = self.storage_path / "checkpoint.json"
        
        if not checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore state
            self.current_state = CognitiveState.from_dict(checkpoint['current_state'])
            
            # Load thought matrix from compressed file
            matrix_file = self.storage_path / checkpoint.get('thought_matrix_file', '')
            if matrix_file.exists():
                with np.load(matrix_file) as data:
                    self.thought_matrix = data['matrix']
            
            # Restore max eigenvalue
            self._cached_max_eigenvalue = checkpoint.get('max_eigenvalue', self._compute_max_eigenvalue())
            
            # Restore history
            for state_dict in checkpoint.get('state_history', []):
                self.state_history.append(CognitiveState.from_dict(state_dict))
            
            logger.info(f"Checkpoint loaded from {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def register_stability_callback(self, callback: StabilityCallback):
        """Register callback for stability monitoring"""
        self.stability_callbacks.append(callback)
    
    def get_current_stability(self) -> Dict[str, Any]:
        """Get current stability metrics"""
        return {
            'state': self.current_state.to_dict(),
            'max_eigenvalue': float(self._cached_max_eigenvalue),
            'is_stable': self._cached_max_eigenvalue < 1.0,
            'processing_state': self.processing_state.value,
            'history_size': len(self.state_history),
            'stability_score': self.current_state.stability_score
        }
    
    async def reset(self):
        """Reset engine to initial state"""
        self.current_state = self._initialize_state()
        self.state_history.clear()
        self.eigenvalue_history.clear()
        self.processing_state = ProcessingState.IDLE
        self.thought_matrix = self._initialize_thought_matrix()
        self._cached_max_eigenvalue = self._compute_max_eigenvalue()
        self._matrix_modified = False
        
        logger.info("CognitiveEngine reset to initial state")
    
    def shutdown(self):
        """Cleanup and shutdown"""
        # Save final checkpoint
        asyncio.create_task(self._save_checkpoint())
        
        logger.info("CognitiveEngine shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    async def test_cognitive_engine():
        """Test the cognitive engine"""
        config = {
            'vector_dim': 256,
            'max_iterations': 100,
            'storage_path': 'data/cognitive_test',
            'convergence_epsilon': 0.001,
            'convergence_steps': 3
        }
        
        engine = CognitiveEngine(config)
        
        # Test processing
        test_input = "What is the nature of consciousness?"
        context = {"domain": "philosophy", "depth": "deep"}
        
        print("Processing input:", test_input)
        result = await engine.process(test_input, context)
        
        print(f"\nSuccess: {result.success}")
        print(f"Final stability: {result.state.stability_score:.3f}")
        print(f"Iterations: {result.metrics['iterations']}")
        print(f"Processing time: {result.metrics['processing_time']:.2f}s")
        
        if result.output:
            print(f"\nInterpretation: {result.output['interpretation']}")
        
        # Check stability
        stability = engine.get_current_stability()
        print(f"\nCurrent stability: {stability['is_stable']}")
        print(f"Max eigenvalue: {stability['max_eigenvalue']:.3f}")
        
        engine.shutdown()
    
    # Run test
    asyncio.run(test_cognitive_engine())
