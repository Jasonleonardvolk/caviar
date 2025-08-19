"""fft_privacy.py - Implements privacy-enhanced memory processing for ALAN.

This module provides mechanisms for privacy-preserving memory updates based on frequency-domain
noise shaping, inspired by the FFTKF approach (Shin et al., 2025). It enables ALAN to:
- Shape differential privacy noise in the frequency domain
- Concentrate noise in high-frequency components
- Preserve low-frequency (long-term) information
- Apply Kalman filtering to denoise concept updates

References:
- Shin et al. (2025) for FFT-Enhanced Kalman Filter (FFTKF) approach
- Differential privacy for memory modularity and information hygiene
"""

import numpy as np
from scipy import fft, signal, linalg
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import math

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("alan_fft_privacy")

@dataclass
class PrivacyBudgetState:
    """Tracks the privacy budget usage across the system."""
    total_budget: float = 1.0  # Total epsilon privacy budget
    used_budget: float = 0.0  # Amount of budget used so far
    remaining_budget: float = 1.0  # Remaining budget
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)  # History of allocations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_budget": float(self.total_budget),
            "used_budget": float(self.used_budget),
            "remaining_budget": float(self.remaining_budget),
            "allocation_history": self.allocation_history
        }
        
    def allocate_budget(
        self, 
        amount: float, 
        purpose: str
    ) -> bool:
        """
        Allocate a portion of the privacy budget.
        
        Args:
            amount: Amount of budget to allocate
            purpose: Purpose of the allocation
            
        Returns:
            True if allocation successful, False if insufficient budget
        """
        if amount > self.remaining_budget:
            logger.warning(f"Privacy budget allocation failed: requested {amount}, "
                          f"but only {self.remaining_budget} remaining")
            return False
            
        self.used_budget += amount
        self.remaining_budget = self.total_budget - self.used_budget
        
        # Record allocation
        self.allocation_history.append({
            "timestamp": datetime.now().isoformat(),
            "amount": amount,
            "purpose": purpose,
            "remaining": self.remaining_budget
        })
        
        logger.debug(f"Privacy budget allocated: {amount} for {purpose}. "
                    f"Remaining: {self.remaining_budget}")
        
        return True
        
    def reset_budget(self, new_total: Optional[float] = None) -> None:
        """
        Reset the privacy budget.
        
        Args:
            new_total: Optional new total budget (if None, use previous total)
        """
        if new_total is not None:
            self.total_budget = new_total
            
        self.used_budget = 0.0
        self.remaining_budget = self.total_budget
        
        # Record reset
        self.allocation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "reset",
            "new_total": self.total_budget
        })
        
        logger.info(f"Privacy budget reset. New total: {self.total_budget}")


class FrequencyDomainNoiseShaper:
    """
    Shapes differential privacy noise in the frequency domain.
    
    Based on Shin et al. (2025), this class concentrates noise in high-frequency
    components while preserving low-frequency (long-term) information.
    """
    
    def __init__(
        self, 
        noise_scale: float = 1.0,
        frequency_mask_type: str = "linear",
        high_freq_noise_ratio: float = 3.0,
        preserve_dc: bool = True
    ):
        """
        Initialize the frequency domain noise shaper.
        
        Args:
            noise_scale: Base scale of the Gaussian noise
            frequency_mask_type: Type of frequency mask ("linear", "exponential", "step")
            high_freq_noise_ratio: Ratio of noise in high vs. low frequencies
            preserve_dc: Whether to preserve the DC component (mean)
        """
        self.noise_scale = noise_scale
        self.frequency_mask_type = frequency_mask_type
        self.high_freq_noise_ratio = high_freq_noise_ratio
        self.preserve_dc = preserve_dc
        
    def create_frequency_mask(
        self, 
        length: int
    ) -> np.ndarray:
        """
        Create a frequency domain mask for noise shaping.
        
        Args:
            length: Length of the signal
            
        Returns:
            Mask array with values between 0 and 1
        """
        # DC component is index 0, highest frequency is length//2
        mask = np.ones(length)
        
        if self.frequency_mask_type == "linear":
            # Linear scaling from low to high frequencies
            # First half of frequencies (excluding DC if needed)
            if self.preserve_dc:
                start_idx = 1  # Preserve DC component
            else:
                start_idx = 0
                
            # Create linear ramp-up for first half (low to mid frequencies)
            freqs = np.arange(start_idx, length // 2 + 1)
            if len(freqs) > 0:  # Check to avoid empty array
                mask[start_idx:length//2+1] = 1.0 + (self.high_freq_noise_ratio - 1.0) * (
                    freqs - start_idx) / (length // 2 - start_idx) if (length // 2 - start_idx) > 0 else 1.0
                
            # Create mask for second half (mirrored for conjugate frequencies)
            if length > 1:  # Check if there is a second half
                mask[length//2+1:] = mask[1:length//2][::-1] if start_idx == 1 else mask[0:length//2][::-1]
                
        elif self.frequency_mask_type == "exponential":
            # Exponential scaling from low to high frequencies
            if self.preserve_dc:
                mask[0] = 0  # Completely preserve DC component (no noise)
                
            # Create exponential ramp-up
            x = np.arange(length) / length
            mask = 1.0 + (self.high_freq_noise_ratio - 1.0) * np.exp(3 * x)
            
            # Normalize to ensure max value is high_freq_noise_ratio
            mask = mask / np.max(mask) * self.high_freq_noise_ratio
            
        elif self.frequency_mask_type == "step":
            # Step function: divide frequencies into low and high bands
            mid_point = length // 4  # First quarter is low frequencies
            
            if self.preserve_dc:
                mask[0] = 0  # No noise on DC
                
            # Low frequencies
            mask[1:mid_point] = 1.0
            
            # High frequencies
            mask[mid_point:] = self.high_freq_noise_ratio
            
        else:
            raise ValueError(f"Unknown frequency mask type: {self.frequency_mask_type}")
            
        return mask
        
    def apply_noise_with_frequency_shaping(
        self, 
        data: np.ndarray,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply frequency-shaped noise to achieve (ε,δ)-differential privacy.
        
        Args:
            data: Original data array
            epsilon: Privacy parameter ε
            delta: Privacy parameter δ
            sensitivity: Sensitivity of the data
            
        Returns:
            Tuple of (noised_data, metadata)
        """
        # Calculate noise scale from privacy parameters
        # For Gaussian mechanism, standard deviation = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Ensure data is 1D array
        original_shape = data.shape
        data = np.asarray(data).flatten()
        
        # Perform FFT
        fft_data = fft.rfft(data)
        
        # Create frequency mask for noise scaling
        freq_mask = self.create_frequency_mask(len(fft_data))
        
        # Create noise in frequency domain
        # Real and imaginary parts need separate noise
        fft_noise_real = np.random.normal(0, noise_scale, len(fft_data)) * freq_mask
        fft_noise_imag = np.random.normal(0, noise_scale, len(fft_data)) * freq_mask
        
        if self.preserve_dc:
            # Preserve DC component (mean of signal)
            fft_noise_real[0] = 0
            fft_noise_imag[0] = 0
            
        # Create complex noise
        fft_noise = fft_noise_real + 1j * fft_noise_imag
        
        # Apply noise in frequency domain
        fft_data_noised = fft_data + fft_noise
        
        # Inverse FFT to get back to time domain
        noised_data = fft.irfft(fft_data_noised, n=len(data))
        
        # Calculate noise metrics
        noise_power = np.sum(np.abs(fft_noise)**2) / len(fft_noise)
        signal_power = np.sum(np.abs(fft_data)**2) / len(fft_data)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Reshape to original shape if necessary
        if len(original_shape) > 1:
            noised_data = noised_data.reshape(original_shape)
            
        # Create metadata
        metadata = {
            "original_shape": original_shape,
            "epsilon": epsilon,
            "delta": delta,
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "frequency_mask_type": self.frequency_mask_type,
            "signal_power": float(signal_power),
            "noise_power": float(noise_power),
            "snr_db": float(snr_db),
            "high_freq_noise_ratio": self.high_freq_noise_ratio
        }
        
        return noised_data, metadata
        
    def analyze_noise_distribution(
        self, 
        original: np.ndarray,
        noised: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of added noise across frequencies.
        
        Args:
            original: Original data array
            noised: Noised data array
            
        Returns:
            Dictionary with noise analysis results
        """
        # Ensure data is 1D array
        original = np.asarray(original).flatten()
        noised = np.asarray(noised).flatten()
        
        if len(original) != len(noised):
            raise ValueError(f"Arrays must have same length, got {len(original)} and {len(noised)}")
            
        # Calculate noise
        noise = noised - original
        
        # Perform FFT on noise
        fft_noise = fft.rfft(noise)
        fft_original = fft.rfft(original)
        
        # Calculate magnitudes
        noise_mag = np.abs(fft_noise)
        signal_mag = np.abs(fft_original)
        
        # Calculate power spectrum
        noise_power_spectrum = noise_mag**2
        signal_power_spectrum = signal_mag**2
        
        # Calculate SNR per frequency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Ignore divide by zero warnings
            snr_per_freq = signal_power_spectrum / noise_power_spectrum
            
        # Replace inf/nan values
        snr_per_freq = np.nan_to_num(snr_per_freq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate average SNR in low and high frequency bands
        mid_point = len(noise_mag) // 2
        low_freq_snr = np.mean(snr_per_freq[1:mid_point]) if mid_point > 1 else 0
        high_freq_snr = np.mean(snr_per_freq[mid_point:]) if mid_point < len(snr_per_freq) else 0
        
        # Create frequency bands for analysis
        band_count = 4
        band_size = len(noise_mag) // band_count
        
        band_analysis = []
        for i in range(band_count):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < band_count - 1 else len(noise_mag)
            
            band_snr = np.mean(snr_per_freq[start_idx:end_idx])
            band_noise_power = np.sum(noise_power_spectrum[start_idx:end_idx])
            band_signal_power = np.sum(signal_power_spectrum[start_idx:end_idx])
            
            band_analysis.append({
                "band_index": i,
                "frequency_range": f"{start_idx/len(original):.2f}-{end_idx/len(original):.2f}",
                "noise_power": float(band_noise_power),
                "signal_power": float(band_signal_power),
                "snr": float(band_snr)
            })
            
        return {
            "total_noise_power": float(np.sum(noise_power_spectrum)),
            "total_signal_power": float(np.sum(signal_power_spectrum)),
            "overall_snr": float(np.sum(signal_power_spectrum) / np.sum(noise_power_spectrum)),
            "low_freq_snr": float(low_freq_snr),
            "high_freq_snr": float(high_freq_snr),
            "noise_to_signal_ratio": float(np.sum(noise_power_spectrum) / np.sum(signal_power_spectrum)),
            "band_analysis": band_analysis
        }


class KalmanMemoryFilter:
    """
    Applies Kalman filtering to denoise concept updates.
    
    Based on the FFTKF approach, this treats the true concept embedding as a hidden
    state and the noised embedding as an observation, using state-estimation to
    reduce variance while maintaining privacy.
    """
    
    def __init__(
        self, 
        state_dim: int,
        process_noise_scale: float = 0.01,
        measurement_noise_scale: float = 0.1,
        initial_estimate_uncertainty: float = 1.0
    ):
        """
        Initialize the Kalman memory filter.
        
        Args:
            state_dim: Dimension of the state vector (embedding dimension)
            process_noise_scale: Scale of process noise covariance
            measurement_noise_scale: Scale of measurement noise covariance
            initial_estimate_uncertainty: Initial uncertainty in state estimate
        """
        self.state_dim = state_dim
        
        # Initialize Kalman filter parameters
        # State transition matrix (identity as we assume constant state)
        self.F = np.eye(state_dim)
        
        # Measurement matrix (identity as we directly observe the state)
        self.H = np.eye(state_dim)
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * process_noise_scale
        
        # Measurement noise covariance
        self.base_R = np.eye(state_dim) * measurement_noise_scale
        self.R = self.base_R.copy()  # Current measurement noise covariance
        
        # Initial state estimate
        self.x = np.zeros(state_dim)
        
        # Initial estimate uncertainty
        self.P = np.eye(state_dim) * initial_estimate_uncertainty
        
        # Track filter state
        self.initialized = False
        self.update_count = 0
        self.last_update_time = None
        
    def initialize_with_state(self, initial_state: np.ndarray) -> None:
        """
        Initialize the filter with a specific state.
        
        Args:
            initial_state: Initial state vector
        """
        if len(initial_state) != self.state_dim:
            raise ValueError(f"Initial state dimension mismatch: expected {self.state_dim}, "
                           f"got {len(initial_state)}")
                           
        self.x = np.asarray(initial_state).flatten()
        self.initialized = True
        self.last_update_time = datetime.now()
        
        logger.debug(f"Kalman filter initialized with state of dimension {self.state_dim}")
        
    def update(
        self, 
        measurement: np.ndarray,
        measurement_noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform a Kalman filter update step.
        
        Args:
            measurement: Observed state vector (noisy embedding)
            measurement_noise: Optional custom measurement noise covariance
            
        Returns:
            Tuple of (filtered_state, metadata)
        """
        measurement = np.asarray(measurement).flatten()
        
        if len(measurement) != self.state_dim:
            raise ValueError(f"Measurement dimension mismatch: expected {self.state_dim}, "
                           f"got {len(measurement)}")
                           
        # First update - initialize state if not already done
        if not self.initialized:
            self.initialize_with_state(measurement)
            return measurement, {"status": "initialized"}
            
        # Update time tracking
        now = datetime.now()
        dt = (now - self.last_update_time).total_seconds() if self.last_update_time else 0
        self.last_update_time = now
        
        # Set measurement noise covariance
        if measurement_noise is not None:
            self.R = measurement_noise
        else:
            self.R = self.base_R.copy()
            
        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Kalman gain calculation
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update step
        self.x = x_pred + K @ (measurement - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred
        
        # Track update count
        self.update_count += 1
        
        # Calculate trace of uncertainty matrix (smaller is more certain)
        uncertainty = np.trace(self.P)
        
        # Estimate how much measurement influenced the update
        measurement_influence = np.mean(np.diag(K))
        
        # Create metadata
        metadata = {
            "status": "updated",
            "update_count": self.update_count,
            "time_since_last_update": dt,
            "uncertainty": float(uncertainty),
            "measurement_influence": float(measurement_influence),
            "innovation": float(np.linalg.norm(measurement - self.H @ x_pred))
        }
        
        return self.x, metadata
    
    def adjust_process_noise(self, scale_factor: float) -> None:
        """
        Adjust the process noise covariance.
        
        Args:
            scale_factor: Factor to scale the process noise by
        """
        self.Q = self.Q * scale_factor
        logger.debug(f"Process noise adjusted by factor {scale_factor}")
        
    def adjust_measurement_noise(self, scale_factor: float) -> None:
        """
        Adjust the measurement noise covariance.
        
        Args:
            scale_factor: Factor to scale the measurement noise by
        """
        self.base_R = self.base_R * scale_factor
        logger.debug(f"Measurement noise adjusted by factor {scale_factor}")
        
    def reset(self) -> None:
        """Reset the filter state."""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)
        self.initialized = False
        self.update_count = 0
        self.last_update_time = None
        
        logger.info("Kalman filter reset")


class PrivacyBudgetManager:
    """
    Manages the privacy budget allocation across different operations.
    
    This class tracks and allocates privacy budget (ε) to ensure the overall
    system satisfies differential privacy guarantees.
    """
    
    def __init__(
        self, 
        initial_budget: float = 1.0,
        min_allocation: float = 0.01,
        delta: float = 1e-5
    ):
        """
        Initialize the privacy budget manager.
        
        Args:
            initial_budget: Initial privacy budget (ε)
            min_allocation: Minimum budget allocation per operation
            delta: Privacy parameter δ
        """
        self.budget_state = PrivacyBudgetState(total_budget=initial_budget)
        self.min_allocation = min_allocation
        self.delta = delta
        self.default_sensitivity = 1.0
        
        # Track all budget-consuming operations
        self.operations = []
        
        logger.info(f"Privacy budget manager initialized with budget ε={initial_budget}")
        
    def allocate_budget(
        self, 
        operation_type: str,
        data_dimension: int,
        importance: float = 1.0
    ) -> Tuple[float, float]:
        """
        Allocate privacy budget for an operation.
        
        Args:
            operation_type: Type of operation requiring privacy budget
            data_dimension: Dimension of the data being processed
            importance: Importance factor (higher = more budget)
            
        Returns:
            Tuple of (epsilon, sensitivity) for the operation
        """
        remaining = self.budget_state.remaining_budget
        
        if remaining < self.min_allocation:
            logger.warning("Privacy budget exhausted, cannot allocate more.")
            return 0.0, 0.0
            
        # Calculate budget based on importance and data dimension
        # Larger dimensions get proportionally smaller budget per dimension
        dimension_factor = 1.0 / np.sqrt(data_dimension)
        
        # Calculate base allocation as a fraction of remaining budget
        base_allocation = min(
            remaining * 0.1,  # Don't use more than 10% at once
            self.min_allocation * importance * dimension_factor
        )
        
        # Ensure minimum allocation
        allocated_budget = max(base_allocation, self.min_allocation)
        
        # Don't exceed remaining budget
        allocated_budget = min(allocated_budget, remaining)
        
        # Perform allocation
        success = self.budget_state.allocate_budget(allocated_budget, operation_type)
        
        if not success:
            logger.error(f"Budget allocation failed for {operation_type}")
            return 0.0, 0.0
            
        # Calculate sensitivity based on operation type and dimension
        if operation_type == "embedding_update":
            # For embedding updates, sensitivity depends on the L2 norm bound
            sensitivity = min(2.0, self.default_sensitivity * np.sqrt(data_dimension))
        elif operation_type == "concept_merge":
            # For concept merging, sensitivity is higher
            sensitivity = min(4.0, self.default_sensitivity * 2 * np.sqrt(data_dimension))
        else:
            sensitivity = self.default_sensitivity
            
        # Record operation
        self.operations.append({
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "data_dimension": data_dimension,
            "allocated_budget": allocated_budget,
            "sensitivity": sensitivity,
            "importance": importance
        })
        
        logger.debug(f"Privacy budget allocated: ε={allocated_budget:.4f} for {operation_type}")
        
        return allocated_budget, sensitivity
        
    def reset_budget(self, new_budget: Optional[float] = None) -> None:
        """
        Reset the privacy budget.
        
        Args:
            new_budget: Optional new total budget (if None, use previous total)
        """
        self.budget_state.reset_budget(new_budget)
        
    def get_budget_state(self) -> Dict[str, Any]:
        """
        Get the current privacy budget state.
        
        Returns:
            Dictionary with budget state information
        """
        state = self.budget_state.to_dict()
        state["operation_count"] = len(self.operations)
        state["operations"] = self.operations[-10:]  # Last 10 operations
        
        return state


class FFTKFPrivacyEngine:
    """
    Main engine applying FFTKF privacy techniques to ALAN's concept embeddings.
    
    This class integrates frequency-domain noise shaping and Kalman filtering
    to enable privacy-preserving memory updates.
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        frequency_mask_type: str = "linear",
        high_freq_noise_ratio: float = 3.0
    ):
        """
        Initialize the FFTKF privacy engine.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            epsilon: Privacy parameter ε
            delta: Privacy parameter δ
            frequency_mask_type: Type of frequency mask for noise shaping
            high_freq_noise_ratio: Ratio of noise in high vs. low frequencies
        """
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.delta = delta
        
        # Initialize components
        self.noise_shaper = FrequencyDomainNoiseShaper(
            frequency_mask_type=frequency_mask_type,
            high_freq_noise_ratio=high_freq_noise_ratio,
            preserve_dc=True
        )
        
        self.kalman_filter = KalmanMemoryFilter(
            state_dim=embedding_dim,
            process_noise_scale=0.01,
            measurement_noise_scale=0.1
        )
        
        self.budget_manager = PrivacyBudgetManager(
            initial_budget=epsilon,
            delta=delta
        )
        
        # Flag whether to apply the full FFTKF pipeline
        self.apply_kalman = True
        
        logger.info(f"FFTKF privacy engine initialized with ε={epsilon}, δ={delta}")
        
    def privatize_embedding(
        self, 
        embedding: np.ndarray,
        operation_type: str = "embedding_update",
        importance: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply privacy-preserving mechanism to a concept embedding.
        
        Args:
            embedding: Original concept embedding
            operation_type: Type of operation for budget allocation
            importance: Importance factor for budget allocation
            
        Returns:
            Tuple of (privatized_embedding, metadata)
        """
        # Ensure embedding is a flattened array
        embedding = np.asarray(embedding).flatten()
        
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                           f"got {len(embedding)}")
                           
        # Allocate privacy budget
        allocated_epsilon, sensitivity = self.budget_manager.allocate_budget(
            operation_type=operation_type,
            data_dimension=self.embedding_dim,
            importance=importance
        )
        
        if allocated_epsilon <= 0:
            logger.warning("No privacy budget allocated, returning original embedding")
            return embedding, {"status": "no_budget", "privacy_applied": False}
            
        # Apply frequency-domain noise shaping
        noised_embedding, noise_metadata = self.noise_shaper.apply_noise_with_frequency_shaping(
            data=embedding,
            epsilon=allocated_epsilon,
            delta=self.delta,
            sensitivity=sensitivity
        )
        
        # Apply Kalman filtering if enabled
        if self.apply_kalman:
            # Create measurement noise covariance based on noise_metadata
            noise_scale = noise_metadata["noise_scale"]
            measurement_noise = np.eye(self.embedding_dim) * (noise_scale**2)
            
            # Apply Kalman filter
            filtered_embedding, kalman_metadata = self.kalman_filter.update(
                measurement=noised_embedding,
                measurement_noise=measurement_noise
            )
            
            # Use filtered embedding as result
            result_embedding = filtered_embedding
            kalman_applied = True
        else:
            # Skip Kalman filtering
            result_embedding = noised_embedding
            kalman_metadata = {"status": "skipped"}
            kalman_applied = False
            
        # Create combined metadata
        metadata = {
            "status": "success",
            "privacy_applied": True,
            "epsilon_allocated": allocated_epsilon,
            "operation_type": operation_type,
            "noise_metadata": noise_metadata,
            "kalman_metadata": kalman_metadata,
            "kalman_applied": kalman_applied,
            "original_norm": float(np.linalg.norm(embedding)),
            "result_norm": float(np.linalg.norm(result_embedding)),
            "remaining_budget": float(self.budget_manager.budget_state.remaining_budget)
        }
        
        return result_embedding, metadata
        
    def privatize_concept(
        self, 
        concept: ConceptTuple,
        importance: float = 1.0
    ) -> Tuple[ConceptTuple, Dict[str, Any]]:
        """
        Apply privacy-preserving mechanism to a concept.
        
        Args:
            concept: Original concept
            importance: Importance factor for budget allocation
            
        Returns:
            Tuple of (privatized_concept, metadata)
        """
        if not hasattr(concept, 'embedding') or concept.embedding is None:
            logger.warning("Concept has no embedding, cannot privatize")
            return concept, {"status": "no_embedding", "privacy_applied": False}
            
        # Privatize the embedding
        private_embedding, metadata = self.privatize_embedding(
            embedding=concept.embedding,
            operation_type="concept_update",
            importance=importance
        )
        
        # Create a copy of the concept with the privatized embedding
        # Use the same attributes as the original concept but with the privatized embedding
        private_concept = ConceptTuple(
            name=concept.name,
            embedding=private_embedding,
            context=concept.context,
            passage_embedding=concept.passage_embedding,
            cluster_members=concept.cluster_members,
            resonance_score=concept.resonance_score,
            narrative_centrality=concept.narrative_centrality,
            predictability_score=concept.predictability_score,
            eigenfunction_id=concept.eigenfunction_id,
            source_provenance=concept.source_provenance,
            spectral_lineage=concept.spectral_lineage,
            cluster_coherence=concept.cluster_coherence
        )
        
        # Add privacy information to concept metadata
        if not hasattr(private_concept, 'metadata'):
            private_concept.metadata = {}
            
        private_concept.metadata["privacy"] = {
            "applied": True,
            "epsilon": metadata.get("epsilon_allocated", 0),
            "kalman_filtered": metadata.get("kalman_applied", False),
            "timestamp": datetime.now().isoformat()
        }
        
        return private_concept, metadata
    
    def privatize_concept_batch(
        self, 
        concepts: List[ConceptTuple],
        importance: float = 1.0
    ) -> Tuple[List[ConceptTuple], Dict[str, Any]]:
        """
        Apply privacy-preserving mechanism to a batch of concepts.
        
        Args:
            concepts: List of concepts to privatize
            importance: Importance factor for budget allocation
            
        Returns:
            Tuple of (privatized_concepts, metadata)
        """
        if not concepts:
            return [], {"status": "no_concepts", "privacy_applied": False}
            
        # Process each concept
        privatized_concepts = []
        concept_metadata = []
        
        for concept in concepts:
            private_concept, metadata = self.privatize_concept(concept, importance)
            privatized_concepts.append(private_concept)
            concept_metadata.append(metadata)
            
        # Create batch metadata
        applied_count = sum(1 for m in concept_metadata if m.get("privacy_applied", False))
        
        batch_metadata = {
            "status": "success" if applied_count > 0 else "no_budget",
            "total_concepts": len(concepts),
            "privatized_count": applied_count,
            "remaining_budget": float(self.budget_manager.budget_state.remaining_budget),
            "individual_metadata": concept_metadata
        }
        
        return privatized_concepts, batch_metadata
    
    def privatize_memory_merge(
        self, 
        concept1: ConceptTuple,
        concept2: ConceptTuple,
        importance: float = 2.0  # Higher importance for merges
    ) -> Tuple[ConceptTuple, Dict[str, Any]]:
        """
        Apply privacy-preserving mechanism to a memory merge operation.
        
        Args:
            concept1, concept2: Concepts to merge
            importance: Importance factor for budget allocation
            
        Returns:
            Tuple of (merged_concept, metadata)
        """
        # Check that both concepts have embeddings
        if (not hasattr(concept1, 'embedding') or concept1.embedding is None or
            not hasattr(concept2, 'embedding') or concept2.embedding is None):
            logger.warning("One or both concepts missing embedding, cannot privatize merge")
            return concept1, {"status": "missing_embedding", "privacy_applied": False}
            
        # Simple average merge (in a real implementation, this would be more sophisticated)
        combined_embedding = (concept1.embedding + concept2.embedding) / 2
        
        # Privatize the merged embedding
        private_embedding, metadata = self.privatize_embedding(
            embedding=combined_embedding,
            operation_type="concept_merge",
            importance=importance
        )
        
        # Create a new concept for the merged result
        merged_name = f"{concept1.name} + {concept2.name}"
        
        # Combine other attributes
        cluster_members = []
        if hasattr(concept1, 'cluster_members') and concept1.cluster_members:
            cluster_members.extend(concept1.cluster_members)
        if hasattr(concept2, 'cluster_members') and concept2.cluster_members:
            cluster_members.extend(concept2.cluster_members)
            
        # Build merged source provenance
        source_provenance = []
        if hasattr(concept1, 'source_provenance') and concept1.source_provenance:
            source_provenance.extend(concept1.source_provenance)
        if hasattr(concept2, 'source_provenance') and concept2.source_provenance:
            source_provenance.extend(concept2.source_provenance)
            
        # Create the merged concept
        merged_concept = ConceptTuple(
            name=merged_name,
            embedding=private_embedding,
            cluster_members=cluster_members,
            source_provenance=source_provenance,
            eigenfunction_id=f"merged_{concept1.eigenfunction_id[:8]}_{concept2.eigenfunction_id[:8]}",
            spectral_lineage={
                "parents": [concept1.eigenfunction_id, concept2.eigenfunction_id],
                "merge_time": datetime.now().isoformat(),
                "privacy_applied": True
            }
        )
        
        return merged_concept, metadata
    
    def adjust_noise_level(
        self, 
        high_freq_ratio: Optional[float] = None,
        frequency_mask_type: Optional[str] = None
    ) -> None:
        """
        Adjust the noise level and shaping.
        
        Args:
            high_freq_ratio: New ratio for high frequency noise
            frequency_mask_type: New frequency mask type
        """
        if high_freq_ratio is not None:
            self.noise_shaper.high_freq_noise_ratio = high_freq_ratio
            logger.info(f"Noise ratio adjusted to {high_freq_ratio}")
            
        if frequency_mask_type is not None:
            self.noise_shaper.frequency_mask_type = frequency_mask_type
            logger.info(f"Frequency mask type set to {frequency_mask_type}")
    
    def adjust_kalman_parameters(
        self, 
        process_noise_scale: Optional[float] = None,
        measurement_noise_scale: Optional[float] = None
    ) -> None:
        """
        Adjust Kalman filter parameters.
        
        Args:
            process_noise_scale: New scale for process noise
            measurement_noise_scale: New scale for measurement noise
        """
        if process_noise_scale is not None:
            self.kalman_filter.adjust_process_noise(process_noise_scale)
            
        if measurement_noise_scale is not None:
            self.kalman_filter.adjust_measurement_noise(measurement_noise_scale)


# Singleton instance for easy access
_fft_privacy_engine = None

def get_fft_privacy_engine(
    embedding_dim: int = 768,  # Default for most embedding models
    epsilon: float = 1.0,
    delta: float = 1e-5
) -> FFTKFPrivacyEngine:
    """
    Get or create the singleton FFTKF privacy engine.
    
    Args:
        embedding_dim: Dimension of concept embeddings
        epsilon: Privacy parameter ε
        delta: Privacy parameter δ
        
    Returns:
        FFTKFPrivacyEngine instance
    """
    global _fft_privacy_engine
    if _fft_privacy_engine is None:
        _fft_privacy_engine = FFTKFPrivacyEngine(
            embedding_dim=embedding_dim,
            epsilon=epsilon,
            delta=delta
        )
    return _fft_privacy_engine
