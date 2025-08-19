"""fractality.py - Implements fractal dynamics analysis for ALAN's cognitive processes.

This module provides tools for measuring and analyzing the fractal-like properties 
of ALAN's cognitive time series, including:
- Hurst exponent calculation for long-range dependence
- Spectral slope analysis for 1/f scaling
- Recurrence quantification for detecting self-similar patterns

References:
- Grela et al. (2025) for methods combining space-filling curves and DFA
- Fractality as a signature of complex cognitive systems
"""

import numpy as np
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import statsmodels.tsa.stattools as ts
import warnings

# Configure logger
logger = logging.getLogger("alan_fractality")

@dataclass
class FractalState:
    """Represents the current fractal state of a cognitive process."""
    timestamp: datetime = field(default_factory=datetime.now)
    hurst_exponent: float = 0.5  # Hurst exponent (0.5=random, >0.5=trend, <0.5=antipersistent)
    spectral_slope: float = 0.0  # Slope of power spectrum (1/f^α)
    correlation_dimension: float = 0.0  # Correlation dimension
    recurrence_rate: float = 0.0  # Rate of recurrence in phase space
    multifractal_spectrum: Dict[str, Any] = field(default_factory=dict)  # Multifractal spectrum metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "hurst_exponent": float(self.hurst_exponent),
            "spectral_slope": float(self.spectral_slope),
            "correlation_dimension": float(self.correlation_dimension),
            "recurrence_rate": float(self.recurrence_rate),
            "multifractal_spectrum": self.multifractal_spectrum
        }


class HurstExponentCalculator:
    """
    Calculates the Hurst exponent to measure long-range dependence in time series.
    
    The Hurst exponent (H) quantifies the degree of self-similarity and long-range
    dependence in a time series:
    - H = 0.5: uncorrelated Brownian motion (random walk)
    - 0.5 < H < 1: persistent series with long-term positive autocorrelation
    - 0 < H < 0.5: anti-persistent series with negative autocorrelation
    
    Based on Grela et al. (2025) approach using DFA (Detrended Fluctuation Analysis).
    """
    
    def __init__(
        self, 
        min_segment_size: int = 10,
        max_segment_size: Optional[int] = None,
        segment_count: int = 10
    ):
        """
        Initialize the Hurst exponent calculator.
        
        Args:
            min_segment_size: Minimum segment size for DFA
            max_segment_size: Maximum segment size for DFA (if None, use 1/4 of series length)
            segment_count: Number of segment sizes to consider
        """
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size
        self.segment_count = segment_count
        self.history = []
    
    def calculate_dfa(
        self, 
        time_series: np.ndarray,
        order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform Detrended Fluctuation Analysis on a time series.
        
        Args:
            time_series: Input time series data
            order: Order of the polynomial fit for detrending
            
        Returns:
            Tuple of (segment_sizes, fluctuations, hurst_exponent)
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        N = len(time_series)
        
        # Determine segment sizes
        if self.max_segment_size is None:
            self.max_segment_size = N // 4
            
        # Generate logarithmically spaced segment sizes
        segment_sizes = np.unique(np.logspace(
            np.log10(self.min_segment_size),
            np.log10(self.max_segment_size),
            self.segment_count
        ).astype(int))
        
        # Cumulative sum of the time series (profile/trajectory)
        profile = np.cumsum(time_series - np.mean(time_series))
        
        # Calculate fluctuation for each segment size
        fluctuations = np.zeros(len(segment_sizes))
        
        for i, size in enumerate(segment_sizes):
            if size >= N:
                fluctuations[i] = np.nan
                continue
                
            # Number of segments
            num_segments = N // size
            
            # Reshape data to segments
            segments = np.reshape(profile[:num_segments*size], (num_segments, size))
            
            # Create time indices for polynomial fit
            x = np.arange(size)
            
            # Calculate local trend for each segment
            segment_fluctuations = np.zeros(num_segments)
            for j in range(num_segments):
                # Fit polynomial of specified order
                p = np.polyfit(x, segments[j], order)
                # Generate fitted trend
                trend = np.polyval(p, x)
                # Calculate RMS of detrended segment
                segment_fluctuations[j] = np.sqrt(np.mean((segments[j] - trend) ** 2))
                
            # Overall fluctuation is the mean across all segments
            fluctuations[i] = np.mean(segment_fluctuations)
            
        # Clean NaN values
        valid_indices = ~np.isnan(fluctuations)
        segment_sizes = segment_sizes[valid_indices]
        fluctuations = fluctuations[valid_indices]
        
        if len(segment_sizes) < 2:
            logger.warning("Not enough valid segment sizes for DFA")
            return np.array([]), np.array([]), 0.5
            
        # Log-log regression to estimate Hurst exponent
        log_sizes = np.log(segment_sizes)
        log_flucts = np.log(fluctuations)
        
        # Linear fit: log(F) = H * log(n) + c
        p = np.polyfit(log_sizes, log_flucts, 1)
        hurst_exponent = p[0]  # Slope is the Hurst exponent
        
        return segment_sizes, fluctuations, hurst_exponent
    
    def calculate_hurst(
        self, 
        time_series: np.ndarray,
        method: str = "dfa"
    ) -> Dict[str, Any]:
        """
        Calculate the Hurst exponent for a time series.
        
        Args:
            time_series: Input time series data
            method: Method to use ("dfa" or "rs" for rescaled range)
            
        Returns:
            Dictionary with Hurst exponent and analysis results
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        
        if len(time_series) < self.min_segment_size * 4:
            return {
                "status": "insufficient_data",
                "message": f"Time series too short: {len(time_series)} < {self.min_segment_size * 4}"
            }
            
        # Calculate Hurst exponent using specified method
        if method == "dfa":
            segment_sizes, fluctuations, hurst = self.calculate_dfa(time_series)
            
            # Create result dictionary
            result = {
                "status": "success",
                "hurst_exponent": hurst,
                "method": "dfa",
                "segment_sizes": segment_sizes.tolist(),
                "fluctuations": fluctuations.tolist(),
                "series_length": len(time_series)
            }
        elif method == "rs":
            # Rescaled range method (alternative to DFA)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hurst = ts.hurst(time_series)
                
            result = {
                "status": "success",
                "hurst_exponent": hurst,
                "method": "rs",
                "series_length": len(time_series)
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown method: {method}"
            }
            
        # Add interpretation
        result["interpretation"] = self._interpret_hurst(hurst)
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "hurst_exponent": hurst,
            "method": method,
            "series_length": len(time_series)
        })
        
        # Log result
        logger.debug(f"Calculated Hurst exponent: {hurst:.4f} (method: {method})")
        
        return result
    
    def _interpret_hurst(self, hurst: float) -> Dict[str, Any]:
        """
        Interpret the meaning of a Hurst exponent value.
        
        Args:
            hurst: Hurst exponent
            
        Returns:
            Dictionary with interpretation details
        """
        if hurst > 0.95:
            category = "super_persistent"
            description = "Extremely persistent, strong long-range correlations"
            cognitive_meaning = "Highly stable, potentially rigid thinking pattern"
        elif hurst > 0.65:
            category = "persistent"
            description = "Persistent, positive long-range correlations"
            cognitive_meaning = "Stable cognitive process with memory across time scales"
        elif hurst > 0.55:
            category = "weakly_persistent"
            description = "Weakly persistent, slight positive correlations"
            cognitive_meaning = "Mostly stable process with some randomness"
        elif hurst > 0.45:
            category = "random"
            description = "Approximately uncorrelated, random-walk like"
            cognitive_meaning = "Balanced cognitive process, neither rigid nor chaotic"
        elif hurst > 0.25:
            category = "anti_persistent"
            description = "Anti-persistent, negative long-range correlations"
            cognitive_meaning = "Rapidly changing process that tends to reverse"
        else:
            category = "strongly_anti_persistent"
            description = "Strongly anti-persistent, strong negative correlations"
            cognitive_meaning = "Highly unstable, chaotic cognitive pattern"
            
        return {
            "category": category,
            "description": description,
            "cognitive_meaning": cognitive_meaning
        }
    
    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze the trend in Hurst exponent over time.
        
        Args:
            window: Number of recent observations to consider
            
        Returns:
            Dictionary with trend information
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
            
        # Get recent values
        recent = self.history[-min(window, len(self.history)):]
        hursts = [entry["hurst_exponent"] for entry in recent]
        
        # Simple trend calculation
        start, end = hursts[0], hursts[-1]
        change = end - start
        
        # Determine trend direction
        if abs(change) < 0.05:
            trend = "stable"
        elif change > 0:
            trend = "increasing_persistence"
        else:
            trend = "decreasing_persistence"
            
        # Interpret trend
        if trend == "increasing_persistence":
            interpretation = "Cognitive dynamics are becoming more stable and predictable"
        elif trend == "decreasing_persistence":
            interpretation = "Cognitive dynamics are becoming more variable and less predictable"
        else:
            interpretation = "Cognitive dynamics are maintaining similar fractal structure"
            
        return {
            "status": "analyzed",
            "trend": trend,
            "start_value": start,
            "current_value": end,
            "change": change,
            "interpretation": interpretation,
            "data_points": len(hursts)
        }


class SpectralSlopeAnalyzer:
    """
    Analyzes the power spectral density slope to detect 1/f scaling.
    
    Many natural and cognitive processes exhibit power spectra that follow a 1/f^α 
    scaling pattern. The slope α characterizes the fractal properties:
    - α ≈ 0: white noise (uncorrelated)
    - α ≈ 1: pink noise (1/f noise, balanced complexity)
    - α ≈ 2: brown noise (random walk, integrated white noise)
    """
    
    def __init__(
        self, 
        min_freq: float = 0.0,
        max_freq: Optional[float] = None,
        window: str = "hann",
        nperseg: Optional[int] = None
    ):
        """
        Initialize the spectral slope analyzer.
        
        Args:
            min_freq: Minimum frequency to include in slope calculation
            max_freq: Maximum frequency to include in slope calculation
            window: Window function for spectral estimation
            nperseg: Length of each segment for Welch's method
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.window = window
        self.nperseg = nperseg
        self.history = []
    
    def calculate_psd(
        self, 
        time_series: np.ndarray,
        fs: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the power spectral density of a time series.
        
        Args:
            time_series: Input time series data
            fs: Sampling frequency
            
        Returns:
            Tuple of (frequencies, psd)
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        
        # Use Welch's method for robust spectral estimation
        nperseg = self.nperseg if self.nperseg else min(256, len(time_series) // 4)
        frequencies, psd = signal.welch(
            time_series, 
            fs=fs,
            window=self.window,
            nperseg=nperseg,
            detrend='constant',
            scaling='density',
            average='mean'
        )
        
        return frequencies, psd
    
    def calculate_spectral_slope(
        self, 
        time_series: np.ndarray,
        fs: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate the spectral slope (1/f^α scaling exponent).
        
        Args:
            time_series: Input time series data
            fs: Sampling frequency
            
        Returns:
            Dictionary with spectral analysis results
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        
        if len(time_series) < 32:
            return {
                "status": "insufficient_data",
                "message": f"Time series too short: {len(time_series)} < 32"
            }
            
        # Calculate power spectral density
        frequencies, psd = self.calculate_psd(time_series, fs)
        
        # Filter frequency range if specified
        if self.max_freq is None:
            self.max_freq = fs / 2  # Nyquist frequency
            
        mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
        if not np.any(mask):
            return {
                "status": "error",
                "message": f"No frequencies in range [{self.min_freq}, {self.max_freq}]"
            }
            
        freq_range = frequencies[mask]
        psd_range = psd[mask]
        
        # Avoid log(0) errors
        nonzero_mask = psd_range > 0
        freq_range = freq_range[nonzero_mask]
        psd_range = psd_range[nonzero_mask]
        
        if len(freq_range) < 2:
            return {
                "status": "error",
                "message": "Insufficient non-zero PSD values for slope calculation"
            }
            
        # Log-log regression to estimate spectral slope
        log_freq = np.log10(freq_range)
        log_psd = np.log10(psd_range)
        
        # Linear fit: log(PSD) = -α * log(f) + c
        p = np.polyfit(log_freq, log_psd, 1)
        slope = -p[0]  # Negative because we want the exponent in 1/f^α
        intercept = p[1]
        
        # Fit quality (R^2)
        psd_fit = 10 ** (p[1] - p[0] * log_freq)
        ss_total = np.sum((psd_range - np.mean(psd_range)) ** 2)
        ss_residual = np.sum((psd_range - psd_fit) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Create result dictionary
        result = {
            "status": "success",
            "spectral_slope": slope,
            "fit_intercept": intercept,
            "r_squared": r_squared,
            "frequencies": frequencies.tolist(),
            "psd": psd.tolist(),
            "fit_range": {
                "min_freq": freq_range[0],
                "max_freq": freq_range[-1],
                "n_points": len(freq_range)
            }
        }
        
        # Add interpretation
        result["interpretation"] = self._interpret_slope(slope)
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "spectral_slope": slope,
            "r_squared": r_squared,
            "series_length": len(time_series)
        })
        
        # Log result
        logger.debug(f"Calculated spectral slope: {slope:.4f} (R² = {r_squared:.4f})")
        
        return result
    
    def _interpret_slope(self, slope: float) -> Dict[str, Any]:
        """
        Interpret the meaning of a spectral slope value.
        
        Args:
            slope: Spectral slope (α in 1/f^α)
            
        Returns:
            Dictionary with interpretation details
        """
        if slope > 2.5:
            category = "very_smooth"
            noise_color = "black_or_beyond"
            description = "Extremely smooth, highly integrated process"
            cognitive_meaning = "Highly predictable, potentially over-rigid dynamics"
        elif slope > 1.5:
            category = "smooth"
            noise_color = "brown"
            description = "Smooth, integrated process (random walk-like)"
            cognitive_meaning = "Persistent cognitive state with smooth transitions"
        elif slope > 0.8:
            category = "balanced_complexity"
            noise_color = "pink"
            description = "1/f-like scaling, balanced complexity"
            cognitive_meaning = "Optimal cognitive dynamics with scale-free processing"
        elif slope > 0.3:
            category = "mild_complexity"
            noise_color = "pink_with_white"
            description = "Between pink and white noise"
            cognitive_meaning = "Mix of structured patterns and random exploration"
        elif slope > -0.3:
            category = "random"
            noise_color = "white"
            description = "Mostly uncorrelated, white noise-like"
            cognitive_meaning = "Highly random, unpredictable cognitive process"
        else:
            category = "oscillatory"
            noise_color = "blue"
            description = "Negative slope, oscillatory or anti-persistent"
            cognitive_meaning = "Rapidly changing cognitive state with possible periodicity"
            
        return {
            "category": category,
            "noise_color": noise_color,
            "description": description,
            "cognitive_meaning": cognitive_meaning
        }
    
    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze the trend in spectral slope over time.
        
        Args:
            window: Number of recent observations to consider
            
        Returns:
            Dictionary with trend information
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
            
        # Get recent values
        recent = self.history[-min(window, len(self.history)):]
        slopes = [entry["spectral_slope"] for entry in recent]
        
        # Simple trend calculation
        start, end = slopes[0], slopes[-1]
        change = end - start
        
        # Determine trend direction
        if abs(change) < 0.1:
            trend = "stable"
        elif change > 0:
            trend = "increasing_smoothness"
        else:
            trend = "decreasing_smoothness"
            
        # Interpret trend
        if trend == "increasing_smoothness":
            interpretation = "Cognitive dynamics are becoming smoother and more predictable"
        elif trend == "decreasing_smoothness":
            interpretation = "Cognitive dynamics are becoming more variable and less predictable"
        else:
            interpretation = "Cognitive dynamics are maintaining similar complexity structure"
            
        return {
            "status": "analyzed",
            "trend": trend,
            "start_value": start,
            "current_value": end,
            "change": change,
            "interpretation": interpretation,
            "data_points": len(slopes)
        }


class RecurrenceQuantifier:
    """
    Quantifies recurrent patterns in state space trajectories.
    
    Recurrence Quantification Analysis (RQA) detects repetitive patterns
    in dynamical systems, which can indicate self-similarity and fractal structure
    in cognitive processes.
    """
    
    def __init__(
        self, 
        embedding_dimension: int = 2,
        delay: int = 1,
        threshold: Optional[float] = None,
        threshold_method: str = "fixed"
    ):
        """
        Initialize the recurrence quantifier.
        
        Args:
            embedding_dimension: Dimension for state space reconstruction
            delay: Time delay for embedding
            threshold: Distance threshold for recurrence detection (if None, auto-calculate)
            threshold_method: Method to set threshold ("fixed", "percentile", "std")
        """
        self.embedding_dimension = embedding_dimension
        self.delay = delay
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.history = []
    
    def embed_time_series(
        self, 
        time_series: np.ndarray
    ) -> np.ndarray:
        """
        Perform time delay embedding of a time series.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Embedded state space trajectory
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        N = len(time_series)
        
        # Check if embedding is possible
        if N < self.embedding_dimension * self.delay:
            logger.warning(f"Time series too short for embedding: {N} < {self.embedding_dimension * self.delay}")
            return np.array([])
            
        # Determine the number of embedded points
        n_points = N - (self.embedding_dimension - 1) * self.delay
        
        # Initialize embedded trajectory
        trajectory = np.zeros((n_points, self.embedding_dimension))
        
        # Construct embedded vectors
        for i in range(self.embedding_dimension):
            trajectory[:, i] = time_series[i*self.delay:i*self.delay + n_points]
            
        return trajectory
    
    def compute_distance_matrix(
        self, 
        trajectory: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix between points in trajectory.
        
        Args:
            trajectory: Embedded state space trajectory
            
        Returns:
            Matrix of pairwise distances
        """
        n_points = trajectory.shape[0]
        distances = np.zeros((n_points, n_points))
        
        # Compute Euclidean distances between all pairs of points
        for i in range(n_points):
            for j in range(i, n_points):
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                distances[i, j] = distances[j, i] = dist
                
        return distances
    
    def compute_recurrence_matrix(
        self, 
        distances: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute recurrence matrix from distance matrix.
        
        Args:
            distances: Matrix of pairwise distances
            threshold: Distance threshold for recurrence (if None, use self.threshold)
            
        Returns:
            Binary recurrence matrix
        """
        # Determine threshold if not provided
        if threshold is None:
            threshold = self.threshold
            
        if threshold is None:
            # Auto-calculate threshold based on method
            if self.threshold_method == "percentile":
                # Use 10th percentile of distances
                threshold = np.percentile(distances, 10)
            elif self.threshold_method == "std":
                # Use mean + 0.1 * std of distances
                threshold = np.mean(distances) - 0.1 * np.std(distances)
            else:
                # Default: use 10% of max distance
                threshold = 0.1 * np.max(distances)
                
        # Create recurrence matrix (1 if distance <= threshold, 0 otherwise)
        recurrence = (distances <= threshold).astype(int)
        
        return recurrence, threshold
    
    def compute_rqa_metrics(
        self, 
        recurrence_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Recurrence Quantification Analysis metrics.
        
        Args:
            recurrence_matrix: Binary recurrence matrix
            
        Returns:
            Dictionary with RQA metrics
        """
        n = recurrence_matrix.shape[0]
        
        # Recurrence Rate (RR): percentage of recurrence points
        recurrence_rate = np.sum(recurrence_matrix) / (n * n)
        
        # Determinism (DET): percentage of recurrence points forming diagonal lines
        min_line_length = 2
        diag_lines = []
        
        # Count diagonal lines (exclude main diagonal)
        for i in range(1, n):
            diagonal = np.diag(recurrence_matrix, k=i)
            line_start = False
            line_length = 0
            
            for val in diagonal:
                if val == 1:
                    if not line_start:
                        line_start = True
                    line_length += 1
                else:
                    if line_start and line_length >= min_line_length:
                        diag_lines.append(line_length)
                    line_start = False
                    line_length = 0
                    
            # Check if the last line was also a valid line
            if line_start and line_length >= min_line_length:
                diag_lines.append(line_length)
                
        determinism = sum(diag_lines) / max(1, np.sum(recurrence_matrix))
        
        # Average diagonal line length
        average_line_length = np.mean(diag_lines) if diag_lines else 0
        
        # Laminarity (LAM): percentage of recurrence points forming vertical lines
        vert_lines = []
        min_vert_length = 2
        
        # Count vertical lines
        for j in range(n):
            line_start = False
            line_length = 0
            
            for i in range(n):
                if recurrence_matrix[i, j] == 1:
                    if not line_start:
                        line_start = True
                    line_length += 1
                else:
                    if line_start and line_length >= min_vert_length:
                        vert_lines.append(line_length)
                    line_start = False
                    line_length = 0
                    
            # Check if the last line was also a valid line
            if line_start and line_length >= min_vert_length:
                vert_lines.append(line_length)
                
        laminarity = sum(vert_lines) / max(1, np.sum(recurrence_matrix))
        
        # Entropy of diagonal line lengths
        if len(diag_lines) > 1:
            # Calculate histogram of line lengths
            hist, bins = np.histogram(diag_lines, bins=10)
            # Normalize histogram to get probabilities
            prob = hist / np.sum(hist)
            # Remove zeros to avoid log(0)
            prob = prob[prob > 0]
            # Calculate entropy
            entropy = -np.sum(prob * np.log2(prob))
        else:
            entropy = 0
            
        return {
            "recurrence_rate": recurrence_rate,
            "determinism": determinism,
            "average_line_length": average_line_length,
            "laminarity": laminarity,
            "entropy": entropy
        }
    
    def analyze_recurrence(
        self, 
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform recurrence quantification analysis on a time series.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Dictionary with RQA results
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        
        required_length = self.embedding_dimension * self.delay + 10
        if len(time_series) < required_length:
            return {
                "status": "insufficient_data",
                "message": f"Time series too short: {len(time_series)} < {required_length}"
            }
            
        # Perform time delay embedding
        trajectory = self.embed_time_series(time_series)
        if len(trajectory) == 0:
            return {
                "status": "error",
                "message": "Embedding failed"
            }
            
        # Compute distance matrix
        distances = self.compute_distance_matrix(trajectory)
        
        # Compute recurrence matrix
        recurrence_matrix, used_threshold = self.compute_recurrence_matrix(distances)
        
        # Compute RQA metrics
        metrics = self.compute_rqa_metrics(recurrence_matrix)
        
        # Create result dictionary
        result = {
            "status": "success",
            "embedding_dimension": self.embedding_dimension,
            "delay": self.delay,
            "threshold": used_threshold,
            "threshold_method": self.threshold_method,
            "metrics": metrics,
            "series_length": len(time_series),
            "embedded_points": trajectory.shape[0]
        }
        
        # Add interpretation
        result["interpretation"] = self._interpret_rqa(metrics)
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "embedding_dimension": self.embedding_dimension,
            "delay": self.delay,
            "threshold": used_threshold,
            "series_length": len(time_series)
        })
        
        # Log result
        logger.debug(f"Recurrence analysis: RR={metrics['recurrence_rate']:.4f}, "
                    f"DET={metrics['determinism']:.4f}")
        
        return result
    
    def _interpret_rqa(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Interpret the RQA metrics in cognitive terms.
        
        Args:
            metrics: Dictionary of RQA metrics
            
        Returns:
            Dictionary with interpretation details
        """
        # Extract metrics
        rr = metrics["recurrence_rate"]
        det = metrics["determinism"]
        lam = metrics["laminarity"]
        entr = metrics["entropy"]
        
        # Interpret overall dynamics
        if det > 0.8:
            dynamic_type = "highly_deterministic"
            description = "Highly structured, deterministic dynamics"
            cognitive_meaning = "Rigid, orderly cognitive process with strong patterns"
        elif det > 0.5:
            dynamic_type = "moderately_deterministic"
            description = "Mix of deterministic and random dynamics"
            cognitive_meaning = "Semi-structured cognitive process with some pattern flexibility"
        else:
            dynamic_type = "weakly_deterministic"
            description = "Mostly stochastic, weakly deterministic dynamics"
            cognitive_meaning = "Flexible, exploratory cognitive process with limited constraints"
            
        # Interpret state persistence (laminarity)
        if lam > 0.7:
            state_persistence = "high"
            state_description = "Strong tendency to remain in similar states"
        elif lam > 0.4:
            state_persistence = "moderate"
            state_description = "Some tendency to persist in states"
        else:
            state_persistence = "low"
            state_description = "Frequent state transitions, low persistence"
            
        # Interpret complexity (entropy)
        if entr > 2.0:
            complexity = "high"
            complexity_description = "High complexity in recurrence patterns"
        elif entr > 1.0:
            complexity = "moderate"
            complexity_description = "Moderate pattern complexity"
        else:
            complexity = "low"
            complexity_description = "Simple, low-complexity patterns"
            
        return {
            "dynamic_type": dynamic_type,
            "description": description,
            "cognitive_meaning": cognitive_meaning,
            "state_persistence": state_persistence,
            "state_description": state_description,
            "complexity": complexity,
            "complexity_description": complexity_description
        }
    
    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze the trend in recurrence metrics over time.
        
        Args:
            window: Number of recent observations to consider
            
        Returns:
            Dictionary with trend information
        """
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
            
        # Get recent values
        recent = self.history[-min(window, len(self.history)):]
        
        # Extract metrics over time
        timestamps = [entry["timestamp"] for entry in recent]
        rr_values = [entry["metrics"]["recurrence_rate"] for entry in recent]
        det_values = [entry["metrics"]["determinism"] for entry in recent]
        
        # Calculate trends
        if len(rr_values) >= 2:
            rr_change = rr_values[-1] - rr_values[0]
            det_change = det_values[-1] - det_values[0]
            
            # Determine trend direction
            if abs(det_change) < 0.1:
                trend = "stable"
            elif det_change > 0:
                trend = "increasing_determinism"
            else:
                trend = "decreasing_determinism"
                
            # Interpret trend
            if trend == "increasing_determinism":
                interpretation = "Cognitive dynamics are becoming more structured and patterned"
            elif trend == "decreasing_determinism":
                interpretation = "Cognitive dynamics are becoming more variable and less structured"
            else:
                interpretation = "Cognitive dynamics are maintaining similar structure"
                
            result = {
                "status": "analyzed",
                "trend": trend,
                "interpretation": interpretation,
                "metrics": {
                    "recurrence_rate": {
                        "start": rr_values[0],
                        "end": rr_values[-1],
                        "change": rr_change
                    },
                    "determinism": {
                        "start": det_values[0],
                        "end": det_values[-1],
                        "change": det_change
                    }
                },
                "data_points": len(recent)
            }
        else:
            result = {"status": "insufficient_data"}
            
        return result


class CognitiveFractalAnalyzer:
    """
    Main class integrating all fractal analysis components.
    
    This provides a unified interface for analyzing fractal properties
    of ALAN's cognitive processes, including long-range dependence,
    1/f scaling, and recurrence patterns.
    """
    
    def __init__(self, log_dir: str = "logs/fractality"):
        """
        Initialize the cognitive fractal analyzer.
        
        Args:
            log_dir: Directory for logging fractal analysis data
        """
        self.log_dir = log_dir
        
        # Initialize components
        self.hurst_calculator = HurstExponentCalculator()
        self.spectral_analyzer = SpectralSlopeAnalyzer()
        self.recurrence_analyzer = RecurrenceQuantifier(embedding_dimension=3)
        
        # Track fractal state history
        self.fractal_states = []
        
        logger.info("Cognitive fractal analyzer initialized")
        
    def analyze_time_series(
        self, 
        time_series: np.ndarray,
        series_name: str = "cognitive_series",
        fs: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fractal analysis on a time series.
        
        Args:
            time_series: Input time series data
            series_name: Name of the time series (for logging)
            fs: Sampling frequency
            
        Returns:
            Dictionary with analysis results
        """
        # Ensure the time series is 1D
        time_series = np.asarray(time_series).flatten()
        
        if len(time_series) < 32:
            return {
                "status": "insufficient_data",
                "message": f"Time series too short: {len(time_series)} < 32"
            }
            
        # Calculate Hurst exponent
        hurst_result = self.hurst_calculator.calculate_hurst(time_series)
        
        # Calculate spectral slope
        slope_result = self.spectral_analyzer.calculate_spectral_slope(time_series, fs)
        
        # Calculate recurrence metrics
        rqa_result = self.recurrence_analyzer.analyze_recurrence(time_series)
        
        # Combine results
        combined_result = {
            "status": "success",
            "series_name": series_name,
            "timestamp": datetime.now().isoformat(),
            "series_length": len(time_series),
            "sampling_frequency": fs,
            "hurst_exponent": hurst_result.get("hurst_exponent") if hurst_result.get("status") == "success" else None,
            "spectral_slope": slope_result.get("spectral_slope") if slope_result.get("status") == "success" else None,
            "recurrence_rate": rqa_result.get("metrics", {}).get("recurrence_rate") if rqa_result.get("status") == "success" else None,
            "determinism": rqa_result.get("metrics", {}).get("determinism") if rqa_result.get("status") == "success" else None,
            "detailed_results": {
                "hurst": hurst_result,
                "spectral": slope_result,
                "recurrence": rqa_result
            }
        }
        
        # Create fractal state
        state = FractalState(
            timestamp=datetime.now(),
            hurst_exponent=hurst_result.get("hurst_exponent", 0.5) if hurst_result.get("status") == "success" else 0.5,
            spectral_slope=slope_result.get("spectral_slope", 0.0) if slope_result.get("status") == "success" else 0.0,
            recurrence_rate=rqa_result.get("metrics", {}).get("recurrence_rate", 0.0) if rqa_result.get("status") == "success" else 0.0,
            correlation_dimension=0.0  # Not calculated here, placeholder
        )
        
        # Save state
        self.fractal_states.append(state)
        
        # Add summary interpretation
        combined_result["interpretation"] = self._create_summary_interpretation(state)
        
        # Log result
        logger.info(f"Fractal analysis of {series_name}: Hurst={state.hurst_exponent:.2f}, "
                   f"Slope={state.spectral_slope:.2f}")
        
        return combined_result
    
    def _create_summary_interpretation(self, state: FractalState) -> Dict[str, Any]:
        """
        Create a summary interpretation of the fractal state.
        
        Args:
            state: Current fractal state
            
        Returns:
            Dictionary with interpretation summary
        """
        # Interpret fractality type
        # Combine Hurst exponent and spectral slope for interpretation
        if state.hurst_exponent > 0.65 and state.spectral_slope > 1.5:
            fractality_type = "strongly_fractal"
            description = "Strong long-range correlations with smooth spectral properties"
            cognitive_interpretation = "Highly structured cognitive process with memory at multiple scales"
        elif state.hurst_exponent > 0.55 and state.spectral_slope > 0.8:
            fractality_type = "moderately_fractal"
            description = "Moderate long-range correlations with 1/f-like spectrum"
            cognitive_interpretation = "Balanced cognitive dynamics with healthy complexity"
        elif state.hurst_exponent > 0.45 and state.spectral_slope > 0.3:
            fractality_type = "weakly_fractal"
            description = "Weak long-range correlations with mild spectral slope"
            cognitive_interpretation = "Partially structured dynamics with some randomness"
        elif state.hurst_exponent < 0.45 and state.spectral_slope < 0.3:
            fractality_type = "anti_persistent"
            description = "Anti-persistent dynamics with flat spectrum"
            cognitive_interpretation = "Rapidly changing cognitive state with high unpredictability"
        else:
            fractality_type = "mixed"
            description = "Mixed fractal properties"
            cognitive_interpretation = "Complex cognitive dynamics with varying characteristics"
            
        # Overall complexity assessment
        if state.spectral_slope > 2.0:
            complexity = "low"
            complexity_description = "Very smooth, possibly over-structured"
        elif state.spectral_slope > 1.5:
            complexity = "moderate_low"
            complexity_description = "Smooth, structured but somewhat predictable"
        elif state.spectral_slope > 0.7:
            complexity = "optimal"
            complexity_description = "Balanced complexity, 1/f-like scaling"
        elif state.spectral_slope > 0.3:
            complexity = "moderate_high"
            complexity_description = "Higher complexity with some structure"
        else:
            complexity = "high"
            complexity_description = "Very high complexity, potentially chaotic"
            
        return {
            "fractality_type": fractality_type,
            "description": description,
            "cognitive_interpretation": cognitive_interpretation,
            "complexity": complexity,
            "complexity_description": complexity_description
        }
    
    def get_current_fractal_state(self) -> Dict[str, Any]:
        """
        Get the current fractal state of the system.
        
        Returns:
            Dictionary with current fractal state
        """
        if not self.fractal_states:
            return {
                "status": "no_data",
                "message": "No fractal states recorded yet"
            }
            
        # Get most recent state
        state = self.fractal_states[-1]
        
        # Get trends
        hurst_trend = self.hurst_calculator.get_trend()
        slope_trend = self.spectral_analyzer.get_trend()
        
        return {
            "status": "success",
            "timestamp": state.timestamp.isoformat(),
            "metrics": {
                "hurst_exponent": state.hurst_exponent,
                "spectral_slope": state.spectral_slope,
                "recurrence_rate": state.recurrence_rate
            },
            "trends": {
                "hurst": hurst_trend.get("trend", "unknown") if hurst_trend.get("status") == "analyzed" else "unknown",
                "spectral_slope": slope_trend.get("trend", "unknown") if slope_trend.get("status") == "analyzed" else "unknown"
            },
            "interpretation": self._create_summary_interpretation(state)
        }
    
    def summarize_fractal_trends(self, window: int = 10) -> Dict[str, Any]:
        """
        Generate a summary of recent fractal trends.
        
        Args:
            window: Number of recent states to consider
            
        Returns:
            Dictionary with trend summary
        """
        if len(self.fractal_states) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 fractal states, got {len(self.fractal_states)}"
            }
            
        # Get recent states
        recent = self.fractal_states[-min(window, len(self.fractal_states)):]
        
        # Calculate average metrics
        avg_hurst = sum(state.hurst_exponent for state in recent) / len(recent)
        avg_slope = sum(state.spectral_slope for state in recent) / len(recent)
        avg_recurrence = sum(state.recurrence_rate for state in recent) / len(recent)
        
        # Calculate trends
        hurst_trend = (recent[-1].hurst_exponent - recent[0].hurst_exponent)
        slope_trend = (recent[-1].spectral_slope - recent[0].spectral_slope)
        
        # Interpret combined trend
        if abs(hurst_trend) < 0.05 and abs(slope_trend) < 0.1:
            overall_trend = "stable"
            trend_description = "Stable fractal dynamics with consistent complexity"
        elif hurst_trend > 0.05 and slope_trend > 0.1:
            overall_trend = "increasing_structure"
            trend_description = "Increasing long-range order and smoother dynamics"
        elif hurst_trend < -0.05 and slope_trend < -0.1:
            overall_trend = "decreasing_structure"
            trend_description = "Decreasing long-range order, more random dynamics"
        elif hurst_trend > 0.05 and slope_trend < -0.1:
            overall_trend = "mixed_increasing_correlation"
            trend_description = "Increasing long-range correlation with higher complexity"
        elif hurst_trend < -0.05 and slope_trend > 0.1:
            overall_trend = "mixed_increasing_smoothness"
            trend_description = "Decreasing long-range correlation but smoother local dynamics"
        else:
            overall_trend = "mixed"
            trend_description = "Mixed trend with unclear direction"
            
        return {
            "status": "success",
            "window_size": len(recent),
            "time_span": (recent[-1].timestamp - recent[0].timestamp).total_seconds(),
            "average_metrics": {
                "hurst_exponent": avg_hurst,
                "spectral_slope": avg_slope,
                "recurrence_rate": avg_recurrence
            },
            "trends": {
                "hurst_exponent": hurst_trend,
                "spectral_slope": slope_trend
            },
            "overall_trend": overall_trend,
            "trend_description": trend_description
        }

# Singleton instance for easy access
_cognitive_fractal_analyzer = None

def get_cognitive_fractal_analyzer() -> CognitiveFractalAnalyzer:
    """
    Get or create the singleton fractal analyzer instance.
    
    Returns:
        CognitiveFractalAnalyzer instance
    """
    global _cognitive_fractal_analyzer
    if _cognitive_fractal_analyzer is None:
        _cognitive_fractal_analyzer = CognitiveFractalAnalyzer()
    return _cognitive_fractal_analyzer
