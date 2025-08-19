"""time_context.py - Implements central time management for ALAN's cognitive processes.

This module provides a unified TimeContext that acts as the shared temporal 
backbone for all of ALAN's cognitive processes, including:

1. Synchronized decay rates across memory systems
2. Consistent timebase for phase coherence measurements
3. Reference points for concept stability and activation
4. Temporal normalization for cross-process comparisons

By centralizing time management, this module ensures that all cognitive 
processes operate with a consistent temporal framework, which is essential
for phase-locked reasoning and memory management.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
import logging
import uuid

# Configure logger
logger = logging.getLogger("time_context")

class TimeContext:
    """
    Central time management for phase coherence and memory operations.
    
    TimeContext provides a unified temporal reference frame for all cognitive
    processes, ensuring consistent time measurements and synchronized decay
    across different subsystems.
    
    Attributes:
        creation_time: When this TimeContext was first initialized
        last_update: Most recent time the context was updated
        clock_rate: Scaling factor for time progression (default 1.0)
        time_steps: Counter for discrete update calls
        decay_reference_points: Dictionary of named temporal reference points
        process_rates: Dictionary of process-specific clock rates
        cognitive_cycle_count: Count of complete cognitive cycles
        uuid: Unique identifier for this time context instance
    """
    
    def __init__(self, 
                 base_clock_rate: float = 1.0, 
                 start_time: Optional[datetime] = None) -> None:
        """
        Initialize a new TimeContext instance.
        
        Args:
            base_clock_rate: Scaling factor for time progression (default 1.0)
            start_time: Optional specific datetime to use as start (default: now)
        """
        self.creation_time = start_time or datetime.now()
        self.last_update = self.creation_time
        self.clock_rate = base_clock_rate
        self.time_steps = 0
        self.decay_reference_points: Dict[str, datetime] = {}
        self.process_rates: Dict[str, float] = {}
        self.cognitive_cycle_count = 0
        self.uuid = str(uuid.uuid4())
        
        # Track temporal statistics
        self._last_10_deltas: List[float] = []
        
        # Initial log
        logger.info(f"TimeContext {self.uuid[:8]} initialized at {self.creation_time.isoformat()}")
    
    def update(self) -> float:
        """
        Update internal clock and return elapsed delta time.
        
        This method should be called at each major processing step to advance
        the internal TimeContext state and maintain temporal coherence.
        
        Returns:
            float: Delta time (in seconds) since last update, adjusted by clock rate
        """
        now = datetime.now()
        delta = (now - self.last_update).total_seconds() * self.clock_rate
        self.last_update = now
        self.time_steps += 1
        
        # Store recent deltas for statistics
        self._last_10_deltas.append(delta)
        if len(self._last_10_deltas) > 10:
            self._last_10_deltas.pop(0)
        
        return delta
    
    def complete_cognitive_cycle(self) -> None:
        """
        Mark the completion of a full cognitive cycle.
        
        This signals that ALAN has completed a full perceptual-conceptual cycle
        and should be called after each major integration phase.
        """
        self.cognitive_cycle_count += 1
        if self.cognitive_cycle_count % 10 == 0:
            avg_cycle_time = sum(self._last_10_deltas) / len(self._last_10_deltas) if self._last_10_deltas else 0
            logger.info(f"Completed {self.cognitive_cycle_count} cognitive cycles. "
                       f"Avg cycle time: {avg_cycle_time:.4f}s")
    
    def register_decay_process(self, process_name: str, rate_factor: float = 1.0) -> None:
        """
        Register a new decay process with current time as reference.
        
        Args:
            process_name: Unique identifier for the decay process
            rate_factor: Optional process-specific clock rate multiplier
        """
        self.decay_reference_points[process_name] = self.last_update
        self.process_rates[process_name] = rate_factor
        logger.debug(f"Registered decay process '{process_name}' with rate {rate_factor}")
    
    def reset_decay_reference(self, process_name: str) -> None:
        """
        Reset the reference point for a specific decay process to current time.
        
        Args:
            process_name: Name of the process to reset
        
        Raises:
            KeyError: If process_name is not registered
        """
        if process_name not in self.decay_reference_points:
            raise KeyError(f"Decay process '{process_name}' not registered")
            
        self.decay_reference_points[process_name] = self.last_update
        logger.debug(f"Reset decay reference for '{process_name}'")
    
    def get_decay_factor(self, process_name: str, half_life: float) -> float:
        """
        Calculate exponential decay factor based on process half-life.
        
        The decay factor follows the formula: factor = 2^(-elapsed/half_life)
        
        Args:
            process_name: Name of the decay process
            half_life: Time period (in seconds) after which value decays by half
            
        Returns:
            float: Decay factor between 0 and 1
            
        Note:
            If process is not registered, it will be registered automatically
            and return a decay factor of 1.0 (no decay yet)
        """
        if process_name not in self.decay_reference_points:
            self.register_decay_process(process_name)
            return 1.0
            
        reference = self.decay_reference_points[process_name]
        process_rate = self.process_rates.get(process_name, 1.0)
        elapsed = (self.last_update - reference).total_seconds() * self.clock_rate * process_rate
        
        # Apply exponential decay: 2^(-elapsed/half_life)
        decay_factor = 2 ** (-elapsed / half_life)
        
        # Ensure factor is between 0 and 1
        return max(0.0, min(1.0, decay_factor))
    
    def get_time_since(self, timestamp: float) -> float:
        """
        Calculate time elapsed since a specific timestamp.
        
        Args:
            timestamp: Unix timestamp to measure from
            
        Returns:
            float: Seconds elapsed since timestamp, adjusted by clock rate
        """
        now_ts = self.last_update.timestamp()
        elapsed = (now_ts - timestamp) * self.clock_rate
        return max(0.0, elapsed)
    
    def get_age_factor(self, timestamp: float, scale_factor: float = 1.0) -> float:
        """
        Calculate normalized age factor from 0.0 (new) to 1.0 (old).
        
        This is useful for age-weighted operations that need to treat
        concepts differently based on their temporal recency.
        
        Args:
            timestamp: Unix timestamp of item creation
            scale_factor: Controls how quickly items age (lower = slower aging)
            
        Returns:
            float: Age factor from 0.0 (new) to 1.0 (old)
        """
        elapsed = self.get_time_since(timestamp)
        # Sigmoid function to map time to 0-1 range
        # scale_factor controls the steepness
        age_factor = 1.0 / (1.0 + math.exp(-scale_factor * elapsed / 86400))  # 86400 = seconds in day
        return age_factor
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """
        Get statistical information about this TimeContext.
        
        Returns:
            Dict containing timing statistics and process information
        """
        runtime = (self.last_update - self.creation_time).total_seconds()
        avg_cycle_time = sum(self._last_10_deltas) / len(self._last_10_deltas) if self._last_10_deltas else 0
        
        return {
            "uuid": self.uuid,
            "creation_time": self.creation_time.isoformat(),
            "runtime_seconds": runtime,
            "time_steps": self.time_steps,
            "cognitive_cycles": self.cognitive_cycle_count,
            "clock_rate": self.clock_rate,
            "avg_cycle_time": avg_cycle_time,
            "active_processes": list(self.decay_reference_points.keys()),
            "process_rates": self.process_rates.copy()
        }

# For convenience, create a singleton instance
import math
default_time_context = TimeContext()
