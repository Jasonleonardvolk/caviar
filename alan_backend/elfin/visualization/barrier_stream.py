"""
Barrier function data streaming for the safety dashboard.

This module provides real-time streaming of barrier function values and thresholds
for the safety dashboard.
"""

import asyncio
import json
import logging
import time
import random
import math
from typing import Dict, Any, AsyncGenerator, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BarrierStream:
    """
    Stream barrier function values and thresholds for a system.
    
    This class simulates and streams barrier function values and thresholds,
    which are used by the SafetyTimeline component in the dashboard.
    """
    
    def __init__(self, system_id: str, update_freq: float = 0.05):
        """
        Initialize barrier stream.
        
        Args:
            system_id: System ID
            update_freq: Update frequency in seconds (default: 50ms = 20Hz)
        """
        self.system_id = system_id
        self.update_freq = update_freq
        self.running = False
        self.current_time = 0.0
        
        # Barrier function parameters
        self.base_barrier_value = 0.5
        self.threshold_value = 0.2
        self.oscillation_freq = 0.2  # Hz
        self.barrier_noise = 0.05
        self.threshold_noise = 0.01
        
        # Instability parameters
        self.instability_prob = 0.005  # Probability of instability event per update
        self.current_instability = 0.0
        self.instability_duration = 2.0  # Duration of instability in seconds
        self.instability_strength = 0.4  # How much the barrier value drops
        self.recovering = False
        self.recovery_start = 0.0
        self.recovery_duration = 3.0  # Recovery duration in seconds
    
    async def generate_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a stream of barrier function values and thresholds.
        
        Yields:
            Dictionary with barrier function data
        """
        self.running = True
        self.current_time = 0.0
        
        while self.running:
            # Generate barrier and threshold values
            barrier_value, threshold_value = self._compute_barrier_values()
            
            # Create data packet
            data = {
                "t": self.current_time,
                "barrier": barrier_value,
                "thr": threshold_value
            }
            
            # Yield data as JSON
            yield json.dumps(data)
            
            # Update current time
            self.current_time += self.update_freq
            
            # Sleep until next update
            await asyncio.sleep(self.update_freq)
    
    def _compute_barrier_values(self) -> tuple[float, float]:
        """
        Compute barrier and threshold values.
        
        Returns:
            Tuple of (barrier_value, threshold_value)
        """
        # Base oscillating barrier value
        barrier = self.base_barrier_value + 0.2 * math.sin(2 * math.pi * self.oscillation_freq * self.current_time)
        
        # Add random noise
        barrier += random.uniform(-self.barrier_noise, self.barrier_noise)
        
        # Apply instability if active
        if self.current_instability > 0.0:
            barrier -= self.instability_strength * self.current_instability
            
            # Decrease instability factor
            self.current_instability -= self.update_freq / self.instability_duration
            self.current_instability = max(0.0, self.current_instability)
        
        # Check for recovery phase
        if self.recovering:
            recovery_progress = (self.current_time - self.recovery_start) / self.recovery_duration
            
            if recovery_progress >= 1.0:
                # Recovery complete
                self.recovering = False
            else:
                # During recovery, barrier gradually increases
                recovery_factor = math.erf(3 * recovery_progress)  # Sigmoid-like curve
                barrier = self.threshold_value + recovery_factor * (self.base_barrier_value - self.threshold_value)
        
        # Random chance of instability event
        elif random.random() < self.instability_prob and self.current_instability == 0.0 and not self.recovering:
            # Start instability
            self.current_instability = 1.0
            
            # Set recovery to start after instability ends
            self.recovering = True
            self.recovery_start = self.current_time + self.instability_duration * 0.8  # Overlap slightly
        
        # Compute threshold with small random variation
        threshold = self.threshold_value + random.uniform(-self.threshold_noise, self.threshold_noise)
        
        return barrier, threshold
    
    def stop(self) -> None:
        """Stop the stream."""
        self.running = False


class QuadrotorBarrierStream(BarrierStream):
    """
    Specialized barrier stream for quadrotor systems.
    
    Simulates more realistic barrier function behavior for a quadrotor,
    including altitude limits and obstacle avoidance.
    """
    
    def __init__(self, system_id: str, update_freq: float = 0.05):
        """
        Initialize quadrotor barrier stream.
        
        Args:
            system_id: System ID
            update_freq: Update frequency in seconds
        """
        super().__init__(system_id, update_freq)
        
        # Quadrotor-specific parameters
        self.altitude_min = 0.1  # Minimum safe altitude
        self.obstacle_distance = 2.0  # Initial distance to obstacle
        self.obstacle_approach_rate = 0.0  # Rate of approach to obstacle
        self.obstacle_radius = 0.5  # Obstacle radius (safety radius)
        
        # Changes to base parameters
        self.base_barrier_value = 0.8  # Higher initial safety margin
        self.threshold_value = 0.0  # Zero threshold for barrier functions
        self.instability_prob = 0.001  # Less frequent instabilities
    
    def _compute_barrier_values(self) -> tuple[float, float]:
        """
        Compute barrier and threshold values for quadrotor.
        
        Returns:
            Tuple of (barrier_value, threshold_value)
        """
        # Calculate height-based barrier component
        # B_height = z - z_min
        altitude = 0.5 + 0.2 * math.sin(2 * math.pi * 0.05 * self.current_time)  # Simulated altitude
        height_barrier = altitude - self.altitude_min
        
        # Update obstacle approach (random chance of starting approach)
        if self.obstacle_approach_rate == 0.0 and random.random() < 0.001:
            self.obstacle_approach_rate = random.uniform(0.1, 0.3)
        
        # Update obstacle distance
        if self.obstacle_approach_rate > 0.0:
            self.obstacle_distance -= self.obstacle_approach_rate * self.update_freq
            
            # If obstacle passed or avoidance maneuver
            if self.obstacle_distance < 1.0 or random.random() < 0.01:
                # End approach, reset distance
                self.obstacle_approach_rate = 0.0
                self.obstacle_distance = random.uniform(1.5, 3.0)
        
        # Calculate obstacle-based barrier component
        # B_obstacle = ||x - x_obstacle||^2 - r^2
        obstacle_barrier = self.obstacle_distance**2 - self.obstacle_radius**2
        
        # Use minimum of the two barrier functions (most conservative)
        barrier = min(height_barrier, obstacle_barrier)
        
        # Add noise
        barrier += random.uniform(-0.03, 0.03)
        
        # Apply instability effects (less frequent for quadrotor)
        if self.current_instability > 0.0:
            barrier -= self.instability_strength * self.current_instability
            self.current_instability -= self.update_freq / self.instability_duration
            self.current_instability = max(0.0, self.current_instability)
        
        # Handle recovery phase
        if self.recovering:
            recovery_progress = (self.current_time - self.recovery_start) / self.recovery_duration
            
            if recovery_progress >= 1.0:
                self.recovering = False
            else:
                recovery_factor = math.erf(3 * recovery_progress)
                barrier = self.threshold_value + recovery_factor * (self.base_barrier_value - self.threshold_value)
        
        # Random instability with lower probability
        elif random.random() < self.instability_prob and self.current_instability == 0.0 and not self.recovering:
            self.current_instability = 1.0
            self.recovering = True
            self.recovery_start = self.current_time + self.instability_duration * 0.8
        
        # For quadrotor, threshold is fixed at zero (barrier functions must be positive)
        threshold = self.threshold_value
        
        return barrier, threshold


# Factory function to create appropriate stream based on system type
def create_barrier_stream(system_id: str, system_type: Optional[str] = None) -> BarrierStream:
    """
    Create an appropriate barrier stream for the given system.
    
    Args:
        system_id: System ID
        system_type: System type (if None, determined from system_id)
        
    Returns:
        Barrier stream instance
    """
    if system_type is None:
        # Determine system type from ID
        if system_id.startswith("quadrotor"):
            system_type = "quadrotor"
        elif system_id.startswith("pendulum"):
            system_type = "pendulum"
        else:
            system_type = "generic"
    
    # Create appropriate stream
    if system_type == "quadrotor":
        return QuadrotorBarrierStream(system_id)
    else:
        return BarrierStream(system_id)


# Active streams
active_streams: Dict[str, BarrierStream] = {}

# Get or create stream
def get_barrier_stream(system_id: str, system_type: Optional[str] = None) -> BarrierStream:
    """
    Get or create a barrier stream for a system.
    
    Args:
        system_id: System ID
        system_type: System type
        
    Returns:
        Barrier stream instance
    """
    if system_id not in active_streams:
        active_streams[system_id] = create_barrier_stream(system_id, system_type)
    
    return active_streams[system_id]


async def generate_barrier_events(system_id: str, update_freq: float = 0.05) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for barrier function values.
    
    Args:
        system_id: System ID
        update_freq: Update frequency in seconds (50ms = 20Hz by default)
        
    Yields:
        SSE formatted data
    """
    stream = get_barrier_stream(system_id)
    
    async for data in stream.generate_stream():
        yield f"data: {data}\n\n"
