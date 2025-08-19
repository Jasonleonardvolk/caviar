"""
Snapshot Buffer for Koopman Spectral Analysis.

This module provides a ring buffer implementation to store system state snapshots
for spectral analysis, enabling identification of dominant modes and stability assessment.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time

logger = logging.getLogger(__name__)


class SnapshotBuffer:
    """
    Ring buffer for capturing and managing system state snapshots over time.
    
    This buffer stores time-series data of concept states (phases, activations, etc.),
    which is used as input for Koopman spectral analysis.
    """
    
    def __init__(self, capacity: int = 100, state_dim: Optional[int] = None):
        """
        Initialize the snapshot buffer.
        
        Args:
            capacity: Maximum number of snapshots to store
            state_dim: Dimension of state vectors (auto-detected if None)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)
        self.metadata = deque(maxlen=capacity)
        self.concept_ids = []  # Ordered list of concept IDs in state vectors
        self.initialized = False
    
    def add_snapshot(self, 
                    state: Union[Dict[str, float], np.ndarray, List[float]],
                    timestamp: Optional[float] = None,
                    metadata: Optional[Dict] = None) -> None:
        """
        Add a system state snapshot to the buffer.
        
        Args:
            state: System state as dictionary (concept_id -> value) or vector
            timestamp: Timestamp for this snapshot (uses current time if None)
            metadata: Additional information about this snapshot
        """
        if timestamp is None:
            timestamp = time.time()
            
        if metadata is None:
            metadata = {}
            
        # Convert state to a standard format
        if isinstance(state, dict):
            # Dictionary of concept states
            if not self.initialized:
                # First snapshot, initialize concept_ids
                self.concept_ids = sorted(state.keys())
                self.state_dim = len(self.concept_ids)
                self.initialized = True
                
            # Convert to vector using consistent concept order
            state_vector = np.array([state.get(cid, 0.0) for cid in self.concept_ids])
        
        elif isinstance(state, list):
            # List of state values
            if not self.initialized:
                if self.state_dim is None:
                    self.state_dim = len(state)
                self.initialized = True
                
            state_vector = np.array(state)
            
        elif isinstance(state, np.ndarray):
            # NumPy array
            if not self.initialized:
                if self.state_dim is None:
                    self.state_dim = len(state)
                self.initialized = True
                
            state_vector = state.copy()
        
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
        
        # Verify vector dimensions
        if len(state_vector) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(state_vector)}")
        
        # Add to buffer
        self.buffer.append(state_vector)
        self.timestamps.append(timestamp)
        self.metadata.append(metadata)
    
    def get_snapshot_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the matrix of all snapshots and corresponding timestamps.
        
        Returns:
            Tuple of (snapshot_matrix, timestamps)
            
            snapshot_matrix: Matrix with shape (state_dim, n_snapshots)
            timestamps: Array of snapshot timestamps
        """
        if not self.buffer:
            raise ValueError("Buffer is empty")
        
        # Stack snapshots as columns in a matrix (state_dim × n_snapshots)
        snapshot_matrix = np.column_stack(self.buffer)
        timestamps = np.array(self.timestamps)
        
        return snapshot_matrix, timestamps
    
    def get_time_shifted_matrices(self, shift: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time-shifted snapshot matrices for DMD/Koopman analysis.
        
        Creates X and Y matrices where Y is shifted 'shift' steps ahead of X.
        
        Args:
            shift: Number of time steps to shift
            
        Returns:
            Tuple of (X, Y) matrices
            
            X: Matrix of snapshots at times t₀, t₁, ..., t_{n-shift-1}
            Y: Matrix of snapshots at times t_{shift}, t_{shift+1}, ..., t_{n-1}
        """
        if len(self.buffer) <= shift:
            raise ValueError(f"Buffer contains {len(self.buffer)} snapshots, need at least {shift+1}")
        
        # Create X and Y matrices
        X = np.column_stack(list(self.buffer)[:-shift])
        Y = np.column_stack(list(self.buffer)[shift:])
        
        return X, Y
    
    def get_trajectory(self, concept_id: Optional[str] = None, 
                      index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory (time-series) for a specific concept or state dimension.
        
        Args:
            concept_id: Concept ID to retrieve trajectory for
            index: State dimension index (alternative to concept_id)
            
        Returns:
            Tuple of (values, timestamps)
            
            values: Time-series of values for the specified concept/index
            timestamps: Corresponding timestamps
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
            
        # Determine index to extract
        if concept_id is not None:
            if concept_id not in self.concept_ids:
                raise ValueError(f"Concept ID '{concept_id}' not found in buffer")
            idx = self.concept_ids.index(concept_id)
        elif index is not None:
            if index < 0 or index >= self.state_dim:
                raise ValueError(f"Index {index} out of bounds (0-{self.state_dim-1})")
            idx = index
        else:
            raise ValueError("Either concept_id or index must be provided")
        
        # Extract trajectory
        values = np.array([snapshot[idx] for snapshot in self.buffer])
        timestamps = np.array(self.timestamps)
        
        return values, timestamps
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.timestamps.clear()
        self.metadata.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity
    
    def get_average_snapshot(self) -> np.ndarray:
        """
        Compute the average snapshot across the buffer.
        
        Returns:
            Average state vector
        """
        if not self.buffer:
            raise ValueError("Buffer is empty")
        
        return np.mean(self.buffer, axis=0)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Compute covariance matrix of state snapshots.
        
        Returns:
            Covariance matrix with shape (state_dim, state_dim)
        """
        if len(self.buffer) < 2:
            raise ValueError("Need at least 2 snapshots to compute covariance")
        
        # Stack snapshots as rows in a matrix
        data = np.vstack(self.buffer)
        
        # Compute covariance
        return np.cov(data, rowvar=False)
    
    def export_data(self) -> Dict:
        """
        Export buffer data as a dictionary.
        
        Returns:
            Dictionary containing buffer data
        """
        return {
            'concept_ids': self.concept_ids,
            'state_dim': self.state_dim,
            'snapshots': list(self.buffer) if self.buffer else [],
            'timestamps': list(self.timestamps) if self.timestamps else [],
            'metadata': list(self.metadata) if self.metadata else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SnapshotBuffer':
        """
        Create a SnapshotBuffer from exported data.
        
        Args:
            data: Dictionary of buffer data (from export_data)
            
        Returns:
            New SnapshotBuffer instance
        """
        buffer = cls(capacity=len(data.get('snapshots', [])), 
                     state_dim=data.get('state_dim'))
        
        buffer.concept_ids = data.get('concept_ids', [])
        buffer.initialized = bool(buffer.concept_ids)
        
        # Restore snapshots
        snapshots = data.get('snapshots', [])
        timestamps = data.get('timestamps', [])
        metadata = data.get('metadata', [])
        
        for i, snapshot in enumerate(snapshots):
            timestamp = timestamps[i] if i < len(timestamps) else None
            meta = metadata[i] if i < len(metadata) else {}
            buffer.buffer.append(np.array(snapshot))
            buffer.timestamps.append(timestamp)
            buffer.metadata.append(meta)
        
        return buffer
