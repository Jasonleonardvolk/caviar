#!/usr/bin/env python3
"""
Lattice Morphing System
========================
Handles lattice/phase transformations, ψ-morphon dynamics,
and audio-visual synchronization for holographic rendering.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Kagome lattice parameters
KAGOME_UNIT_VECTORS = np.array([
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [-0.5, np.sqrt(3)/2, 0.0]
])

# Phase transition parameters
PHASE_TRANSITION_STEPS = 100
MORPH_INTERPOLATION = "cubic"  # linear, cubic, quantum
COHERENCE_THRESHOLD = 0.95

# Audio sync parameters
AUDIO_SAMPLE_RATE = 44100
AUDIO_FRAME_SIZE = 512
AV_SYNC_TOLERANCE_MS = 20

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LatticeState:
    """Represents current lattice configuration."""
    vertices: np.ndarray  # Nx3 array of vertex positions
    edges: List[Tuple[int, int]]  # Edge connectivity
    phase: float  # Current phase (0-1)
    energy: float  # System energy
    coherence: float  # Quantum coherence measure
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "vertices": self.vertices.tolist(),
            "edges": self.edges,
            "phase": self.phase,
            "energy": self.energy,
            "coherence": self.coherence,
            "timestamp": self.timestamp
        }

@dataclass
class MorphTransition:
    """Defines a morphing transition between states."""
    source_state: LatticeState
    target_state: LatticeState
    duration_ms: float
    interpolation: str
    audio_sync: Optional[Dict] = None
    
    def interpolate(self, t: float) -> LatticeState:
        """Interpolate between states at time t (0-1)."""
        if self.interpolation == "linear":
            alpha = t
        elif self.interpolation == "cubic":
            alpha = t * t * (3 - 2 * t)  # Smoothstep
        else:  # quantum
            alpha = 0.5 * (1 + np.sin(np.pi * (t - 0.5)))
        
        # Interpolate vertices
        vertices = (1 - alpha) * self.source_state.vertices + alpha * self.target_state.vertices
        
        # Interpolate phase
        phase = (1 - alpha) * self.source_state.phase + alpha * self.target_state.phase
        
        # Calculate energy and coherence
        energy = self._calculate_energy(vertices)
        coherence = self._calculate_coherence(vertices, phase)
        
        return LatticeState(
            vertices=vertices,
            edges=self.source_state.edges,  # Topology preserved
            phase=phase,
            energy=energy,
            coherence=coherence,
            timestamp=time.time()
        )
    
    def _calculate_energy(self, vertices: np.ndarray) -> float:
        """Calculate system energy."""
        # Simplified energy calculation
        return np.sum(np.linalg.norm(vertices, axis=1))
    
    def _calculate_coherence(self, vertices: np.ndarray, phase: float) -> float:
        """Calculate quantum coherence."""
        # Simplified coherence based on vertex alignment
        alignment = np.mean(np.dot(vertices, vertices.T))
        return np.abs(np.cos(phase * np.pi) * alignment)

# ============================================================================
# LATTICE GENERATOR
# ============================================================================

class LatticeGenerator:
    """Generates various lattice configurations."""
    
    @staticmethod
    def generate_kagome(size: int = 5) -> LatticeState:
        """Generate kagome lattice."""
        vertices = []
        edges = []
        
        # Generate hexagonal kagome pattern
        for i in range(size):
            for j in range(size):
                # Three vertices per unit cell
                base = i * KAGOME_UNIT_VECTORS[0] + j * KAGOME_UNIT_VECTORS[1]
                
                v1 = base
                v2 = base + KAGOME_UNIT_VECTORS[2] / 3
                v3 = base + 2 * KAGOME_UNIT_VECTORS[2] / 3
                
                idx = len(vertices)
                vertices.extend([v1, v2, v3])
                
                # Connect vertices in triangular pattern
                edges.extend([
                    (idx, idx + 1),
                    (idx + 1, idx + 2),
                    (idx + 2, idx)
                ])
        
        vertices = np.array(vertices)
        
        return LatticeState(
            vertices=vertices,
            edges=edges,
            phase=0.0,
            energy=np.sum(np.linalg.norm(vertices, axis=1)),
            coherence=1.0,
            timestamp=time.time()
        )
    
    @staticmethod
    def generate_cubic(size: int = 5) -> LatticeState:
        """Generate cubic lattice."""
        vertices = []
        edges = []
        
        # Generate cubic grid
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    vertices.append([i, j, k])
                    idx = len(vertices) - 1
                    
                    # Connect to neighbors
                    if i > 0:
                        edges.append((idx, idx - size * size))
                    if j > 0:
                        edges.append((idx, idx - size))
                    if k > 0:
                        edges.append((idx, idx - 1))
        
        vertices = np.array(vertices, dtype=np.float32)
        vertices -= np.mean(vertices, axis=0)  # Center
        
        return LatticeState(
            vertices=vertices,
            edges=edges,
            phase=0.5,
            energy=np.sum(np.linalg.norm(vertices, axis=1)),
            coherence=0.8,
            timestamp=time.time()
        )
    
    @staticmethod
    def generate_soliton(amplitude: float = 1.0, wavelength: float = 2.0) -> LatticeState:
        """Generate soliton wave lattice."""
        x = np.linspace(-10, 10, 100)
        y = np.zeros_like(x)
        z = amplitude * np.exp(-x**2 / wavelength**2) * np.cos(2 * np.pi * x / wavelength)
        
        vertices = np.column_stack([x, y, z])
        edges = [(i, i+1) for i in range(len(x)-1)]
        
        return LatticeState(
            vertices=vertices,
            edges=edges,
            phase=0.25,
            energy=np.sum(z**2),
            coherence=0.9,
            timestamp=time.time()
        )

# ============================================================================
# LATTICE MORPHER
# ============================================================================

class LatticeMorpher:
    """Manages lattice morphing and phase transitions."""
    
    def __init__(self):
        self.current_state: Optional[LatticeState] = None
        self.target_state: Optional[LatticeState] = None
        self.transition: Optional[MorphTransition] = None
        self.morphing = False
        self.morph_progress = 0.0
        self.morph_thread: Optional[threading.Thread] = None
        self.callbacks = []
        self.mesh_integration = None
        
        # Audio sync
        self.audio_sync_enabled = False
        self.audio_phase = 0.0
        
        # Initialize with kagome
        self.current_state = LatticeGenerator.generate_kagome()
    
    def morph_to(self,
                target: str,
                duration_ms: float = 1000,
                interpolation: str = "cubic",
                audio_sync: bool = False) -> bool:
        """
        Initiate morphing to target lattice type.
        
        Args:
            target: Target lattice type ("kagome", "cubic", "soliton")
            duration_ms: Morph duration in milliseconds
            interpolation: Interpolation method
            audio_sync: Whether to sync with audio
            
        Returns:
            Success flag
        """
        if self.morphing:
            logger.warning("Already morphing, please wait")
            return False
        
        # Generate target state
        if target == "kagome":
            self.target_state = LatticeGenerator.generate_kagome()
        elif target == "cubic":
            self.target_state = LatticeGenerator.generate_cubic()
        elif target == "soliton":
            self.target_state = LatticeGenerator.generate_soliton()
        else:
            logger.error(f"Unknown target lattice: {target}")
            return False
        
        # Create transition
        self.transition = MorphTransition(
            source_state=self.current_state,
            target_state=self.target_state,
            duration_ms=duration_ms,
            interpolation=interpolation,
            audio_sync={"enabled": audio_sync} if audio_sync else None
        )
        
        # Start morphing thread
        self.morphing = True
        self.morph_progress = 0.0
        self.morph_thread = threading.Thread(target=self._morph_worker)
        self.morph_thread.start()
        
        logger.info(f"Started morphing to {target}")
        return True
    
    def _morph_worker(self):
        """Worker thread for morphing animation."""
        start_time = time.time()
        duration_s = self.transition.duration_ms / 1000.0
        
        while self.morph_progress < 1.0:
            elapsed = time.time() - start_time
            self.morph_progress = min(1.0, elapsed / duration_s)
            
            # Interpolate state
            self.current_state = self.transition.interpolate(self.morph_progress)
            
            # Sync with audio if enabled
            if self.audio_sync_enabled:
                self._sync_with_audio()
            
            # Notify callbacks
            self._notify_callbacks()
            
            # Log to mesh if integrated
            if self.mesh_integration:
                self._log_to_mesh()
            
            # Sleep for smooth animation
            time.sleep(0.016)  # ~60 FPS
        
        # Finalize
        self.current_state = self.target_state
        self.morphing = False
        self.morph_progress = 1.0
        
        logger.info("Morphing complete")
    
    def _sync_with_audio(self):
        """Synchronize morphing with audio phase."""
        # This would integrate with audio processing
        # For now, simulate with sine wave
        self.audio_phase = np.sin(time.time() * 2 * np.pi)
        
        # Modulate coherence with audio
        if self.current_state:
            self.current_state.coherence *= (0.5 + 0.5 * self.audio_phase)
    
    def _notify_callbacks(self):
        """Notify registered callbacks of state change."""
        for callback in self.callbacks:
            try:
                callback(self.current_state)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _log_to_mesh(self):
        """Log morphing state to mesh/memory."""
        if not self.mesh_integration:
            return
        
        # This would integrate with concept mesh
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.current_state.phase,
            "coherence": self.current_state.coherence,
            "energy": self.current_state.energy,
            "morph_progress": self.morph_progress
        }
        
        # Write to mesh context
        try:
            log_file = Path("data/mesh_contexts/morphing_log.jsonl")
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log to mesh: {e}")
    
    def register_callback(self, callback):
        """Register callback for state updates."""
        self.callbacks.append(callback)
    
    def get_current_state(self) -> Optional[LatticeState]:
        """Get current lattice state."""
        return self.current_state
    
    def get_morph_progress(self) -> float:
        """Get morphing progress (0-1)."""
        return self.morph_progress
    
    def stop_morphing(self):
        """Stop current morphing."""
        self.morphing = False
        if self.morph_thread:
            self.morph_thread.join(timeout=1.0)
    
    def export_state(self, path: str):
        """Export current state to file."""
        if not self.current_state:
            return
        
        with open(path, 'w') as f:
            json.dump(self.current_state.to_dict(), f, indent=2)
        
        logger.info(f"Exported state to {path}")

# ============================================================================
# AV SYNC MANAGER
# ============================================================================

class AVSyncManager:
    """Manages audio-visual synchronization."""
    
    def __init__(self, morpher: LatticeMorpher):
        self.morpher = morpher
        self.audio_buffer = np.zeros(AUDIO_FRAME_SIZE)
        self.audio_phase = 0.0
        self.sync_offset_ms = 0.0
        self.coherence_history = []
        
    def process_audio_frame(self, audio_data: np.ndarray):
        """Process audio frame and update morphing."""
        # Calculate audio features
        amplitude = np.mean(np.abs(audio_data))
        frequency = self._estimate_frequency(audio_data)
        
        # Update audio phase
        self.audio_phase = (self.audio_phase + frequency * 0.01) % (2 * np.pi)
        
        # Modulate lattice based on audio
        if self.morpher.current_state:
            # Modulate coherence
            self.morpher.current_state.coherence = min(1.0, amplitude * 2)
            
            # Modulate phase
            self.morpher.current_state.phase = (self.morpher.current_state.phase + 
                                               frequency * 0.001) % 1.0
            
            # Track coherence
            self.coherence_history.append(self.morpher.current_state.coherence)
            if len(self.coherence_history) > 100:
                self.coherence_history.pop(0)
    
    def _estimate_frequency(self, audio_data: np.ndarray) -> float:
        """Estimate dominant frequency from audio."""
        # Simple zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        frequency = zero_crossings * AUDIO_SAMPLE_RATE / (2 * len(audio_data))
        return frequency
    
    def get_sync_quality(self) -> float:
        """Get AV sync quality metric (0-1)."""
        if not self.coherence_history:
            return 0.0
        
        # Calculate coherence stability
        coherence_std = np.std(self.coherence_history)
        quality = max(0, 1 - coherence_std)
        
        return quality

# ============================================================================
# API EXPOSURE
# ============================================================================

def create_lattice_morpher() -> LatticeMorpher:
    """Create and initialize lattice morpher."""
    morpher = LatticeMorpher()
    return morpher

def create_av_sync_manager(morpher: LatticeMorpher) -> AVSyncManager:
    """Create AV sync manager."""
    return AVSyncManager(morpher)

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for lattice morphing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lattice Morphing System")
    parser.add_argument("--target", choices=["kagome", "cubic", "soliton"],
                       default="cubic", help="Target lattice")
    parser.add_argument("--duration", type=int, default=2000,
                       help="Morph duration in ms")
    parser.add_argument("--interpolation", choices=["linear", "cubic", "quantum"],
                       default="cubic", help="Interpolation method")
    parser.add_argument("--audio_sync", action="store_true",
                       help="Enable audio sync")
    parser.add_argument("--export", help="Export final state to file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create morpher
    morpher = LatticeMorpher()
    
    # Register progress callback
    def progress_callback(state):
        print(f"\rMorphing: {morpher.get_morph_progress():.0%} | "
              f"Phase: {state.phase:.3f} | "
              f"Coherence: {state.coherence:.3f}", end="")
    
    morpher.register_callback(progress_callback)
    
    # Start morphing
    print(f"Morphing to {args.target}...")
    morpher.morph_to(
        target=args.target,
        duration_ms=args.duration,
        interpolation=args.interpolation,
        audio_sync=args.audio_sync
    )
    
    # Wait for completion
    while morpher.morphing:
        time.sleep(0.1)
    
    print("\nMorphing complete!")
    
    # Export if requested
    if args.export:
        morpher.export_state(args.export)
        print(f"State exported to {args.export}")

if __name__ == "__main__":
    main()
