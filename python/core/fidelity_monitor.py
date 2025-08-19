#!/usr/bin/env python3
"""
Soliton Fidelity Monitor
Tracks soliton preservation during topology morphing and other operations
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class FidelitySnapshot:
    """Snapshot of system state for fidelity tracking"""
    timestamp: float
    wavefunctions: np.ndarray  # Complex amplitudes
    phases: np.ndarray
    amplitudes: np.ndarray
    total_norm: float
    topology: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelityReport:
    """Fidelity analysis report"""
    drift: float  # ||ψ(t) - ψ₀||
    relative_drift: float  # drift / ||ψ₀||
    phase_coherence: float  # Phase correlation
    amplitude_preservation: float  # Amplitude correlation
    norm_conservation: float  # |N(t) - N₀| / N₀
    duration: float  # Time since reference
    warnings: List[str] = field(default_factory=list)


class SolitonFidelityMonitor:
    """
    Monitor soliton wavefunction fidelity during system operations
    
    Key metrics:
    1. Wavefunction drift: ||ψ(t) - ψ₀||
    2. Phase coherence: correlation of phase evolution
    3. Amplitude preservation: correlation of amplitude profile
    4. Norm conservation: total probability conservation
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.01,
        critical_threshold: float = 0.05,
        history_size: int = 1000
    ):
        """
        Initialize fidelity monitor
        
        Args:
            warning_threshold: Drift threshold for warnings
            critical_threshold: Drift threshold for critical alerts
            history_size: Number of snapshots to retain
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        # Reference state
        self.reference_snapshot: Optional[FidelitySnapshot] = None
        
        # History tracking
        self.snapshot_history: deque = deque(maxlen=history_size)
        self.fidelity_history: deque = deque(maxlen=history_size)
        
        # Current monitoring state
        self.is_monitoring = False
        self.morph_start_snapshot: Optional[FidelitySnapshot] = None
        
    def set_reference(self, system: Any) -> FidelitySnapshot:
        """
        Set reference state for fidelity comparison
        
        Args:
            system: System object with wavefunction data
            
        Returns:
            Reference snapshot
        """
        snapshot = self._capture_snapshot(system)
        self.reference_snapshot = snapshot
        self.is_monitoring = True
        
        logger.info(f"Set fidelity reference: norm={snapshot.total_norm:.6f}, "
                   f"topology={snapshot.topology}")
        
        return snapshot
    
    def _capture_snapshot(self, system: Any) -> FidelitySnapshot:
        """Capture current system state"""
        # Extract wavefunction data
        if hasattr(system, 'psi'):
            # Direct wavefunction
            psi = np.asarray(system.psi)
            phases = np.angle(psi)
            amplitudes = np.abs(psi)
        elif hasattr(system, 'phases') and hasattr(system, 'amplitudes'):
            # Separate phase/amplitude
            phases = np.asarray(system.phases)
            amplitudes = np.asarray(system.amplitudes)
            psi = amplitudes * np.exp(1j * phases)
        elif hasattr(system, 'oscillators'):
            # Oscillator lattice
            phases = []
            amplitudes = []
            for osc in system.oscillators:
                if isinstance(osc, dict):
                    phases.append(osc.get('phase', 0))
                    amplitudes.append(osc.get('amplitude', 0))
                else:
                    phases.append(osc.phase)
                    amplitudes.append(osc.amplitude)
            phases = np.array(phases)
            amplitudes = np.array(amplitudes)
            psi = amplitudes * np.exp(1j * phases)
        else:
            raise ValueError("Cannot extract wavefunction from system")
        
        # Get topology
        topology = "unknown"
        if hasattr(system, 'current_topology'):
            topology = system.current_topology
        elif hasattr(system, 'topology'):
            topology = system.topology
        
        # Calculate total norm
        total_norm = np.sum(np.abs(psi)**2)
        
        return FidelitySnapshot(
            timestamp=time.time(),
            wavefunctions=psi,
            phases=phases,
            amplitudes=amplitudes,
            total_norm=total_norm,
            topology=topology
        )
    
    def check_fidelity(self, system: Any) -> FidelityReport:
        """
        Check current fidelity against reference
        
        Args:
            system: Current system state
            
        Returns:
            Fidelity report
        """
        if self.reference_snapshot is None:
            raise ValueError("No reference snapshot set")
        
        # Capture current state
        current = self._capture_snapshot(system)
        
        # Store in history
        self.snapshot_history.append(current)
        
        # Compute fidelity metrics
        report = self._compute_fidelity(self.reference_snapshot, current)
        
        # Store report
        self.fidelity_history.append(report)
        
        # Check thresholds
        if report.relative_drift > self.critical_threshold:
            logger.critical(f"Critical fidelity loss: {report.relative_drift:.2%}")
            report.warnings.append(f"CRITICAL: Fidelity drift {report.relative_drift:.2%}")
        elif report.relative_drift > self.warning_threshold:
            logger.warning(f"Fidelity warning: {report.relative_drift:.2%}")
            report.warnings.append(f"WARNING: Fidelity drift {report.relative_drift:.2%}")
        
        return report
    
    def _compute_fidelity(
        self,
        reference: FidelitySnapshot,
        current: FidelitySnapshot
    ) -> FidelityReport:
        """Compute fidelity metrics between snapshots"""
        # Ensure compatible sizes
        if reference.wavefunctions.shape != current.wavefunctions.shape:
            # Pad or truncate as needed
            min_size = min(len(reference.wavefunctions), len(current.wavefunctions))
            ref_psi = reference.wavefunctions[:min_size]
            cur_psi = current.wavefunctions[:min_size]
        else:
            ref_psi = reference.wavefunctions
            cur_psi = current.wavefunctions
        
        # Wavefunction drift
        drift = np.linalg.norm(cur_psi - ref_psi)
        ref_norm = np.linalg.norm(ref_psi)
        relative_drift = drift / (ref_norm + 1e-10)
        
        # Phase coherence
        ref_phases = np.angle(ref_psi)
        cur_phases = np.angle(cur_psi)
        
        # Unwrap phases for better comparison
        phase_diff = np.unwrap(cur_phases - ref_phases)
        phase_coherence = 1.0 - np.std(phase_diff) / np.pi
        phase_coherence = max(0, min(1, phase_coherence))
        
        # Amplitude preservation
        ref_amp = np.abs(ref_psi)
        cur_amp = np.abs(cur_psi)
        
        # Correlation coefficient
        if np.std(ref_amp) > 1e-10 and np.std(cur_amp) > 1e-10:
            amplitude_preservation = np.corrcoef(ref_amp, cur_amp)[0, 1]
        else:
            amplitude_preservation = 1.0 if np.allclose(ref_amp, cur_amp) else 0.0
        
        # Norm conservation
        norm_drift = abs(current.total_norm - reference.total_norm) / (reference.total_norm + 1e-10)
        norm_conservation = 1.0 - norm_drift
        
        # Duration
        duration = current.timestamp - reference.timestamp
        
        # Build warnings
        warnings = []
        
        if norm_drift > 0.01:
            warnings.append(f"Norm drift: {norm_drift:.2%}")
        
        if phase_coherence < 0.9:
            warnings.append(f"Low phase coherence: {phase_coherence:.2f}")
        
        if amplitude_preservation < 0.9:
            warnings.append(f"Low amplitude preservation: {amplitude_preservation:.2f}")
        
        return FidelityReport(
            drift=drift,
            relative_drift=relative_drift,
            phase_coherence=phase_coherence,
            amplitude_preservation=amplitude_preservation,
            norm_conservation=norm_conservation,
            duration=duration,
            warnings=warnings
        )
    
    def start_morph_monitoring(self, system: Any) -> None:
        """Start monitoring for topology morphing"""
        self.morph_start_snapshot = self._capture_snapshot(system)
        logger.info(f"Started morph monitoring from topology: {self.morph_start_snapshot.topology}")
    
    def end_morph_monitoring(self, system: Any) -> FidelityReport:
        """End morphing and get final fidelity report"""
        if self.morph_start_snapshot is None:
            raise ValueError("No morph monitoring started")
        
        final_snapshot = self._capture_snapshot(system)
        report = self._compute_fidelity(self.morph_start_snapshot, final_snapshot)
        
        logger.info(f"Morph complete: {self.morph_start_snapshot.topology} → "
                   f"{final_snapshot.topology}, fidelity={1-report.relative_drift:.2%}")
        
        self.morph_start_snapshot = None
        return report
    
    def get_drift_history(self, window: int = 100) -> List[float]:
        """Get recent drift values"""
        recent = list(self.fidelity_history)[-window:]
        return [r.relative_drift for r in recent]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fidelity statistics"""
        if not self.fidelity_history:
            return {
                'num_checks': 0,
                'avg_drift': 0.0,
                'max_drift': 0.0,
                'warnings': 0,
                'critical': 0
            }
        
        drifts = [r.relative_drift for r in self.fidelity_history]
        warnings = sum(1 for r in self.fidelity_history if r.warnings)
        critical = sum(1 for r in self.fidelity_history 
                      if any('CRITICAL' in w for w in r.warnings))
        
        return {
            'num_checks': len(self.fidelity_history),
            'avg_drift': np.mean(drifts),
            'max_drift': np.max(drifts),
            'std_drift': np.std(drifts),
            'warnings': warnings,
            'critical': critical,
            'current_drift': drifts[-1] if drifts else 0.0
        }
    
    def suggest_safe_parameters(
        self,
        current_drift: float,
        target_fidelity: float = 0.99
    ) -> Dict[str, float]:
        """
        Suggest parameters to maintain target fidelity
        
        Args:
            current_drift: Current relative drift
            target_fidelity: Target fidelity (1 - drift)
            
        Returns:
            Suggested parameters
        """
        target_drift = 1.0 - target_fidelity
        
        if current_drift <= target_drift:
            # Already meeting target
            return {
                'morph_rate': 0.02,  # Can use normal rate
                'damping': 0.0,
                'recommendation': 'Current parameters are safe'
            }
        
        # Need to slow down
        rate_reduction = target_drift / (current_drift + 1e-10)
        suggested_rate = 0.02 * rate_reduction
        
        # May need damping
        suggested_damping = 0.0
        if current_drift > 2 * target_drift:
            suggested_damping = 0.01 * (current_drift / target_drift - 1)
        
        return {
            'morph_rate': max(0.001, suggested_rate),
            'damping': min(0.1, suggested_damping),
            'recommendation': f'Reduce morph rate to {suggested_rate:.4f}'
        }
    
    def reset(self):
        """Reset monitor state"""
        self.reference_snapshot = None
        self.morph_start_snapshot = None
        self.snapshot_history.clear()
        self.fidelity_history.clear()
        self.is_monitoring = False
        logger.info("Fidelity monitor reset")


# Global monitor instance
_global_monitor: Optional[SolitonFidelityMonitor] = None


def get_fidelity_monitor() -> SolitonFidelityMonitor:
    """Get global fidelity monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SolitonFidelityMonitor()
    return _global_monitor


# Integration functions
def monitor_morphing(hot_swap_system: Any) -> None:
    """
    Monitor fidelity during topology morphing
    
    Args:
        hot_swap_system: HotSwapLaplacian instance
    """
    monitor = get_fidelity_monitor()
    
    # Set up monitoring before morph
    if not hot_swap_system.is_morphing:
        monitor.start_morph_monitoring(hot_swap_system)
    
    # Check fidelity during morph
    while hot_swap_system.is_morphing:
        report = monitor.check_fidelity(hot_swap_system)
        
        # Log progress
        logger.debug(f"Morph progress: {hot_swap_system.morph_progress:.1%}, "
                    f"fidelity: {1-report.relative_drift:.2%}")
        
        # Apply safety measures if needed
        if report.relative_drift > monitor.critical_threshold:
            logger.error("Critical fidelity loss during morph - slowing down")
            hot_swap_system.morph_rate *= 0.5
        
        time.sleep(0.1)
    
    # Final report
    final_report = monitor.end_morph_monitoring(hot_swap_system)
    logger.info(f"Morphing complete with {1-final_report.relative_drift:.2%} fidelity")


if __name__ == "__main__":
    # Test the fidelity monitor
    print("Testing Soliton Fidelity Monitor")
    print("="*50)
    
    # Create test system
    class TestSystem:
        def __init__(self, size=100):
            self.phases = np.random.uniform(0, 2*np.pi, size)
            self.amplitudes = np.ones(size)
            self.current_topology = "kagome"
    
    system = TestSystem()
    monitor = SolitonFidelityMonitor()
    
    # Set reference
    ref = monitor.set_reference(system)
    print(f"Reference set: norm={ref.total_norm:.6f}")
    
    # Simulate evolution with drift
    for step in range(50):
        # Add small perturbations
        system.phases += 0.01 * np.random.randn(len(system.phases))
        system.amplitudes *= (1 + 0.001 * np.random.randn(len(system.amplitudes)))
        
        # Check fidelity
        report = monitor.check_fidelity(system)
        
        if step % 10 == 0:
            print(f"Step {step}: drift={report.relative_drift:.4f}, "
                  f"phase_coherence={report.phase_coherence:.3f}")
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nStatistics:")
    print(f"  Average drift: {stats['avg_drift']:.4f}")
    print(f"  Max drift: {stats['max_drift']:.4f}")
    print(f"  Warnings: {stats['warnings']}")
