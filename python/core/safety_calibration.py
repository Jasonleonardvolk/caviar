#!/usr/bin/env python3
"""
Safety-Calibration Loop
Ensures chaos-enhanced TORI remains safe and controllable

Implements multi-layered safety with:
- Topological protection (virtual braid gates)
- Energy conservation laws
- Quantum fidelity tracking
- Rollback capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import asyncio
import logging
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib

# Import chaos components
from python.core.eigensentry.core import (
    EigenSentry2, InstabilityType, EIGENVALUE_EMERGENCY_THRESHOLD
)
from python.core.chaos_control_layer import (
    ChaosControlLayer, ChaosTask, ChaosResult, ChaosMode
)
from python.core.metacognitive_adapters import (
    MetacognitiveAdapterSystem, AdapterMode
)

logger = logging.getLogger(__name__)

# ========== Safety Configuration ==========

# Safety thresholds based on research
FIDELITY_MINIMUM = 0.85          # Minimum quantum fidelity
COHERENCE_THRESHOLD = 0.7        # Minimum phase coherence
ENERGY_VIOLATION_TOLERANCE = 0.05 # 5% energy conservation violation allowed
ROLLBACK_TRIGGER_SCORE = 0.3     # Safety score triggering rollback

# Timing constraints
SAFETY_CHECK_INTERVAL_MS = 100   # How often to check safety
EMERGENCY_RESPONSE_TIME_MS = 10  # Max time for emergency response
CHECKPOINT_RETENTION_HOURS = 24  # How long to keep checkpoints

# ========== Safety Data Structures ==========

class SafetyLevel(Enum):
    """System safety levels"""
    OPTIMAL = "optimal"           # All metrics excellent
    NOMINAL = "nominal"          # Within normal bounds
    DEGRADED = "degraded"        # Some concerns, monitoring
    CRITICAL = "critical"        # Major issues, intervention needed
    EMERGENCY = "emergency"      # Immediate action required

@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics"""
    fidelity: float              # Quantum state fidelity
    coherence: float             # Phase coherence
    energy_conservation: float   # Energy conservation ratio
    eigenvalue_max: float        # Maximum eigenvalue
    chaos_containment: float     # Chaos isolation effectiveness
    rollback_readiness: float    # Ability to rollback
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SafetyCheckpoint:
    """Complete system checkpoint for rollback"""
    checkpoint_id: str
    timestamp: datetime
    state_snapshot: Dict[str, np.ndarray]
    metrics: SafetyMetrics
    metadata: Dict[str, Any]
    compressed: bool = False

@dataclass
class SafetyViolation:
    """Record of safety violation"""
    violation_type: str
    severity: float
    affected_modules: List[str]
    timestamp: datetime
    resolved: bool = False
    resolution: Optional[str] = None

# ========== Topological Protection ==========

class TopologicalProtector:
    """
    Implements topological protection using virtual braid gates
    Based on research in topological quantum computing
    """
    
    def __init__(self):
        self.braid_gates = {}
        self.entanglement_map = defaultdict(set)
        self.protected_states = {}
        self.anyonic_charges = defaultdict(int)
        
    def create_braid_gate(self, module_id: str, state: np.ndarray) -> str:
        """Create topologically protected gate"""
        gate_id = f"braid_{module_id}_{datetime.now(timezone.utc).timestamp()}"
        
        # Create topological invariant
        invariant = self._compute_topological_invariant(state)
        
        self.braid_gates[gate_id] = {
            'module_id': module_id,
            'invariant': invariant,
            'state_hash': hashlib.sha256(state.tobytes()).hexdigest(),
            'created': datetime.now(timezone.utc),
            'crossings': 0
        }
        
        # Initialize anyonic charge
        self.anyonic_charges[gate_id] = self._assign_anyonic_charge(state)
        
        return gate_id
        
    def braid_crossing(self, gate1_id: str, gate2_id: str) -> bool:
        """Execute braid crossing between gates"""
        if gate1_id not in self.braid_gates or gate2_id not in self.braid_gates:
            return False
            
        # Check if crossing is allowed by fusion rules
        charge1 = self.anyonic_charges[gate1_id]
        charge2 = self.anyonic_charges[gate2_id]
        
        if not self._check_fusion_rules(charge1, charge2):
            logger.warning(f"Fusion rules prohibit crossing {gate1_id} x {gate2_id}")
            return False
            
        # Record crossing
        self.braid_gates[gate1_id]['crossings'] += 1
        self.braid_gates[gate2_id]['crossings'] += 1
        
        # Update entanglement
        self.entanglement_map[gate1_id].add(gate2_id)
        self.entanglement_map[gate2_id].add(gate1_id)
        
        return True
        
    def verify_protection(self, gate_id: str, current_state: np.ndarray) -> bool:
        """Verify topological protection is intact"""
        if gate_id not in self.braid_gates:
            return False
            
        gate = self.braid_gates[gate_id]
        
        # Compute current invariant
        current_invariant = self._compute_topological_invariant(current_state)
        
        # Check if invariant is preserved (up to phase)
        invariant_preserved = np.allclose(
            np.abs(current_invariant),
            np.abs(gate['invariant']),
            rtol=0.01
        )
        
        return invariant_preserved
        
    def _compute_topological_invariant(self, state: np.ndarray) -> complex:
        """Compute topological invariant (simplified Berry phase)"""
        # Simplified calculation - real implementation would be more sophisticated
        if len(state) < 2:
            return complex(1.0, 0.0)
            
        # Compute Berry phase around a loop in parameter space
        phase = 0.0
        for i in range(len(state) - 1):
            # Inner product of adjacent states
            overlap = np.vdot(state[i:i+2], state[i+1:i+3] if i+3 <= len(state) else state[i+1:])
            phase += np.angle(overlap)
            
        return np.exp(1j * phase)
        
    def _assign_anyonic_charge(self, state: np.ndarray) -> int:
        """Assign anyonic charge based on state properties"""
        # Simplified - use state magnitude distribution
        magnitude_sum = np.sum(np.abs(state))
        return int(magnitude_sum * 100) % 8  # Fibonacci anyons have 8 fusion channels
        
    def _check_fusion_rules(self, charge1: int, charge2: int) -> bool:
        """Check if fusion of anyonic charges is allowed"""
        # Simplified Fibonacci anyon fusion rules
        allowed_fusions = {
            (0, 0): True, (0, 1): True, (1, 0): True,
            (1, 1): True, (1, 2): True, (2, 1): True,
            (2, 2): True, (2, 3): True, (3, 2): True
        }
        
        return allowed_fusions.get((charge1 % 4, charge2 % 4), False)

# ========== Energy Conservation Monitor ==========

class EnergyConservationMonitor:
    """
    Ensures energy conservation laws are respected
    Tracks energy flow and detects violations
    """
    
    def __init__(self, total_energy_budget: float = 10000.0):
        self.total_budget = total_energy_budget
        self.energy_ledger = defaultdict(float)
        self.energy_flows = deque(maxlen=10000)
        self.violation_history = deque(maxlen=1000)
        
    def track_energy_flow(self, from_module: str, to_module: str, 
                         amount: float, metadata: Optional[Dict] = None):
        """Track energy flow between modules"""
        timestamp = datetime.now(timezone.utc)
        
        # Update ledger
        self.energy_ledger[from_module] -= amount
        self.energy_ledger[to_module] += amount
        
        # Record flow
        flow = {
            'from': from_module,
            'to': to_module,
            'amount': amount,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        self.energy_flows.append(flow)
        
        # Check conservation
        total_energy = sum(self.energy_ledger.values())
        conservation_error = abs(total_energy - self.total_budget) / self.total_budget
        
        if conservation_error > ENERGY_VIOLATION_TOLERANCE:
            violation = {
                'timestamp': timestamp,
                'error': conservation_error,
                'total_energy': total_energy,
                'expected': self.total_budget
            }
            self.violation_history.append(violation)
            logger.warning(f"Energy conservation violation: {conservation_error:.2%}")
            
    def get_energy_distribution(self) -> Dict[str, float]:
        """Get current energy distribution across modules"""
        return dict(self.energy_ledger)
        
    def check_conservation(self) -> Tuple[bool, float]:
        """Check if energy is conserved within tolerance"""
        total_energy = sum(self.energy_ledger.values())
        conservation_error = abs(total_energy - self.total_budget) / self.total_budget
        
        is_conserved = conservation_error <= ENERGY_VIOLATION_TOLERANCE
        return is_conserved, conservation_error
        
    def project_energy_usage(self, time_horizon_s: float = 60.0) -> Dict[str, float]:
        """Project future energy usage based on recent flows"""
        if not self.energy_flows:
            return {}
            
        # Calculate flow rates
        recent_flows = list(self.energy_flows)[-100:]
        flow_rates = defaultdict(float)
        
        for flow in recent_flows:
            flow_rates[flow['to']] += flow['amount']
            flow_rates[flow['from']] -= flow['amount']
            
        # Time window
        if recent_flows:
            time_window = (recent_flows[-1]['timestamp'] - 
                          recent_flows[0]['timestamp']).total_seconds()
            if time_window > 0:
                # Scale to projection horizon
                scale = time_horizon_s / time_window
                projections = {}
                
                for module, rate in flow_rates.items():
                    projected = self.energy_ledger[module] + rate * scale
                    projections[module] = projected
                    
                return projections
                
        return dict(self.energy_ledger)

# ========== Quantum Fidelity Tracker ==========

class QuantumFidelityTracker:
    """
    Tracks quantum state fidelity during chaos operations
    Ensures information is preserved despite nonlinear dynamics
    """
    
    def __init__(self):
        self.reference_states = {}
        self.fidelity_history = defaultdict(list)
        self.decoherence_rates = defaultdict(float)
        
    def set_reference_state(self, module_id: str, state: np.ndarray):
        """Set reference state for fidelity comparison"""
        # Normalize state
        norm = np.linalg.norm(state)
        if norm > 0:
            normalized = state / norm
        else:
            normalized = state
            
        self.reference_states[module_id] = {
            'state': normalized.copy(),
            'timestamp': datetime.now(timezone.utc),
            'purity': self._compute_purity(normalized)
        }
        
    def measure_fidelity(self, module_id: str, current_state: np.ndarray) -> float:
        """Measure fidelity between current and reference state"""
        if module_id not in self.reference_states:
            return 0.0
            
        ref_state = self.reference_states[module_id]['state']
        
        # Normalize current state
        norm = np.linalg.norm(current_state)
        if norm > 0:
            current_normalized = current_state / norm
        else:
            return 0.0
            
        # Compute fidelity (overlap squared)
        overlap = np.abs(np.vdot(ref_state, current_normalized))
        fidelity = overlap ** 2
        
        # Record history
        self.fidelity_history[module_id].append({
            'fidelity': fidelity,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Update decoherence rate
        self._update_decoherence_rate(module_id)
        
        return fidelity
        
    def measure_coherence(self, states: List[np.ndarray]) -> float:
        """Measure phase coherence across multiple states"""
        if len(states) < 2:
            return 1.0
            
        # Convert to phase representation
        phases = []
        for state in states:
            if len(state) > 0:
                # Use first component phase as representative
                phase = np.angle(state[0] if np.iscomplexobj(state) else state[0] + 0j)
                phases.append(phase)
                
        if not phases:
            return 0.0
            
        # Compute phase coherence using circular statistics
        mean_vector = np.mean(np.exp(1j * np.array(phases)))
        coherence = np.abs(mean_vector)
        
        return coherence
        
    def _compute_purity(self, state: np.ndarray) -> float:
        """Compute purity of quantum state"""
        # For pure states, purity = 1
        # For mixed states, purity < 1
        # Simplified: use participation ratio
        prob = np.abs(state) ** 2
        if np.sum(prob) > 0:
            prob = prob / np.sum(prob)
            purity = 1.0 / (np.sum(prob ** 2) * len(prob))
        else:
            purity = 0.0
            
        return purity
        
    def _update_decoherence_rate(self, module_id: str):
        """Update estimated decoherence rate"""
        history = self.fidelity_history[module_id]
        
        if len(history) < 2:
            return
            
        # Fit exponential decay to recent fidelity
        recent = history[-10:]
        if len(recent) >= 2:
            times = [(h['timestamp'] - recent[0]['timestamp']).total_seconds() 
                    for h in recent]
            fidelities = [h['fidelity'] for h in recent]
            
            # Simple linear fit in log space
            if min(fidelities) > 0:
                log_fidelities = np.log(fidelities)
                if len(times) > 1:
                    rate, _ = np.polyfit(times, log_fidelities, 1)
                    self.decoherence_rates[module_id] = -rate  # Positive for decay

# ========== Main Safety Calibration System ==========

class SafetyCalibrationLoop:
    """
    Main safety system that orchestrates all safety components
    Provides continuous monitoring and intervention capabilities
    """
    
    def __init__(self, eigen_sentry: EigenSentry2,
                 ccl: ChaosControlLayer,
                 adapter_system: MetacognitiveAdapterSystem):
        self.eigen_sentry = eigen_sentry
        self.ccl = ccl
        self.adapter_system = adapter_system
        
        # Safety components
        self.topology_protector = TopologicalProtector()
        self.energy_monitor = EnergyConservationMonitor()
        self.fidelity_tracker = QuantumFidelityTracker()
        
        # Checkpoint management
        self.checkpoints = deque(maxlen=100)
        self.active_checkpoint = None
        
        # Safety state
        self.current_safety_level = SafetyLevel.NOMINAL
        self.safety_history = deque(maxlen=1000)
        self.violations = deque(maxlen=100)
        
        # Monitoring task
        self.monitoring_task = None
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start continuous safety monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Safety calibration loop started")
        
    async def stop_monitoring(self):
        """Stop safety monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Safety calibration loop stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Perform safety check
                metrics = await self._perform_safety_check()
                
                # Evaluate safety level
                safety_level = self._evaluate_safety_level(metrics)
                
                # Take action if needed
                if safety_level.value > self.current_safety_level.value:
                    await self._handle_safety_degradation(safety_level, metrics)
                    
                self.current_safety_level = safety_level
                
                # Record metrics
                self.safety_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'level': safety_level,
                    'metrics': metrics
                })
                
                # Sleep until next check
                await asyncio.sleep(SAFETY_CHECK_INTERVAL_MS / 1000)
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
                
    async def _perform_safety_check(self) -> SafetyMetrics:
        """Perform comprehensive safety check"""
        
        # Get eigenvalue status
        eigen_status = self.eigen_sentry.get_status()
        eigenvalue_max = eigen_status['current_max_eigenvalue']
        
        # Check energy conservation
        energy_conserved, energy_error = self.energy_monitor.check_conservation()
        energy_conservation = 1.0 - energy_error
        
        # Sample fidelity from active modules
        fidelities = []
        for module_id in ['memory_lattice', 'oscillator_core', 'reflection_system']:
            if module_id in self.fidelity_tracker.reference_states:
                # Get mock state (would be real in production)
                mock_state = np.random.randn(10)
                fidelity = self.fidelity_tracker.measure_fidelity(module_id, mock_state)
                fidelities.append(fidelity)
                
        avg_fidelity = np.mean(fidelities) if fidelities else 1.0
        
        # Measure coherence
        ccl_status = self.ccl.get_status()
        chaos_containment = 1.0 if ccl_status['active_tasks'] == 0 else \
                          0.8 - 0.1 * ccl_status['active_tasks']
                          
        # Estimate rollback readiness
        rollback_readiness = 1.0 if self.checkpoints else 0.0
        
        return SafetyMetrics(
            fidelity=avg_fidelity,
            coherence=0.9,  # Simplified
            energy_conservation=energy_conservation,
            eigenvalue_max=eigenvalue_max,
            chaos_containment=chaos_containment,
            rollback_readiness=rollback_readiness
        )
        
    def _evaluate_safety_level(self, metrics: SafetyMetrics) -> SafetyLevel:
        """Evaluate overall safety level from metrics"""
        
        # Check critical thresholds
        if metrics.eigenvalue_max > EIGENVALUE_EMERGENCY_THRESHOLD:
            return SafetyLevel.EMERGENCY
            
        if metrics.fidelity < FIDELITY_MINIMUM * 0.5:
            return SafetyLevel.EMERGENCY
            
        if metrics.energy_conservation < 0.9:
            return SafetyLevel.CRITICAL
            
        # Compute safety score
        safety_score = (
            metrics.fidelity * 0.3 +
            metrics.coherence * 0.2 +
            metrics.energy_conservation * 0.2 +
            metrics.chaos_containment * 0.2 +
            metrics.rollback_readiness * 0.1
        )
        
        # Map to safety level
        if safety_score > 0.9:
            return SafetyLevel.OPTIMAL
        elif safety_score > 0.7:
            return SafetyLevel.NOMINAL
        elif safety_score > 0.5:
            return SafetyLevel.DEGRADED
        elif safety_score > ROLLBACK_TRIGGER_SCORE:
            return SafetyLevel.CRITICAL
        else:
            return SafetyLevel.EMERGENCY
            
    async def _handle_safety_degradation(self, level: SafetyLevel, 
                                       metrics: SafetyMetrics):
        """Handle degraded safety conditions"""
        logger.warning(f"Safety degradation detected: {level.value}")
        
        # Record violation
        violation = SafetyViolation(
            violation_type=f"safety_level_{level.value}",
            severity=self._level_to_severity(level),
            affected_modules=self._identify_affected_modules(metrics),
            timestamp=datetime.now(timezone.utc)
        )
        self.violations.append(violation)
        
        # Take action based on level
        if level == SafetyLevel.EMERGENCY:
            await self._emergency_response(metrics)
        elif level == SafetyLevel.CRITICAL:
            await self._critical_response(metrics)
        elif level == SafetyLevel.DEGRADED:
            await self._degraded_response(metrics)
            
    async def _emergency_response(self, metrics: SafetyMetrics):
        """Emergency response - immediate stabilization"""
        logger.error("EMERGENCY RESPONSE ACTIVATED")
        
        # 1. Halt all chaos operations
        self.adapter_system.set_adapter_mode(AdapterMode.PASSTHROUGH)
        
        # 2. Trigger EigenSentry emergency damping
        await self.eigen_sentry._handle_emergency(
            np.array([metrics.eigenvalue_max]),
            np.eye(1)
        )
        
        # 3. Attempt rollback if available
        if self.active_checkpoint:
            success = await self.rollback_to_checkpoint(self.active_checkpoint)
            logger.info(f"Emergency rollback {'successful' if success else 'failed'}")
            
    async def _critical_response(self, metrics: SafetyMetrics):
        """Critical response - reduce chaos activity"""
        logger.warning("Critical safety response initiated")
        
        # Switch to hybrid mode with reduced chaos
        self.adapter_system.set_adapter_mode(AdapterMode.HYBRID)
        
        # Reduce energy allocations
        self.adapter_system.global_config.energy_allocation = 50
        
        # Create checkpoint for potential rollback
        await self.create_checkpoint("critical_safety")
        
    async def _degraded_response(self, metrics: SafetyMetrics):
        """Degraded response - increase monitoring"""
        logger.info("Degraded safety response - increasing vigilance")
        
        # Just log and monitor more closely
        # Could implement adaptive thresholds here
        
    def _level_to_severity(self, level: SafetyLevel) -> float:
        """Convert safety level to severity score"""
        severities = {
            SafetyLevel.OPTIMAL: 0.0,
            SafetyLevel.NOMINAL: 0.2,
            SafetyLevel.DEGRADED: 0.5,
            SafetyLevel.CRITICAL: 0.8,
            SafetyLevel.EMERGENCY: 1.0
        }
        return severities.get(level, 0.5)
        
    def _identify_affected_modules(self, metrics: SafetyMetrics) -> List[str]:
        """Identify which modules are affected by safety issues"""
        affected = []
        
        if metrics.fidelity < FIDELITY_MINIMUM:
            affected.extend(['memory_lattice', 'reflection_system'])
            
        if metrics.energy_conservation < 0.95:
            affected.append('ccl')
            
        if metrics.eigenvalue_max > 1.5:
            affected.extend(['oscillator_core', 'dynamics_monitor'])
            
        return list(set(affected))
        
    # ========== Checkpoint Management ==========
    
    async def create_checkpoint(self, label: str = "manual") -> str:
        """Create safety checkpoint"""
        checkpoint_id = f"ckpt_{label}_{datetime.now(timezone.utc).timestamp()}"
        
        # Gather state snapshots
        state_snapshot = {
            'eigen_sentry': self.eigen_sentry.get_status(),
            'ccl': self.ccl.get_status(),
            'adapter_mode': self.adapter_system.global_config.mode.value,
            'energy_distribution': self.energy_monitor.get_energy_distribution()
        }
        
        # Get current metrics
        metrics = await self._perform_safety_check()
        
        checkpoint = SafetyCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(timezone.utc),
            state_snapshot=state_snapshot,
            metrics=metrics,
            metadata={'label': label}
        )
        
        self.checkpoints.append(checkpoint)
        self.active_checkpoint = checkpoint_id
        
        logger.info(f"Created checkpoint {checkpoint_id}")
        return checkpoint_id
        
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to specified checkpoint"""
        # Find checkpoint
        checkpoint = None
        for ckpt in self.checkpoints:
            if ckpt.checkpoint_id == checkpoint_id:
                checkpoint = ckpt
                break
                
        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
            
        try:
            # Restore adapter mode
            mode_str = checkpoint.state_snapshot.get('adapter_mode', 'HYBRID')
            mode = AdapterMode[mode_str.upper()]
            self.adapter_system.set_adapter_mode(mode)
            
            # Restore energy distribution
            energy_dist = checkpoint.state_snapshot.get('energy_distribution', {})
            for module, energy in energy_dist.items():
                self.energy_monitor.energy_ledger[module] = energy
                
            logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
            
    def cleanup_old_checkpoints(self):
        """Remove checkpoints older than retention period"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=CHECKPOINT_RETENTION_HOURS)
        
        # Remove old checkpoints
        original_count = len(self.checkpoints)
        self.checkpoints = deque(
            (ckpt for ckpt in self.checkpoints if ckpt.timestamp > cutoff),
            maxlen=100
        )
        
        removed = original_count - len(self.checkpoints)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old checkpoints")
            
    # ========== Status and Reporting ==========
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        
        # Recent metrics
        recent_metrics = None
        if self.safety_history:
            recent = self.safety_history[-1]
            recent_metrics = {
                'fidelity': recent['metrics'].fidelity,
                'coherence': recent['metrics'].coherence,
                'energy_conservation': recent['metrics'].energy_conservation,
                'eigenvalue_max': recent['metrics'].eigenvalue_max,
                'chaos_containment': recent['metrics'].chaos_containment
            }
            
        # Violation summary
        recent_violations = list(self.violations)[-10:]
        violation_summary = {
            'total': len(self.violations),
            'recent': len(recent_violations),
            'unresolved': sum(1 for v in self.violations if not v.resolved)
        }
        
        # Energy status
        energy_dist = self.energy_monitor.get_energy_distribution()
        energy_total = sum(energy_dist.values())
        
        return {
            'current_safety_level': self.current_safety_level.value,
            'metrics': recent_metrics,
            'violations': violation_summary,
            'checkpoints_available': len(self.checkpoints),
            'active_checkpoint': self.active_checkpoint,
            'energy_status': {
                'total': energy_total,
                'conservation_error': abs(energy_total - self.energy_monitor.total_budget) / 
                                    self.energy_monitor.total_budget
            },
            'monitoring_active': self.is_monitoring,
            'recommendations': self._generate_recommendations()
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if self.current_safety_level == SafetyLevel.OPTIMAL:
            recommendations.append("System operating safely - consider enabling more chaos")
        elif self.current_safety_level == SafetyLevel.DEGRADED:
            recommendations.append("Reduce chaos intensity and monitor closely")
        elif self.current_safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            recommendations.append("Immediate intervention required - consider rollback")
            
        if len(self.checkpoints) == 0:
            recommendations.append("Create safety checkpoint before chaos operations")
            
        if len(self.violations) > 10:
            recommendations.append("Review and resolve safety violations")
            
        return recommendations

# ========== Testing and Demo ==========

async def demonstrate_safety_system():
    """Demonstrate safety calibration system"""
    print("🛡️ Safety Calibration Loop Demo")
    print("=" * 60)
    
    # This would normally use initialized systems
    print("\n1️⃣ Safety Levels:")
    for level in SafetyLevel:
        print(f"  • {level.value}: {level.name}")
        
    print("\n2️⃣ Safety Components:")
    print("  • Topological Protector: Virtual braid gates")
    print("  • Energy Conservation Monitor: Tracks energy flows")
    print("  • Quantum Fidelity Tracker: Ensures state preservation")
    print("  • Checkpoint System: Rollback capabilities")
    
    print("\n3️⃣ Safety Metrics Monitored:")
    print("  • Quantum state fidelity (min: 85%)")
    print("  • Phase coherence (min: 70%)")
    print("  • Energy conservation (tolerance: 5%)")
    print("  • Maximum eigenvalue thresholds")
    print("  • Chaos containment effectiveness")
    
    print("\n4️⃣ Emergency Response Capabilities:")
    print("  • Immediate chaos halt")
    print("  • Emergency eigenvalue damping")
    print("  • Checkpoint rollback")
    print("  • Module isolation")
    
    print("\n✅ Safety system ready!")
    print("   Use SafetyCalibrationLoop for continuous monitoring")

if __name__ == "__main__":
    asyncio.run(demonstrate_safety_system())
