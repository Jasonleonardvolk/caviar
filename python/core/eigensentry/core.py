#!/usr/bin/env python3
"""
EigenSentry 2.0 - Dynamic Stability Conductor
Transforms from stability guardrail to orchestrator of productive chaos

Based on the comprehensive upgrade plan and incorporating insights from 
2023-2025 chaos computing research.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from enum import Enum
from collections import defaultdict, deque
import asyncio
import json
from pathlib import Path

# Import existing components
from python.core.unified_metacognitive_integration import (
    MetacognitiveState, CognitiveStateManager
)
from python.core.cognitive_dynamics_monitor import (
    DynamicsState, DynamicsMetrics
)

# Import BdG monitoring
try:
    from alan_backend.lyap_exporter import update_watchlist, get_lyapunov_exporter
    BDG_AVAILABLE = True
except ImportError:
    logger.warning("BdG monitoring not available")
    BDG_AVAILABLE = False

logger = logging.getLogger(__name__)

# ========== Configuration Constants ==========

# Adaptive thresholds based on research showing edge-of-chaos sweet spots
EIGENVALUE_STABLE_THRESHOLD = 1.0
EIGENVALUE_SOFT_MARGIN = 1.3  # Allow growth up to 30% above threshold
EIGENVALUE_EMERGENCY_THRESHOLD = 2.0  # Hard cutoff

# Energy budget constants (inspired by credit-based systems)
MAX_ENERGY_CREDITS = 1000
INITIAL_CREDIT_ALLOCATION = 100
CREDIT_REFRESH_RATE = 10  # Credits per second

# Timing windows for chaos events (nanosecond precision from research)
INSTABILITY_TIMING_WINDOW_NS = 100_000_000  # 100ms window
BURST_COORDINATION_LEAD_TIME_NS = 10_000_000  # 10ms advance notice

# ========== Core Data Structures ==========

class InstabilityType(Enum):
    """Types of instability events we can orchestrate"""
    SOLITON_FISSION = "soliton_fission"
    SOLITON_FUSION = "soliton_fusion"
    ATTRACTOR_HOP = "attractor_hop"
    RESONANCE_BURST = "resonance_burst"
    PHASE_EXPLOSION = "phase_explosion"
    CHAOTIC_SEARCH = "chaotic_search"

@dataclass
class InstabilityEvent:
    """Represents a detected or planned instability event"""
    event_type: InstabilityType
    eigenvalues: np.ndarray
    mode_shapes: Optional[np.ndarray]
    growth_rate: float
    predicted_time_ns: int
    energy_required: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationSignal:
    """Signal sent to other modules to prepare for chaos event"""
    event: InstabilityEvent
    target_modules: List[str]
    preparation_actions: Dict[str, str]
    timestamp_ns: int

# ========== Energy Budget Broker ==========

class EnergyBudgetBroker:
    """
    Manages energy credits for modules entering chaotic states
    Based on research showing 3-16x energy efficiency at edge-of-chaos
    """
    
    def __init__(self, max_credits: int = MAX_ENERGY_CREDITS):
        self.max_credits = max_credits
        self._credits = defaultdict(lambda: INITIAL_CREDIT_ALLOCATION)
        self._usage_history = defaultdict(deque)
        self._efficiency_scores = defaultdict(lambda: 1.0)
        
    def request(self, module_id: str, joule_tau: int) -> bool:
        """
        Request energy credits to enter CCL (Chaos Control Layer)
        Returns True if approved, False if denied
        """
        # Check current balance
        if self._credits[module_id] < joule_tau:
            logger.warning(f"Module {module_id} denied {joule_tau} credits (balance: {self._credits[module_id]})")
            return False
            
        # Check efficiency score - modules that use chaos well get priority
        if self._efficiency_scores[module_id] < 0.5 and joule_tau > 50:
            logger.warning(f"Module {module_id} denied due to low efficiency score: {self._efficiency_scores[module_id]}")
            return False
            
        # Approve and deduct
        self._credits[module_id] -= joule_tau
        self._usage_history[module_id].append({
            'timestamp': datetime.now(timezone.utc),
            'amount': joule_tau,
            'balance_after': self._credits[module_id]
        })
        
        # Trim history
        if len(self._usage_history[module_id]) > 1000:
            self._usage_history[module_id].popleft()
            
        return True
        
    def refund(self, module_id: str, joule_tau: int, success: bool = True):
        """
        Refund credits after chaos event completes
        Success flag affects efficiency score
        """
        refund_amount = joule_tau if success else joule_tau // 2
        self._credits[module_id] = min(
            self._credits[module_id] + refund_amount, 
            self.max_credits
        )
        
        # Update efficiency score
        old_score = self._efficiency_scores[module_id]
        if success:
            self._efficiency_scores[module_id] = min(1.0, old_score * 1.1)
        else:
            self._efficiency_scores[module_id] = max(0.1, old_score * 0.9)
            
    def refresh_credits(self, delta_time_s: float):
        """Called periodically to refresh credits"""
        refresh_amount = int(CREDIT_REFRESH_RATE * delta_time_s)
        for module_id in self._credits:
            self._credits[module_id] = min(
                self._credits[module_id] + refresh_amount,
                self.max_credits
            )
            
    def get_module_stats(self, module_id: str) -> Dict[str, Any]:
        """Get statistics for a module's energy usage"""
        return {
            'current_credits': self._credits[module_id],
            'efficiency_score': self._efficiency_scores[module_id],
            'recent_usage': list(self._usage_history[module_id])[-10:]
        }

# ========== Eigenvalue Monitor & Analyzer ==========

class EigenvalueMonitor:
    """
    Monitors system eigenvalues and detects productive instabilities
    Uses insights from Lyapunov monitoring research
    """
    
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        self.eigenvalue_history = deque(maxlen=1000)
        self.mode_shape_cache = {}
        
    def compute_eigenspectrum(self, jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of system Jacobian
        Uses efficient algorithms for real-time monitoring
        """
        # For large systems, use sparse methods or approximations
        if jacobian.shape[0] > 50:
            # Use Arnoldi iteration for largest eigenvalues
            from scipy.sparse.linalg import eigs
            k = min(10, jacobian.shape[0] - 2)
            eigenvalues, eigenvectors = eigs(jacobian, k=k, which='LR')
            # Pad with zeros for consistency
            eigenvalues = np.pad(eigenvalues, (0, self.state_dim - k), mode='constant')
            eigenvectors = np.pad(eigenvectors, ((0, 0), (0, self.state_dim - k)), mode='constant')
        else:
            eigenvalues, eigenvectors = np.linalg.eig(jacobian)
            
        # Sort by real part (growth rate)
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
        
    def analyze_instability(self, eigenvalues: np.ndarray, 
                          eigenvectors: np.ndarray) -> Optional[InstabilityEvent]:
        """
        Analyze eigenspectrum to detect and classify instabilities
        """
        max_real = np.max(np.real(eigenvalues))
        
        # Check if we're in the soft margin zone
        if EIGENVALUE_STABLE_THRESHOLD < max_real <= EIGENVALUE_SOFT_MARGIN:
            # Classify the type of instability
            dominant_idx = np.argmax(np.real(eigenvalues))
            dominant_mode = eigenvectors[:, dominant_idx]
            
            event_type = self._classify_instability_type(
                eigenvalues, dominant_mode, dominant_idx
            )
            
            # Estimate growth rate and timing
            growth_rate = np.real(eigenvalues[dominant_idx])
            time_to_threshold = self._estimate_time_to_threshold(growth_rate)
            
            # Estimate energy requirement based on mode shape
            energy_required = self._estimate_energy_requirement(
                dominant_mode, growth_rate, event_type
            )
            
            return InstabilityEvent(
                event_type=event_type,
                eigenvalues=eigenvalues,
                mode_shapes=eigenvectors,
                growth_rate=growth_rate,
                predicted_time_ns=int(time_to_threshold * 1e9),
                energy_required=energy_required,
                metadata={
                    'dominant_frequency': np.imag(eigenvalues[dominant_idx]),
                    'mode_participation': np.abs(dominant_mode)
                }
            )
            
        return None
        
    def _classify_instability_type(self, eigenvalues: np.ndarray, 
                                 mode: np.ndarray, idx: int) -> InstabilityType:
        """Classify instability based on eigenvalue patterns"""
        
        # Check for complex conjugate pairs (oscillatory instability)
        imag_part = np.imag(eigenvalues[idx])
        if abs(imag_part) > 0.1:
            # High frequency suggests resonance burst
            if abs(imag_part) > 10:
                return InstabilityType.RESONANCE_BURST
            # Low frequency suggests attractor hop
            else:
                return InstabilityType.ATTRACTOR_HOP
                
        # Check mode shape for soliton signatures
        mode_abs = np.abs(mode)
        if np.max(mode_abs) / np.mean(mode_abs) > 5:
            # Localized mode suggests soliton dynamics
            n_peaks = np.sum(np.diff(np.sign(np.diff(mode_abs))) < 0)
            if n_peaks == 1:
                return InstabilityType.SOLITON_FISSION
            elif n_peaks > 1:
                return InstabilityType.SOLITON_FUSION
                
        # Check for global desynchronization
        if np.std(np.angle(mode)) > np.pi/4:
            return InstabilityType.PHASE_EXPLOSION
            
        # Default to chaotic search
        return InstabilityType.CHAOTIC_SEARCH
        
    def _estimate_time_to_threshold(self, growth_rate: float) -> float:
        """Estimate time until instability reaches action threshold"""
        if growth_rate <= 0:
            return float('inf')
            
        # Simple exponential growth model
        current_amplitude = 1.0  # Normalized
        threshold_amplitude = 2.0  # When action needed
        
        time_s = np.log(threshold_amplitude / current_amplitude) / growth_rate
        return max(0.0, time_s)
        
    def _estimate_energy_requirement(self, mode: np.ndarray, 
                                   growth_rate: float,
                                   event_type: InstabilityType) -> int:
        """Estimate energy credits needed for this instability"""
        base_energy = {
            InstabilityType.SOLITON_FISSION: 50,
            InstabilityType.SOLITON_FUSION: 40,
            InstabilityType.ATTRACTOR_HOP: 60,
            InstabilityType.RESONANCE_BURST: 80,
            InstabilityType.PHASE_EXPLOSION: 100,
            InstabilityType.CHAOTIC_SEARCH: 30
        }
        
        # Scale by growth rate and mode complexity
        energy = base_energy.get(event_type, 50)
        energy *= (1 + growth_rate)
        energy *= (1 + np.std(np.abs(mode)))
        
        return int(energy)

# ========== Coordination Engine ==========

class CoordinationEngine:
    """
    Coordinates chaos events across modules
    Implements the conductor pattern from the upgrade plan
    """
    
    def __init__(self):
        self.registered_modules = {}
        self.active_events = {}
        self.coordination_history = deque(maxlen=1000)
        
    def register_module(self, module_id: str, 
                       prepare_callback: Callable,
                       complete_callback: Callable):
        """Register a module for coordination"""
        self.registered_modules[module_id] = {
            'prepare': prepare_callback,
            'complete': complete_callback,
            'state': 'idle'
        }
        
    async def coordinate_event(self, event: InstabilityEvent, 
                             participating_modules: List[str]) -> bool:
        """
        Coordinate a chaos event across modules
        Returns True if successful, False if aborted
        """
        event_id = f"{event.event_type.value}_{datetime.now(timezone.utc).timestamp()}"
        
        # Create coordination signal
        signal = CoordinationSignal(
            event=event,
            target_modules=participating_modules,
            preparation_actions=self._generate_preparation_actions(event),
            timestamp_ns=int(datetime.now(timezone.utc).timestamp() * 1e9)
        )
        
        # Phase 1: Notify all modules to prepare
        prepare_tasks = []
        for module_id in participating_modules:
            if module_id in self.registered_modules:
                module = self.registered_modules[module_id]
                module['state'] = 'preparing'
                prepare_tasks.append(
                    self._prepare_module(module_id, signal)
                )
                
        # Wait for all preparations with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*prepare_tasks, return_exceptions=True),
                timeout=BURST_COORDINATION_LEAD_TIME_NS / 1e9
            )
            
            # Check if any module failed to prepare
            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    logger.error(f"Module {participating_modules[i]} failed to prepare: {result}")
                    await self._abort_event(event_id, participating_modules)
                    return False
                    
        except asyncio.TimeoutError:
            logger.error("Preparation timeout - aborting chaos event")
            await self._abort_event(event_id, participating_modules)
            return False
            
        # Phase 2: Execute the chaos event
        self.active_events[event_id] = {
            'event': event,
            'modules': participating_modules,
            'start_time': datetime.now(timezone.utc)
        }
        
        # Let the chaos unfold (monitored by other systems)
        await asyncio.sleep(event.predicted_time_ns / 1e9)
        
        # Phase 3: Complete and cleanup
        complete_tasks = []
        for module_id in participating_modules:
            if module_id in self.registered_modules:
                module = self.registered_modules[module_id]
                module['state'] = 'completing'
                complete_tasks.append(
                    self._complete_module(module_id, event_id)
                )
                
        await asyncio.gather(*complete_tasks, return_exceptions=True)
        
        # Record in history
        self.coordination_history.append({
            'event_id': event_id,
            'event_type': event.event_type.value,
            'modules': participating_modules,
            'success': True,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Cleanup
        del self.active_events[event_id]
        for module_id in participating_modules:
            if module_id in self.registered_modules:
                self.registered_modules[module_id]['state'] = 'idle'
                
        return True
        
    async def _prepare_module(self, module_id: str, 
                            signal: CoordinationSignal) -> bool:
        """Prepare a single module for chaos event"""
        try:
            callback = self.registered_modules[module_id]['prepare']
            return await callback(signal)
        except Exception as e:
            logger.error(f"Module {module_id} preparation failed: {e}")
            return False
            
    async def _complete_module(self, module_id: str, event_id: str) -> bool:
        """Complete chaos event for a single module"""
        try:
            callback = self.registered_modules[module_id]['complete']
            return await callback(event_id)
        except Exception as e:
            logger.error(f"Module {module_id} completion failed: {e}")
            return False
            
    async def _abort_event(self, event_id: str, modules: List[str]):
        """Abort a chaos event and restore stability"""
        logger.warning(f"Aborting chaos event {event_id}")
        
        abort_tasks = []
        for module_id in modules:
            if module_id in self.registered_modules:
                # Call complete with abort flag
                callback = self.registered_modules[module_id]['complete']
                abort_tasks.append(callback(event_id, abort=True))
                
        await asyncio.gather(*abort_tasks, return_exceptions=True)
        
    def _generate_preparation_actions(self, event: InstabilityEvent) -> Dict[str, str]:
        """Generate module-specific preparation actions"""
        actions = {}
        
        if event.event_type == InstabilityType.SOLITON_FISSION:
            actions['memory_lattice'] = 'increase_plasticity'
            actions['oscillator_core'] = 'prepare_burst_mode'
            actions['ghost_memory'] = 'record_current_state'
            
        elif event.event_type == InstabilityType.ATTRACTOR_HOP:
            actions['memory_lattice'] = 'unlock_attractors'
            actions['oscillator_core'] = 'frequency_sweep_ready'
            actions['dynamics_monitor'] = 'increase_sampling_rate'
            
        elif event.event_type == InstabilityType.PHASE_EXPLOSION:
            actions['oscillator_core'] = 'desync_prepare'
            actions['ghost_memory'] = 'strengthen_coupling'
            actions['reflection_system'] = 'pause_iterations'
            
        # Add common actions
        actions['all'] = 'checkpoint_state'
        
        return actions

# ========== Main EigenSentry 2.0 Class ==========

class EigenSentry2:
    """
    EigenSentry 2.0 - Symphony Conductor for Productive Chaos
    
    No longer a simple stability guard, but an orchestrator that:
    1. Monitors eigenvalues with soft margins
    2. Manages energy budgets for chaos events  
    3. Coordinates module preparation and recovery
    4. Shapes instabilities toward productive outcomes
    """
    
    def __init__(self, state_manager: CognitiveStateManager,
                 enable_chaos: bool = True):
        self.state_manager = state_manager
        self.enable_chaos = enable_chaos
        
        # Core components
        self.energy_broker = EnergyBudgetBroker()
        self.eigenvalue_monitor = EigenvalueMonitor(state_manager.state_dim)
        self.coordination_engine = CoordinationEngine()
        
        # State tracking
        self.current_eigenvalues = None
        self.stability_state = DynamicsState.STABLE
        self.last_refresh_time = datetime.now(timezone.utc)
        
        # Damping system (for emergency use only)
        self.emergency_damping_factor = 0.5
        self.is_emergency_damping = False
        
        # Virtual braid gates for topology protection
        self.braid_gates = {}
        
        # Phase-energy ledger
        self.phase_energy_ledger = defaultdict(lambda: {
            'phase': 0.0,
            'energy': 0.0,
            'locked': False
        })
        
    async def monitor_and_conduct(self, jacobian: np.ndarray, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main monitoring and conducting loop
        Returns status and any coordination actions taken
        """
        # Refresh energy credits periodically
        now = datetime.now(timezone.utc)
        delta_time = (now - self.last_refresh_time).total_seconds()
        if delta_time > 1.0:
            self.energy_broker.refresh_credits(delta_time)
            self.last_refresh_time = now
            
        # Compute eigenspectrum
        eigenvalues, eigenvectors = self.eigenvalue_monitor.compute_eigenspectrum(jacobian)
        self.current_eigenvalues = eigenvalues
        
        # Update BdG stability monitoring if available
        lambda_max = 0.0
        if BDG_AVAILABLE and state is not None:
            try:
                lambda_max = update_watchlist(state)
                logger.debug(f"BdG lambda_max: {lambda_max}")
            except Exception as e:
                logger.warning(f"BdG monitoring failed: {e}")
        
        # Store in history
        self.eigenvalue_monitor.eigenvalue_history.append({
            'timestamp': now,
            'eigenvalues': eigenvalues,
            'max_real': np.max(np.real(eigenvalues))
        })
        
        # Check for emergency conditions
        max_real = np.max(np.real(eigenvalues))
        if max_real > EIGENVALUE_EMERGENCY_THRESHOLD:
            return await self._handle_emergency(eigenvalues, eigenvectors)
            
        # Analyze for productive instabilities
        instability_event = self.eigenvalue_monitor.analyze_instability(
            eigenvalues, eigenvectors
        )
        
        if instability_event and self.enable_chaos:
            # Orchestrate the chaos event
            return await self._orchestrate_chaos(instability_event)
        else:
            # Normal stable operation
            return {
                'state': 'stable',
                'max_eigenvalue': max_real,
                'energy_available': self._get_total_available_energy(),
                'actions_taken': []
            }
            
    async def _orchestrate_chaos(self, event: InstabilityEvent) -> Dict[str, Any]:
        """Orchestrate a productive chaos event"""
        logger.info(f"Orchestrating {event.event_type.value} event")
        
        # Determine participating modules based on event type
        modules = self._select_participating_modules(event)
        
        # Check energy budget
        total_energy_needed = event.energy_required
        approved_modules = []
        
        for module in modules:
            if self.energy_broker.request(module, event.energy_required // len(modules)):
                approved_modules.append(module)
                
        if len(approved_modules) < len(modules) // 2:
            # Not enough energy available - abort
            logger.warning("Insufficient energy for chaos event - aborting")
            # Refund the approved modules
            for module in approved_modules:
                self.energy_broker.refund(module, event.energy_required // len(modules), False)
            return {
                'state': 'stable',
                'max_eigenvalue': np.max(np.real(event.eigenvalues)),
                'energy_available': self._get_total_available_energy(),
                'actions_taken': ['chaos_aborted_low_energy']
            }
            
        # Coordinate the event
        success = await self.coordination_engine.coordinate_event(event, approved_modules)
        
        # Update energy credits based on outcome
        for module in approved_modules:
            self.energy_broker.refund(
                module, 
                event.energy_required // len(modules),
                success
            )
            
        return {
            'state': event.event_type.value,
            'max_eigenvalue': np.max(np.real(event.eigenvalues)),
            'energy_available': self._get_total_available_energy(),
            'actions_taken': [f'orchestrated_{event.event_type.value}'],
            'participating_modules': approved_modules,
            'success': success
        }
        
    async def _handle_emergency(self, eigenvalues: np.ndarray, 
                              eigenvectors: np.ndarray) -> Dict[str, Any]:
        """Handle emergency instability with hard damping"""
        logger.error(f"EMERGENCY: Max eigenvalue {np.max(np.real(eigenvalues)):.3f} exceeds threshold")
        
        # Apply emergency damping
        self.is_emergency_damping = True
        
        # Get current state and apply strong damping
        current_state = self.state_manager.get_state()
        damped_state = current_state * self.emergency_damping_factor
        
        # Update state with emergency metadata
        self.state_manager.update_state(
            new_state=damped_state,
            metadata={
                'emergency_damping': True,
                'pre_damping_eigenvalue': np.max(np.real(eigenvalues)),
                'damping_factor': self.emergency_damping_factor
            }
        )
        
        # Notify all modules of emergency
        emergency_signal = CoordinationSignal(
            event=InstabilityEvent(
                event_type=InstabilityType.PHASE_EXPLOSION,
                eigenvalues=eigenvalues,
                mode_shapes=eigenvectors,
                growth_rate=np.max(np.real(eigenvalues)),
                predicted_time_ns=0,
                energy_required=0
            ),
            target_modules=list(self.coordination_engine.registered_modules.keys()),
            preparation_actions={'all': 'emergency_shutdown'},
            timestamp_ns=int(datetime.now(timezone.utc).timestamp() * 1e9)
        )
        
        # Send emergency signal to all modules
        for module_id, module in self.coordination_engine.registered_modules.items():
            try:
                await module['prepare'](emergency_signal)
            except Exception as e:
                logger.error(f"Module {module_id} emergency response failed: {e}")
                
        return {
            'state': 'emergency_damped',
            'max_eigenvalue': np.max(np.real(eigenvalues)),
            'energy_available': 0,  # All energy diverted to stabilization
            'actions_taken': ['emergency_damping', 'global_notification'],
            'damping_applied': self.emergency_damping_factor
        }
        
    def _select_participating_modules(self, event: InstabilityEvent) -> List[str]:
        """Select which modules should participate in a chaos event"""
        # Base selection on event type
        module_sets = {
            InstabilityType.SOLITON_FISSION: [
                'memory_lattice', 'ghost_memory', 'oscillator_core'
            ],
            InstabilityType.SOLITON_FUSION: [
                'memory_lattice', 'soliton_memory', 'phase_controller'
            ],
            InstabilityType.ATTRACTOR_HOP: [
                'memory_lattice', 'dynamics_monitor', 'oscillator_core'
            ],
            InstabilityType.RESONANCE_BURST: [
                'oscillator_core', 'energy_manager', 'memory_lattice'
            ],
            InstabilityType.PHASE_EXPLOSION: [
                'oscillator_core', 'ghost_memory', 'reflection_system'
            ],
            InstabilityType.CHAOTIC_SEARCH: [
                'memory_lattice', 'reflection_system', 'dynamics_monitor'
            ]
        }
        
        base_modules = module_sets.get(event.event_type, [])
        
        # Filter to only registered modules
        available_modules = [
            m for m in base_modules 
            if m in self.coordination_engine.registered_modules
        ]
        
        return available_modules
        
    def _get_total_available_energy(self) -> int:
        """Get total energy credits available across all modules"""
        return sum(self.energy_broker._credits.values())
        
    # ========== Module Registration Interface ==========
    
    def register_module(self, module_id: str,
                       prepare_callback: Callable,
                       complete_callback: Callable):
        """Register a module for chaos coordination"""
        self.coordination_engine.register_module(
            module_id, prepare_callback, complete_callback
        )
        logger.info(f"Module {module_id} registered with EigenSentry 2.0")
        
    # ========== Virtual Braid Gate API ==========
    
    def enter_ccl(self, module_id: str, requested_energy: int) -> Optional[str]:
        """
        Enter the Chaos Control Layer (CCL)
        Returns gate_id if approved, None if denied
        """
        if not self.energy_broker.request(module_id, requested_energy):
            return None
            
        gate_id = f"gate_{module_id}_{datetime.now(timezone.utc).timestamp()}"
        self.braid_gates[gate_id] = {
            'module_id': module_id,
            'entry_time': datetime.now(timezone.utc),
            'energy_allocated': requested_energy,
            'active': True
        }
        
        return gate_id
        
    def exit_ccl(self, gate_id: str, success: bool = True):
        """Exit the Chaos Control Layer"""
        if gate_id not in self.braid_gates:
            logger.error(f"Unknown gate_id: {gate_id}")
            return
            
        gate = self.braid_gates[gate_id]
        if not gate['active']:
            logger.warning(f"Gate {gate_id} already closed")
            return
            
        # Refund remaining energy
        self.energy_broker.refund(
            gate['module_id'],
            gate['energy_allocated'] // 2,  # Partial refund
            success
        )
        
        gate['active'] = False
        gate['exit_time'] = datetime.now(timezone.utc)
        
    # ========== Damping Control ==========
    
    def apply_shaped_damping(self, state: np.ndarray, 
                           damping_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply shaped damping that preserves productive oscillations
        Only used when soft margin is exceeded
        """
        if damping_profile is None:
            # Default profile: stronger damping for high-frequency modes
            fft = np.fft.fft(state)
            freqs = np.fft.fftfreq(len(state))
            
            # Create frequency-dependent damping
            damping_profile = np.ones_like(freqs)
            high_freq_mask = np.abs(freqs) > 0.3
            damping_profile[high_freq_mask] = 0.7  # Damp high frequencies more
            
            # Apply damping in frequency domain
            fft_damped = fft * damping_profile
            return np.real(np.fft.ifft(fft_damped))
        else:
            # Apply provided damping profile
            return state * damping_profile
            
    # ========== Status and Diagnostics ==========
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of EigenSentry 2.0"""
        recent_history = list(self.eigenvalue_monitor.eigenvalue_history)[-10:]
        
        return {
            'stability_state': self.stability_state.value,
            'is_emergency_damping': self.is_emergency_damping,
            'current_max_eigenvalue': (
                np.max(np.real(self.current_eigenvalues)) 
                if self.current_eigenvalues is not None else 0.0
            ),
            'energy_stats': {
                'total_available': self._get_total_available_energy(),
                'module_breakdown': dict(self.energy_broker._credits),
                'efficiency_scores': dict(self.energy_broker._efficiency_scores)
            },
            'active_chaos_events': len(self.coordination_engine.active_events),
            'registered_modules': list(self.coordination_engine.registered_modules.keys()),
            'active_braid_gates': sum(1 for g in self.braid_gates.values() if g['active']),
            'recent_eigenvalue_trend': [
                h['max_real'] for h in recent_history
            ] if recent_history else []
        }
        
    def get_module_stats(self, module_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific module"""
        return self.energy_broker.get_module_stats(module_id)


# ========== Testing and Demo ==========

async def demonstrate_eigensentry2():
    """Demonstrate EigenSentry 2.0 capabilities"""
    print("🎼 EigenSentry 2.0 - Symphony Conductor Demo")
    print("=" * 60)
    
    # Create state manager
    state_manager = CognitiveStateManager(state_dim=10)
    
    # Initialize EigenSentry 2.0
    eigen_sentry = EigenSentry2(state_manager, enable_chaos=True)
    
    # Register mock modules
    async def mock_prepare(signal):
        print(f"  Module preparing for {signal.event.event_type.value}")
        await asyncio.sleep(0.1)
        return True
        
    async def mock_complete(event_id, abort=False):
        print(f"  Module completing event {event_id} (abort={abort})")
        return True
        
    for module in ['memory_lattice', 'oscillator_core', 'ghost_memory']:
        eigen_sentry.register_module(module, mock_prepare, mock_complete)
    
    # Test 1: Stable operation
    print("\n1️⃣ Testing stable operation...")
    stable_jacobian = np.eye(10) * 0.5  # All eigenvalues = 0.5
    result = await eigen_sentry.monitor_and_conduct(stable_jacobian)
    print(f"  Result: {result['state']}, Max eigenvalue: {result['max_eigenvalue']:.3f}")
    
    # Test 2: Soft margin chaos event
    print("\n2️⃣ Testing orchestrated chaos...")
    unstable_jacobian = np.eye(10)
    unstable_jacobian[0, 0] = 1.2  # One eigenvalue in soft margin
    unstable_jacobian[0, 1] = 0.3  # Add coupling for interesting dynamics
    
    result = await eigen_sentry.monitor_and_conduct(unstable_jacobian)
    print(f"  Result: {result['state']}, Actions: {result.get('actions_taken', [])}")
    
    # Test 3: Energy budget
    print("\n3️⃣ Testing energy budget system...")
    print(f"  Total energy available: {eigen_sentry._get_total_available_energy()}")
    
    gate_id = eigen_sentry.enter_ccl('memory_lattice', 50)
    if gate_id:
        print(f"  Memory lattice entered CCL with gate {gate_id}")
        eigen_sentry.exit_ccl(gate_id, success=True)
        print("  Successfully exited CCL")
    
    # Test 4: Emergency damping
    print("\n4️⃣ Testing emergency damping...")
    emergency_jacobian = np.eye(10) * 2.5  # Way over threshold
    result = await eigen_sentry.monitor_and_conduct(emergency_jacobian)
    print(f"  Result: {result['state']}, Damping applied: {result.get('damping_applied', 0)}")
    
    # Final status
    print("\n📊 Final Status:")
    status = eigen_sentry.get_status()
    print(f"  Stability state: {status['stability_state']}")
    print(f"  Energy available: {status['energy_stats']['total_available']}")
    print(f"  Registered modules: {status['registered_modules']}")

if __name__ == "__main__":
    asyncio.run(demonstrate_eigensentry2())
