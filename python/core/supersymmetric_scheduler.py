#!/usr/bin/env python3
"""
Supersymmetric Scheduler - Enhanced Autonomous BPS Control System

PURPOSE:
    Advanced BPS-aware supervisor that orchestrates compute cycles, topology transitions,
    memory harvesting, and phase control with autonomous behavior capabilities.
    
ENHANCEMENTS:
    - Autonomous energy monitoring and target-based actions
    - Automatic topology swap triggers based on configurable conditions
    - Energy ramping and gradual transition capabilities
    - Background monitoring with safety mechanisms
    - Configurable autonomous behaviors with manual overrides
    
AUTHOR: TORI System Enhancement
LAST UPDATED: 2025-01-26
"""

import logging
import time
import threading
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from datetime import datetime, timedelta

# Import BPS topology functions
try:
    from python.core.bps_topology import (
        bps_energy_harvest, 
        compute_topological_charge,
        bps_topology_transition,
        verify_bps_conservation,
        bps_stability_check,
        interpolated_bps_transition,
        EnergyBundle
    )
    BPS_TOPOLOGY_AVAILABLE = True
except ImportError:
    logging.warning("BPS topology not available - using mock functions")
    BPS_TOPOLOGY_AVAILABLE = False
    
    # Mock functions for fallback
    class EnergyBundle:
        def __init__(self, energy=0.0, Q=0, soliton_data=None):
            self.energy = energy
            self.Q = Q
            self.soliton_data = soliton_data or []
    
    def bps_energy_harvest(memory, lattice, efficiency=0.95):
        return EnergyBundle()
    
    def compute_topological_charge(memory):
        return 0
    
    def bps_topology_transition(lattice, new_laplacian, memory, energy_bundle):
        return True
    
    def verify_bps_conservation(memory, energy_bundle):
        return {'charge_conserved': True, 'bps_bound_satisfied': True}
    
    def bps_stability_check(laplacian, memory):
        return True
    
    def interpolated_bps_transition(lattice, new_laplacian, memory, energy_bundle, steps=10):
        return True

# Import diagnostics
try:
    from python.monitoring.bps_diagnostics import BPSDiagnostics
except ImportError:
    logging.warning("BPS diagnostics not available - using mock")
    
    class BPSDiagnostics:
        def __init__(self, memory):
            self.memory = memory
        
        def record_state(self, label=""):
            return {"Lagrangian_OK": True, "Q": 0, "E": 0.0}
        
        def clear(self):
            pass

logger = logging.getLogger("SupersymmetricScheduler")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration and Types
# ═══════════════════════════════════════════════════════════════════════════════

class AutomationMode(Enum):
    """Automation levels for the scheduler"""
    MANUAL = "manual"              # No automation, manual control only
    MONITORING = "monitoring"      # Monitor only, log alerts
    SEMI_AUTO = "semi_auto"        # Auto-trigger with confirmations
    FULL_AUTO = "full_auto"        # Fully autonomous operation

class TriggerCondition(Enum):
    """Types of trigger conditions"""
    ENERGY_THRESHOLD = "energy_threshold"
    CHARGE_DRIFT = "charge_drift"
    STABILITY_LOSS = "stability_loss"
    TIME_INTERVAL = "time_interval"
    CONSERVATION_VIOLATION = "conservation_violation"

@dataclass
class AutonomousConfig:
    """Configuration for autonomous behaviors"""
    
    # Main automation toggle
    enabled: bool = True
    mode: AutomationMode = AutomationMode.MONITORING
    
    # Energy management
    enable_energy_monitoring: bool = True
    target_energy: Optional[float] = None
    energy_threshold_low: float = 0.1
    energy_threshold_high: float = 10.0
    enable_energy_ramping: bool = True
    energy_ramp_rate: float = 0.1  # energy units per cycle
    
    # Topology management
    enable_topology_swaps: bool = True
    topology_swap_threshold: float = 2.0  # charge drift threshold
    enable_gradual_transitions: bool = True
    transition_steps: int = 10
    
    # Stability monitoring
    enable_stability_monitoring: bool = True
    stability_check_interval: int = 10  # cycles
    auto_stability_correction: bool = True
    
    # Safety mechanisms
    max_auto_actions_per_hour: int = 10
    require_confirmation_above_energy: float = 5.0
    emergency_stop_threshold: float = 20.0
    enable_manual_override: bool = True
    
    # Monitoring intervals
    monitoring_interval: float = 1.0  # seconds
    background_monitoring: bool = True
    
    # Logging and diagnostics
    detailed_logging: bool = True
    save_diagnostics: bool = True
    diagnostics_path: str = "logs/scheduler_diagnostics.json"
    
    # Advanced features
    enable_predictive_actions: bool = False
    learning_mode: bool = False
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'AutonomousConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file"""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")

@dataclass
class SystemState:
    """Current system state tracking"""
    current_energy: float = 0.0
    current_charge: int = 0
    last_harvest_time: Optional[datetime] = None
    last_topology_swap: Optional[datetime] = None
    stability_score: float = 1.0
    conservation_violations: int = 0
    autonomous_actions_taken: int = 0
    emergency_stop_active: bool = False
    manual_override_active: bool = False
    
    def reset_hourly_counters(self):
        """Reset counters that track hourly limits"""
        self.autonomous_actions_taken = 0

@dataclass
class TriggerRule:
    """Definition of an autonomous trigger rule"""
    name: str
    condition: TriggerCondition
    threshold: float
    action: str
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger"""
        if not self.enabled:
            return False
        if self.last_triggered is None:
            return True
        return datetime.now() - self.last_triggered > timedelta(minutes=self.cooldown_minutes)
    
    def mark_triggered(self):
        """Mark this rule as having been triggered"""
        self.last_triggered = datetime.now()

# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced SupersymmetricScheduler with Autonomous Capabilities
# ═══════════════════════════════════════════════════════════════════════════════

class SupersymmetricScheduler:
    """Enhanced BPS-aware supervisor with autonomous behavior capabilities"""
    
    def __init__(self, memory, oscillators, lattice, config: Optional[AutonomousConfig] = None):
        self.memory = memory
        self.oscillators = oscillators
        self.lattice = lattice
        self.tick = 0
        self.diagnostics = BPSDiagnostics(memory)
        
        # Autonomous configuration
        self.config = config or AutonomousConfig()
        self.state = SystemState()
        
        # Trigger rules
        self.trigger_rules = self._setup_default_triggers()
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._monitoring_active = False
        
        # Action callbacks
        self.action_callbacks: Dict[str, Callable] = {}
        self._setup_default_actions()
        
        # Metrics and history
        self.metrics_history = []
        self.action_history = []
        
        # Initialize diagnostics path
        if self.config.save_diagnostics:
            Path(self.config.diagnostics_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SupersymmetricScheduler initialized in {self.config.mode.value} mode")
        
        # Start background monitoring if enabled
        if self.config.background_monitoring and self.config.enabled:
            self.start_background_monitoring()
    
    def _setup_default_triggers(self) -> List[TriggerRule]:
        """Setup default trigger rules based on configuration"""
        rules = []
        
        # Energy thresholds
        if self.config.enable_energy_monitoring:
            rules.append(TriggerRule(
                name="low_energy_harvest",
                condition=TriggerCondition.ENERGY_THRESHOLD,
                threshold=self.config.energy_threshold_low,
                action="energy_harvest"
            ))
            
            rules.append(TriggerRule(
                name="high_energy_topology_swap",
                condition=TriggerCondition.ENERGY_THRESHOLD,
                threshold=self.config.energy_threshold_high,
                action="topology_swap"
            ))
        
        # Charge drift monitoring
        if self.config.enable_topology_swaps:
            rules.append(TriggerRule(
                name="charge_drift_correction",
                condition=TriggerCondition.CHARGE_DRIFT,
                threshold=self.config.topology_swap_threshold,
                action="drift_correction"
            ))
        
        # Stability monitoring
        if self.config.enable_stability_monitoring:
            rules.append(TriggerRule(
                name="stability_loss_correction",
                condition=TriggerCondition.STABILITY_LOSS,
                threshold=0.5,  # Stability score threshold
                action="stability_correction"
            ))
        
        return rules
    
    def _setup_default_actions(self):
        """Setup default action callbacks"""
        self.action_callbacks.update({
            "energy_harvest": self._action_energy_harvest,
            "topology_swap": self._action_topology_swap,
            "drift_correction": self._action_drift_correction,
            "stability_correction": self._action_stability_correction,
            "energy_ramp": self._action_energy_ramp,
            "emergency_stop": self._action_emergency_stop
        })
    
    def run_cycle(self):
        """Enhanced cycle with autonomous monitoring and actions"""
        self.tick += 1
        
        if self.config.detailed_logging:
            logger.info(f"[SCHED] Cycle {self.tick} (mode: {self.config.mode.value})")
        
        # Update system state
        self._update_system_state()
        
        # Check for emergency conditions
        if self._check_emergency_conditions():
            return
        
        # Step oscillators
        for o in self.oscillators:
            try:
                o.step()
            except Exception as e:
                logger.error(f"Oscillator step failed: {e}")
        
        # Validate topological state
        snap = self.diagnostics.record_state(label=f"tick_{self.tick}")
        
        # Process autonomous triggers if enabled
        if self.config.enabled and not self.state.manual_override_active:
            self._process_autonomous_triggers(snap)
        
        # Traditional auto-correct for backward compatibility
        if snap["Lagrangian_OK"] is False:
            logger.warning(f"[SCHED] Lagrangian drift detected: Q={snap['Q']} E={snap['E']:.3f}")
            if self.config.mode in [AutomationMode.SEMI_AUTO, AutomationMode.FULL_AUTO]:
                self.auto_resolve_drift()
            else:
                logger.info("[SCHED] Autonomous drift correction disabled - manual intervention required")
        
        # Save metrics if configured
        if self.config.save_diagnostics:
            self._save_cycle_metrics(snap)
        
        # Energy ramping if enabled and target is set
        if (self.config.enable_energy_ramping and 
            self.config.target_energy is not None and
            self.config.mode == AutomationMode.FULL_AUTO):
            self._perform_energy_ramping()
    
    def _update_system_state(self):
        """Update current system state metrics"""
        try:
            # Update charge
            self.state.current_charge = compute_topological_charge(self.memory)
            
            # Estimate current energy from solitons
            solitons = self.memory.get_active_solitons() if hasattr(self.memory, 'get_active_solitons') else []
            total_energy = 0.0
            for soliton in solitons:
                if hasattr(soliton, 'amplitude'):
                    total_energy += abs(soliton.amplitude)**2
                elif isinstance(soliton, dict):
                    total_energy += abs(soliton.get('amplitude', 1.0))**2
                else:
                    total_energy += 1.0
            self.state.current_energy = total_energy
            
            # Update stability score
            if BPS_TOPOLOGY_AVAILABLE:
                self.state.stability_score = 1.0 if bps_stability_check(self.lattice, self.memory) else 0.0
            
        except Exception as e:
            logger.error(f"Failed to update system state: {e}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        if self.state.current_energy > self.config.emergency_stop_threshold:
            logger.critical(f"EMERGENCY: Energy {self.state.current_energy} exceeds threshold {self.config.emergency_stop_threshold}")
            self._action_emergency_stop()
            return True
        
        if self.state.conservation_violations > 5:
            logger.critical(f"EMERGENCY: Too many conservation violations ({self.state.conservation_violations})")
            self._action_emergency_stop()
            return True
        
        return False
    
    def _process_autonomous_triggers(self, snap: Dict[str, Any]):
        """Process autonomous trigger rules"""
        if self.state.autonomous_actions_taken >= self.config.max_auto_actions_per_hour:
            if self.config.detailed_logging:
                logger.info("[AUTO] Hourly action limit reached - skipping autonomous actions")
            return
        
        for rule in self.trigger_rules:
            if not rule.can_trigger():
                continue
            
            triggered = False
            
            # Check trigger conditions
            if rule.condition == TriggerCondition.ENERGY_THRESHOLD:
                if rule.name.startswith("low_energy") and self.state.current_energy <= rule.threshold:
                    triggered = True
                elif rule.name.startswith("high_energy") and self.state.current_energy >= rule.threshold:
                    triggered = True
            
            elif rule.condition == TriggerCondition.CHARGE_DRIFT:
                # Check charge drift from expected values
                charge_drift = abs(self.state.current_charge - snap.get('Q', 0))
                if charge_drift >= rule.threshold:
                    triggered = True
            
            elif rule.condition == TriggerCondition.STABILITY_LOSS:
                if self.state.stability_score <= rule.threshold:
                    triggered = True
            
            elif rule.condition == TriggerCondition.CONSERVATION_VIOLATION:
                if not snap.get("Lagrangian_OK", True):
                    triggered = True
            
            if triggered:
                self._execute_autonomous_action(rule)
    
    def _execute_autonomous_action(self, rule: TriggerRule):
        """Execute an autonomous action based on trigger rule"""
        try:
            # Check if confirmation is required
            if (self.config.mode == AutomationMode.SEMI_AUTO or 
                (self.state.current_energy > self.config.require_confirmation_above_energy and 
                 self.config.mode != AutomationMode.FULL_AUTO)):
                
                logger.info(f"[AUTO] Action {rule.action} triggered by {rule.name} - confirmation required")
                # In a real implementation, this would trigger a UI confirmation
                # For now, we'll skip the action
                return
            
            logger.info(f"[AUTO] Executing action: {rule.action} (triggered by {rule.name})")
            
            # Execute the action
            if rule.action in self.action_callbacks:
                success = self.action_callbacks[rule.action]()
                
                if success:
                    rule.mark_triggered()
                    self.state.autonomous_actions_taken += 1
                    
                    # Log action
                    action_record = {
                        'timestamp': datetime.now().isoformat(),
                        'rule_name': rule.name,
                        'action': rule.action,
                        'trigger_condition': rule.condition.value,
                        'system_state': {
                            'energy': self.state.current_energy,
                            'charge': self.state.current_charge,
                            'stability': self.state.stability_score
                        }
                    }
                    self.action_history.append(action_record)
                    
                    logger.info(f"[AUTO] Action {rule.action} completed successfully")
                else:
                    logger.warning(f"[AUTO] Action {rule.action} failed")
            else:
                logger.error(f"[AUTO] Unknown action: {rule.action}")
                
        except Exception as e:
            logger.error(f"Failed to execute autonomous action {rule.action}: {e}")
    
    def _perform_energy_ramping(self):
        """Perform gradual energy ramping toward target"""
        if self.config.target_energy is None:
            return
        
        energy_diff = self.config.target_energy - self.state.current_energy
        
        if abs(energy_diff) < 0.01:  # Close enough to target
            return
        
        # Calculate ramp step
        ramp_step = min(abs(energy_diff), self.config.energy_ramp_rate)
        if energy_diff < 0:
            ramp_step = -ramp_step
        
        logger.info(f"[RAMP] Energy ramping: {self.state.current_energy:.3f} → {self.config.target_energy:.3f} (step: {ramp_step:+.3f})")
        
        # Perform energy adjustment (simplified - would need actual energy injection/extraction)
        if ramp_step > 0:
            # Need to add energy - trigger harvest
            self._action_energy_harvest()
        else:
            # Need to reduce energy - perform controlled dissipation
            # This would need actual implementation in the physical system
            logger.info("[RAMP] Energy reduction not yet implemented")

    
    # ═══════════════════════════════════════════════════════════════
    # Action Implementations
    # ═══════════════════════════════════════════════════════════════
    
    def _action_energy_harvest(self) -> bool:
        """Perform autonomous energy harvesting"""
        try:
            logger.info("[AUTO] Performing energy harvest...")
            
            if BPS_TOPOLOGY_AVAILABLE:
                energy_bundle = bps_energy_harvest(self.memory, self.lattice)
                
                # Verify conservation
                verification = verify_bps_conservation(self.memory, energy_bundle)
                if not verification.get('charge_conserved', False):
                    self.state.conservation_violations += 1
                    logger.warning("[AUTO] Energy harvest violated charge conservation")
                    return False
                
                self.state.last_harvest_time = datetime.now()
                logger.info(f"[AUTO] Energy harvest completed: E={energy_bundle.energy:.3f}, Q={energy_bundle.Q}")
                return True
            else:
                # Fallback for when BPS topology not available
                self.auto_resolve_drift()
                return True
                
        except Exception as e:
            logger.error(f"[AUTO] Energy harvest failed: {e}")
            return False
    
    def _action_topology_swap(self) -> bool:
        """Perform autonomous topology swap"""
        try:
            logger.info("[AUTO] Performing topology swap...")
            
            if not BPS_TOPOLOGY_AVAILABLE:
                logger.warning("[AUTO] BPS topology not available - cannot perform swap")
                return False
            
            # First harvest energy to prepare for transition
            energy_bundle = bps_energy_harvest(self.memory, self.lattice)
            
            # Generate a new topology (simplified - would use actual topology generator)
            # For now, we'll use the existing lattice as the "new" topology
            new_laplacian = self.lattice.copy() if hasattr(self.lattice, 'copy') else self.lattice
            
            # Perform transition
            if self.config.enable_gradual_transitions:
                success = interpolated_bps_transition(
                    self.lattice, 
                    new_laplacian, 
                    self.memory, 
                    energy_bundle, 
                    steps=self.config.transition_steps
                )
            else:
                success = bps_topology_transition(
                    self.lattice, 
                    new_laplacian, 
                    self.memory, 
                    energy_bundle
                )
            
            if success:
                self.state.last_topology_swap = datetime.now()
                logger.info("[AUTO] Topology swap completed successfully")
            else:
                logger.warning("[AUTO] Topology swap failed")
            
            return success
            
        except Exception as e:
            logger.error(f"[AUTO] Topology swap failed: {e}")
            return False
    
    def _action_drift_correction(self) -> bool:
        """Perform autonomous drift correction"""
        try:
            logger.info("[AUTO] Performing drift correction...")
            
            # Use the existing auto_resolve_drift method
            self.auto_resolve_drift()
            
            # Verify correction worked
            current_charge = compute_topological_charge(self.memory)
            if abs(current_charge - self.state.current_charge) < 0.1:
                logger.info("[AUTO] Drift correction successful")
                return True
            else:
                logger.warning("[AUTO] Drift correction may not have been effective")
                return False
                
        except Exception as e:
            logger.error(f"[AUTO] Drift correction failed: {e}")
            return False
    
    def _action_stability_correction(self) -> bool:
        """Perform autonomous stability correction"""
        try:
            logger.info("[AUTO] Performing stability correction...")
            
            # Perform energy harvest to stabilize system
            harvest_success = self._action_energy_harvest()
            
            # Check if stability improved
            if BPS_TOPOLOGY_AVAILABLE:
                stable = bps_stability_check(self.lattice, self.memory)
                if stable:
                    logger.info("[AUTO] Stability correction successful")
                    return True
                else:
                    # Try topology swap as second attempt
                    logger.info("[AUTO] Stability correction via energy harvest insufficient - trying topology swap")
                    return self._action_topology_swap()
            
            return harvest_success
            
        except Exception as e:
            logger.error(f"[AUTO] Stability correction failed: {e}")
            return False
    
    def _action_energy_ramp(self) -> bool:
        """Perform controlled energy ramping"""
        try:
            logger.info("[AUTO] Performing energy ramp...")
            
            if self.config.target_energy is None:
                logger.warning("[AUTO] Energy ramp requested but no target energy set")
                return False
            
            self._perform_energy_ramping()
            return True
            
        except Exception as e:
            logger.error(f"[AUTO] Energy ramp failed: {e}")
            return False
    
    def _action_emergency_stop(self) -> bool:
        """Perform emergency stop"""
        try:
            logger.critical("[EMERGENCY] Activating emergency stop!")
            
            self.state.emergency_stop_active = True
            
            # Stop background monitoring
            if self._monitoring_active:
                self.stop_background_monitoring()
            
            # Disable autonomous actions
            self.config.enabled = False
            
            # Stop all oscillators
            for oscillator in self.oscillators:
                if hasattr(oscillator, 'stop'):
                    oscillator.stop()
                elif hasattr(oscillator, 'active'):
                    oscillator.active = False
            
            logger.critical("[EMERGENCY] Emergency stop completed - system halted")
            return True
            
        except Exception as e:
            logger.error(f"[EMERGENCY] Emergency stop failed: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # Background Monitoring System
    # ═══════════════════════════════════════════════════════════════
    
    def start_background_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_active:
            logger.warning("Background monitoring already active")
            return
        
        logger.info("Starting background monitoring...")
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitoring_thread.start()
        self._monitoring_active = True
    
    def stop_background_monitoring(self):
        """Stop background monitoring thread"""
        if not self._monitoring_active:
            return
        
        logger.info("Stopping background monitoring...")
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        self._monitoring_active = False
    
    def _background_monitor(self):
        """Background monitoring loop"""
        logger.info("Background monitoring started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Update system state
                self._update_system_state()
                
                # Check for immediate action triggers
                if (self.config.enabled and 
                    not self.state.manual_override_active and 
                    not self.state.emergency_stop_active):
                    
                    snap = {
                        'Q': self.state.current_charge,
                        'E': self.state.current_energy,
                        'Lagrangian_OK': self.state.stability_score > 0.5
                    }
                    
                    self._process_autonomous_triggers(snap)
                
                # Save periodic diagnostics
                if self.config.save_diagnostics:
                    self._save_monitoring_metrics()
                
                # Reset hourly counters if needed
                current_time = datetime.now()
                if (not hasattr(self, '_last_hour_reset') or 
                    current_time.hour != self._last_hour_reset.hour):
                    self.state.reset_hourly_counters()
                    self._last_hour_reset = current_time
                
                # Sleep until next monitoring cycle
                self._stop_monitoring.wait(timeout=self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(1.0)  # Brief pause before retrying
        
        logger.info("Background monitoring stopped")
    
    def _save_cycle_metrics(self, snap: Dict[str, Any]):
        """Save cycle metrics to history"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cycle': self.tick,
                'energy': self.state.current_energy,
                'charge': self.state.current_charge,
                'stability': self.state.stability_score,
                'lagrangian_ok': snap.get('Lagrangian_OK', True),
                'autonomous_actions': self.state.autonomous_actions_taken,
                'conservation_violations': self.state.conservation_violations
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only recent history to prevent memory growth
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
                
        except Exception as e:
            logger.error(f"Failed to save cycle metrics: {e}")
    
    def _save_monitoring_metrics(self):
        """Save background monitoring metrics"""
        try:
            diagnostics_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'state': self.state.__dict__,
                'recent_metrics': self.metrics_history[-10:],
                'recent_actions': self.action_history[-10:]
            }
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            with open(self.config.diagnostics_path, 'w') as f:
                json.dump(diagnostics_data, f, indent=2, default=datetime_handler)
                
        except Exception as e:
            logger.error(f"Failed to save monitoring metrics: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # Public Control Interface
    # ═══════════════════════════════════════════════════════════════
    
    def set_target_energy(self, target: float):
        """Set target energy for autonomous ramping"""
        logger.info(f"Setting target energy: {target:.3f}")
        self.config.target_energy = target
    
    def enable_autonomous_mode(self, mode: AutomationMode = AutomationMode.FULL_AUTO):
        """Enable autonomous operation"""
        logger.info(f"Enabling autonomous mode: {mode.value}")
        self.config.enabled = True
        self.config.mode = mode
        self.state.manual_override_active = False
        
        if not self._monitoring_active and self.config.background_monitoring:
            self.start_background_monitoring()
    
    def disable_autonomous_mode(self):
        """Disable autonomous operation"""
        logger.info("Disabling autonomous mode")
        self.config.enabled = False
        self.state.manual_override_active = True
    
    def add_custom_trigger(self, rule: TriggerRule):
        """Add a custom trigger rule"""
        logger.info(f"Adding custom trigger: {rule.name}")
        self.trigger_rules.append(rule)
    
    def add_custom_action(self, name: str, callback: Callable[[], bool]):
        """Add a custom action callback"""
        logger.info(f"Adding custom action: {name}")
        self.action_callbacks[name] = callback
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'mode': self.config.mode.value,
            'enabled': self.config.enabled,
            'energy': self.state.current_energy,
            'charge': self.state.current_charge,
            'stability': self.state.stability_score,
            'emergency_stop': self.state.emergency_stop_active,
            'manual_override': self.state.manual_override_active,
            'actions_taken': self.state.autonomous_actions_taken,
            'conservation_violations': self.state.conservation_violations,
            'monitoring_active': self._monitoring_active,
            'target_energy': self.config.target_energy
        }
    
    def get_recent_actions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent autonomous actions taken"""
        return self.action_history[-limit:]
    
    def reset_counters(self):
        """Reset error and action counters"""
        logger.info("Resetting system counters")
        self.state.autonomous_actions_taken = 0
        self.state.conservation_violations = 0
    
    def shutdown(self):
        """Gracefully shutdown the scheduler"""
        logger.info("Shutting down SupersymmetricScheduler...")
        
        # Stop background monitoring
        if self._monitoring_active:
            self.stop_background_monitoring()
        
        # Save final diagnostics
        if self.config.save_diagnostics:
            self._save_monitoring_metrics()
        
        # Disable autonomous actions
        self.config.enabled = False
        
        logger.info("SupersymmetricScheduler shutdown complete")

    def auto_resolve_drift(self):
        """Legacy drift resolution method - maintained for backward compatibility"""
        logger.info("[SCHED] Resolving topological drift via harvesting...")
        if BPS_TOPOLOGY_AVAILABLE:
            bps_energy_harvest(self.memory, self.lattice)
        else:
            # Fallback behavior
            logger.warning("[SCHED] BPS topology not available - using basic drift resolution")

    def dispatch(self, task: str):
        """Enhanced dispatch with autonomous capabilities"""
        logger.info(f"[SCHED] Dispatching task: {task}")
        
        if task == "harvest":
            if self.config.enabled and self.config.mode == AutomationMode.FULL_AUTO:
                # Use autonomous harvest
                success = self._action_energy_harvest()
                logger.info(f"[SCHED] Autonomous harvest {'succeeded' if success else 'failed'}")
            else:
                # Use legacy method
                if BPS_TOPOLOGY_AVAILABLE:
                    bps_energy_harvest(self.memory, self.lattice)
                else:
                    self.auto_resolve_drift()
                    
        elif task == "diagnose":
            snap = self.diagnostics.record_state(label="manual_diagnose")
            if self.config.detailed_logging:
                logger.info(f"[SCHED] Diagnosis: Q={snap.get('Q', 0)}, E={snap.get('E', 0):.3f}, Stable={snap.get('Lagrangian_OK', True)}")
                
        elif task == "clear":
            self.diagnostics.clear()
            if self.config.save_diagnostics:
                # Also clear metrics history
                self.metrics_history.clear()
                self.action_history.clear()
                logger.info("[SCHED] Cleared diagnostics and metrics history")
                
        elif task == "topology_swap":
            if self.config.enabled:
                success = self._action_topology_swap()
                logger.info(f"[SCHED] Topology swap {'succeeded' if success else 'failed'}")
            else:
                logger.warning("[SCHED] Autonomous mode disabled - cannot perform topology swap")
                
        elif task == "enable_auto":
            self.enable_autonomous_mode()
            
        elif task == "disable_auto":
            self.disable_autonomous_mode()
            
        elif task == "status":
            status = self.get_system_status()
            logger.info(f"[SCHED] System status: {status}")
            
        elif task == "emergency_stop":
            self._action_emergency_stop()
            
        else:
            logger.warning(f"[SCHED] Unknown dispatch task: {task}")
            logger.info("[SCHED] Available tasks: harvest, diagnose, clear, topology_swap, enable_auto, disable_auto, status, emergency_stop")


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions and Examples
# ═══════════════════════════════════════════════════════════════════════════════

def create_autonomous_scheduler(memory, oscillators, lattice, 
                              config_path: Optional[Path] = None) -> SupersymmetricScheduler:
    """Factory function to create an autonomous scheduler with configuration"""
    
    # Load configuration if provided
    if config_path and config_path.exists():
        config = AutonomousConfig.load_from_file(config_path)
        logger.info(f"Loaded scheduler configuration from {config_path}")
    else:
        # Create default configuration optimized for autonomous operation
        config = AutonomousConfig(
            enabled=True,
            mode=AutomationMode.MONITORING,  # Start conservative
            enable_energy_monitoring=True,
            enable_topology_swaps=True,
            enable_stability_monitoring=True,
            background_monitoring=True,
            detailed_logging=True,
            save_diagnostics=True
        )
        
        # Save default config for future reference
        if config_path:
            config.save_to_file(config_path)
            logger.info(f"Saved default configuration to {config_path}")
    
    return SupersymmetricScheduler(memory, oscillators, lattice, config)

def create_example_config() -> AutonomousConfig:
    """Create an example configuration for autonomous operation"""
    return AutonomousConfig(
        # Main settings
        enabled=True,
        mode=AutomationMode.FULL_AUTO,
        
        # Energy management
        enable_energy_monitoring=True,
        target_energy=5.0,  # Set target for autonomous ramping
        energy_threshold_low=0.5,
        energy_threshold_high=15.0,
        enable_energy_ramping=True,
        energy_ramp_rate=0.2,
        
        # Topology management
        enable_topology_swaps=True,
        topology_swap_threshold=3.0,
        enable_gradual_transitions=True,
        transition_steps=15,
        
        # Safety and limits
        max_auto_actions_per_hour=20,
        require_confirmation_above_energy=10.0,
        emergency_stop_threshold=25.0,
        
        # Monitoring
        monitoring_interval=0.5,  # More frequent monitoring
        background_monitoring=True,
        detailed_logging=True,
        save_diagnostics=True
    )

if __name__ == "__main__":
    # Example usage and testing
    logger.info("SupersymmetricScheduler module loaded")
    logger.info("Enhanced features:")
    logger.info("  • Autonomous energy monitoring and ramping")
    logger.info("  • Automatic topology swap triggers")
    logger.info("  • Background monitoring with safety mechanisms")
    logger.info("  • Configurable automation levels")
    logger.info("  • Manual override and emergency stop")
    logger.info("  • Comprehensive diagnostics and logging")
    logger.info("")
    logger.info("Usage example:")
    logger.info("  config = create_example_config()")
    logger.info("  scheduler = SupersymmetricScheduler(memory, oscillators, lattice, config)")
    logger.info("  scheduler.set_target_energy(7.5)")
    logger.info("  scheduler.enable_autonomous_mode(AutomationMode.FULL_AUTO)")
