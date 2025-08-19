#!/usr/bin/env python3
"""
Fix for blowup_harness.py - Add comprehensive safety checks and runaway detection
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
import psutil
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class SafetyConfig:
    """Configuration for safety thresholds"""
    max_amplitude: float = 10.0
    max_energy: float = 1000.0
    max_oscillator_count: int = 100000
    max_memory_usage_mb: float = 8000.0
    max_cpu_percent: float = 90.0
    runaway_detection_window: int = 60  # seconds
    runaway_growth_threshold: float = 2.0  # 2x growth
    emergency_brake_threshold: float = 0.9  # 90% of limits
    
    # Actions
    on_amplitude_blowup: str = "dampen"  # dampen, reset, halt
    on_energy_blowup: str = "redistribute"
    on_oscillator_blowup: str = "prune"
    on_runaway_detected: str = "emergency_brake"
    on_memory_exceeded: str = "garbage_collect"
    on_cpu_exceeded: str = "throttle"

class BlowupHarness:
    """
    Safety harness to prevent system blowup and runaway conditions
    Monitors system health and takes corrective actions
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics history for runaway detection
        self.amplitude_history = deque(maxlen=60)  # 1 minute at 1Hz
        self.energy_history = deque(maxlen=60)
        self.oscillator_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.cpu_history = deque(maxlen=60)
        
        # State tracking
        self.emergency_brake_active = False
        self.throttle_factor = 1.0  # 1.0 = normal, 0.5 = half speed
        self.violations = []
        self.actions_taken = []
        
    def start_monitoring(self, system):
        """Start the safety monitoring thread"""
        if self.is_monitoring:
            logger.warning("Blowup harness already monitoring")
            return
        
        self.is_monitoring = True
        self.system = system
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Blowup harness monitoring started")
    
    def stop_monitoring(self):
        """Stop the safety monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Blowup harness monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store in history
                self._update_history(metrics)
                
                # Check for violations
                violations = self._check_violations(metrics)
                
                # Check for runaway conditions
                runaway = self._detect_runaway()
                
                # Take corrective actions
                if violations or runaway:
                    self._take_corrective_actions(violations, runaway, metrics)
                
                # Adjust throttle based on system load
                self._adjust_throttle(metrics)
                
                # Sleep (adjusted by throttle)
                time.sleep(1.0 * self.throttle_factor)
                
            except Exception as e:
                logger.error(f"Blowup harness error: {e}")
                time.sleep(5.0)  # Backoff on error
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        # Amplitude metrics
        if hasattr(self.system, 'lattice') and hasattr(self.system.lattice, 'oscillators'):
            amplitudes = [osc.get('amplitude', 0) for osc in self.system.lattice.oscillators 
                         if osc.get('active', True)]
            metrics['max_amplitude'] = max(amplitudes) if amplitudes else 0
            metrics['avg_amplitude'] = np.mean(amplitudes) if amplitudes else 0
            metrics['oscillator_count'] = len([o for o in self.system.lattice.oscillators 
                                             if o.get('active', True)])
        else:
            metrics['max_amplitude'] = 0
            metrics['avg_amplitude'] = 0
            metrics['oscillator_count'] = 0
        
        # Energy metrics
        if hasattr(self.system, 'total_energy'):
            metrics['total_energy'] = self.system.total_energy
        elif hasattr(self.system, 'lattice') and hasattr(self.system.lattice, 'total_charge'):
            metrics['total_energy'] = self.system.lattice.total_charge
        else:
            metrics['total_energy'] = 0
        
        # Memory metrics
        if hasattr(self.system, 'memory_entries'):
            metrics['memory_count'] = len(self.system.memory_entries)
        else:
            metrics['memory_count'] = 0
        
        # System resource metrics
        process = psutil.Process()
        metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        metrics['cpu_percent'] = process.cpu_percent(interval=0.1)
        
        metrics['timestamp'] = time.time()
        
        return metrics
    
    def _update_history(self, metrics: Dict[str, float]):
        """Update metrics history"""
        self.amplitude_history.append(metrics['max_amplitude'])
        self.energy_history.append(metrics['total_energy'])
        self.oscillator_history.append(metrics['oscillator_count'])
        self.memory_history.append(metrics['memory_usage_mb'])
        self.cpu_history.append(metrics['cpu_percent'])
    
    def _check_violations(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for threshold violations"""
        violations = []
        
        # Amplitude violation
        if metrics['max_amplitude'] > self.config.max_amplitude:
            violations.append({
                'type': 'amplitude',
                'value': metrics['max_amplitude'],
                'threshold': self.config.max_amplitude,
                'severity': 'critical' if metrics['max_amplitude'] > self.config.max_amplitude * 2 else 'warning'
            })
        
        # Energy violation
        if metrics['total_energy'] > self.config.max_energy:
            violations.append({
                'type': 'energy',
                'value': metrics['total_energy'],
                'threshold': self.config.max_energy,
                'severity': 'critical' if metrics['total_energy'] > self.config.max_energy * 2 else 'warning'
            })
        
        # Oscillator count violation
        if metrics['oscillator_count'] > self.config.max_oscillator_count:
            violations.append({
                'type': 'oscillator_count',
                'value': metrics['oscillator_count'],
                'threshold': self.config.max_oscillator_count,
                'severity': 'warning'
            })
        
        # Memory usage violation
        if metrics['memory_usage_mb'] > self.config.max_memory_usage_mb:
            violations.append({
                'type': 'memory',
                'value': metrics['memory_usage_mb'],
                'threshold': self.config.max_memory_usage_mb,
                'severity': 'critical'
            })
        
        # CPU usage violation
        if metrics['cpu_percent'] > self.config.max_cpu_percent:
            violations.append({
                'type': 'cpu',
                'value': metrics['cpu_percent'],
                'threshold': self.config.max_cpu_percent,
                'severity': 'warning'
            })
        
        return violations
    
    def _detect_runaway(self) -> Optional[Dict[str, Any]]:
        """Detect runaway growth conditions"""
        if len(self.amplitude_history) < 10:
            return None
        
        runaway_info = {}
        
        # Check amplitude runaway
        recent_amp = list(self.amplitude_history)[-10:]
        old_amp = list(self.amplitude_history)[-60:-50] if len(self.amplitude_history) >= 60 else recent_amp[:1]
        
        if old_amp and recent_amp:
            amp_growth = np.mean(recent_amp) / max(0.001, np.mean(old_amp))
            if amp_growth > self.config.runaway_growth_threshold:
                runaway_info['amplitude_growth'] = amp_growth
        
        # Check energy runaway
        recent_energy = list(self.energy_history)[-10:]
        old_energy = list(self.energy_history)[-60:-50] if len(self.energy_history) >= 60 else recent_energy[:1]
        
        if old_energy and recent_energy:
            energy_growth = np.mean(recent_energy) / max(0.001, np.mean(old_energy))
            if energy_growth > self.config.runaway_growth_threshold:
                runaway_info['energy_growth'] = energy_growth
        
        # Check oscillator count runaway
        recent_osc = list(self.oscillator_history)[-10:]
        old_osc = list(self.oscillator_history)[-60:-50] if len(self.oscillator_history) >= 60 else recent_osc[:1]
        
        if old_osc and recent_osc:
            osc_growth = np.mean(recent_osc) / max(1, np.mean(old_osc))
            if osc_growth > self.config.runaway_growth_threshold:
                runaway_info['oscillator_growth'] = osc_growth
        
        return runaway_info if runaway_info else None
    
    def _take_corrective_actions(self, violations: List[Dict], runaway: Optional[Dict], metrics: Dict):
        """Take corrective actions based on violations and runaway conditions"""
        timestamp = datetime.now()
        
        # Handle runaway first (most critical)
        if runaway:
            logger.warning(f"Runaway condition detected: {runaway}")
            self._handle_runaway(runaway, metrics)
            self.actions_taken.append({
                'timestamp': timestamp,
                'type': 'runaway',
                'details': runaway,
                'action': self.config.on_runaway_detected
            })
        
        # Handle violations
        for violation in violations:
            logger.warning(f"Safety violation: {violation['type']} = {violation['value']:.2f} "
                         f"(threshold: {violation['threshold']})")
            
            action_taken = self._handle_violation(violation, metrics)
            
            self.actions_taken.append({
                'timestamp': timestamp,
                'type': violation['type'],
                'value': violation['value'],
                'threshold': violation['threshold'],
                'action': action_taken
            })
        
        # Trim action history
        if len(self.actions_taken) > 1000:
            self.actions_taken = self.actions_taken[-1000:]
    
    def _handle_violation(self, violation: Dict, metrics: Dict) -> str:
        """Handle a specific violation"""
        vtype = violation['type']
        
        if vtype == 'amplitude':
            return self._handle_amplitude_blowup(metrics)
        elif vtype == 'energy':
            return self._handle_energy_blowup(metrics)
        elif vtype == 'oscillator_count':
            return self._handle_oscillator_blowup(metrics)
        elif vtype == 'memory':
            return self._handle_memory_exceeded(metrics)
        elif vtype == 'cpu':
            return self._handle_cpu_exceeded(metrics)
        
        return "unknown"
    
    def _handle_amplitude_blowup(self, metrics: Dict) -> str:
        """Handle amplitude blowup"""
        action = self.config.on_amplitude_blowup
        
        if action == "dampen":
            # Dampen all high amplitude oscillators
            if hasattr(self.system, 'lattice') and hasattr(self.system.lattice, 'oscillators'):
                dampen_factor = 0.5
                for osc in self.system.lattice.oscillators:
                    if osc.get('amplitude', 0) > self.config.max_amplitude:
                        osc['amplitude'] *= dampen_factor
                logger.info(f"Dampened high amplitude oscillators by {dampen_factor}")
            
        elif action == "reset":
            # Reset all amplitudes to safe values
            if hasattr(self.system, 'lattice') and hasattr(self.system.lattice, 'oscillators'):
                for osc in self.system.lattice.oscillators:
                    osc['amplitude'] = min(osc.get('amplitude', 0), 1.0)
                logger.info("Reset all amplitudes to safe values")
            
        elif action == "halt":
            # Emergency stop
            self.emergency_brake_active = True
            logger.critical("Emergency brake activated due to amplitude blowup")
        
        return action
    
    def _handle_energy_blowup(self, metrics: Dict) -> str:
        """Handle energy blowup"""
        action = self.config.on_energy_blowup
        
        if action == "redistribute":
            # Redistribute energy more evenly
            if hasattr(self.system, 'lattice'):
                total_energy = metrics['total_energy']
                num_oscillators = metrics['oscillator_count']
                if num_oscillators > 0:
                    avg_energy = min(total_energy / num_oscillators, 1.0)
                    if hasattr(self.system.lattice, 'oscillators'):
                        for osc in self.system.lattice.oscillators:
                            if osc.get('active', True):
                                osc['amplitude'] = np.sqrt(avg_energy)
                    logger.info("Redistributed energy across oscillators")
        
        return action
    
    def _handle_oscillator_blowup(self, metrics: Dict) -> str:
        """Handle too many oscillators"""
        action = self.config.on_oscillator_blowup
        
        if action == "prune":
            # Remove inactive or weak oscillators
            if hasattr(self.system, 'lattice') and hasattr(self.system.lattice, 'oscillators'):
                pruned = 0
                oscillators = self.system.lattice.oscillators
                for i in range(len(oscillators)-1, -1, -1):
                    if not oscillators[i].get('active', True) or oscillators[i].get('amplitude', 0) < 0.01:
                        oscillators[i]['active'] = False
                        oscillators[i]['amplitude'] = 0
                        pruned += 1
                        if metrics['oscillator_count'] - pruned <= self.config.max_oscillator_count * 0.9:
                            break
                logger.info(f"Pruned {pruned} weak oscillators")
        
        return action
    
    def _handle_memory_exceeded(self, metrics: Dict) -> str:
        """Handle memory limit exceeded"""
        action = self.config.on_memory_exceeded
        
        if action == "garbage_collect":
            import gc
            gc.collect()
            logger.info("Forced garbage collection")
        
        return action
    
    def _handle_cpu_exceeded(self, metrics: Dict) -> str:
        """Handle CPU limit exceeded"""
        action = self.config.on_cpu_exceeded
        
        if action == "throttle":
            # Increase throttle factor to slow down processing
            self.throttle_factor = min(self.throttle_factor * 1.5, 5.0)
            logger.info(f"Increased throttle factor to {self.throttle_factor}")
        
        return action
    
    def _handle_runaway(self, runaway: Dict, metrics: Dict):
        """Handle runaway condition"""
        action = self.config.on_runaway_detected
        
        if action == "emergency_brake":
            self.emergency_brake_active = True
            logger.critical("EMERGENCY BRAKE ACTIVATED - Runaway condition detected")
            
            # Take immediate action to stop growth
            if 'amplitude_growth' in runaway:
                self._handle_amplitude_blowup(metrics)
            if 'energy_growth' in runaway:
                self._handle_energy_blowup(metrics)
            if 'oscillator_growth' in runaway:
                self._handle_oscillator_blowup(metrics)
    
    def _adjust_throttle(self, metrics: Dict):
        """Adjust throttle based on system load"""
        # Gradually reduce throttle if system is stable
        if not self.emergency_brake_active and self.throttle_factor > 1.0:
            self.throttle_factor = max(1.0, self.throttle_factor * 0.95)
        
        # Check if we're approaching limits
        load_factors = []
        
        if self.config.max_amplitude > 0:
            load_factors.append(metrics['max_amplitude'] / self.config.max_amplitude)
        if self.config.max_energy > 0:
            load_factors.append(metrics['total_energy'] / self.config.max_energy)
        if self.config.max_oscillator_count > 0:
            load_factors.append(metrics['oscillator_count'] / self.config.max_oscillator_count)
        
        max_load = max(load_factors) if load_factors else 0
        
        # If approaching limits, increase throttle
        if max_load > self.config.emergency_brake_threshold:
            self.throttle_factor = min(self.throttle_factor * 1.2, 5.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        recent_actions = self.actions_taken[-10:] if self.actions_taken else []
        
        return {
            'is_monitoring': self.is_monitoring,
            'emergency_brake_active': self.emergency_brake_active,
            'throttle_factor': self.throttle_factor,
            'recent_violations': len([a for a in recent_actions if a['type'] != 'runaway']),
            'recent_runaway_detections': len([a for a in recent_actions if a['type'] == 'runaway']),
            'recent_actions': recent_actions,
            'current_metrics': {
                'max_amplitude': self.amplitude_history[-1] if self.amplitude_history else 0,
                'total_energy': self.energy_history[-1] if self.energy_history else 0,
                'oscillator_count': self.oscillator_history[-1] if self.oscillator_history else 0,
                'memory_usage_mb': self.memory_history[-1] if self.memory_history else 0,
                'cpu_percent': self.cpu_history[-1] if self.cpu_history else 0
            }
        }
    
    def reset_emergency_brake(self):
        """Reset emergency brake (use with caution)"""
        self.emergency_brake_active = False
        self.throttle_factor = 1.0
        logger.info("Emergency brake reset")
