#!/usr/bin/env python3
"""
Complete refactor of topology_policy.py to be config-driven
Loads all parameters from YAML configuration
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class TopologyPolicy:
    """Policy engine for dynamic topology switching - fully config-driven"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._init_from_config()
        
        # Runtime state
        self.current_state = "idle"
        self.metrics_history = []
        self.last_switch_time = None
        self.switch_count = 0
        self.decision_history = []
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback locations"""
        if config_path is None:
            # Search for config in standard locations
            search_paths = [
                Path("conf/soliton_memory_config.yaml"),
                Path("conf/lattice_config.yaml"),
                Path("../conf/soliton_memory_config.yaml"),
                Path("../conf/lattice_config.yaml"),
                Path.home() / ".tori" / "soliton_config.yaml",
            ]
            
            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded topology policy config from {config_path}")
                return config
        else:
            logger.warning("No config file found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration structure"""
        return {
            'topology': {
                'policy': {
                    'enabled': True,
                    'min_switch_interval': 60,
                    'decision_window': 300,  # 5 minutes of metrics
                    'hysteresis_factor': 0.1,  # Prevent flapping
                    
                    'states': {
                        'idle': {
                            'preferred_topology': 'kagome',
                            'thresholds': {
                                'memory_density': {'min': 0.0, 'max': 0.3},
                                'soliton_count': {'min': 0, 'max': 500},
                                'activity_rate': {'min': 0.0, 'max': 0.2},
                                'avg_comfort': {'min': 0.0, 'max': 0.3}
                            },
                            'transitions': {
                                'active': {
                                    'all_of': ['memory_density > 0.3', 'activity_rate > 0.2']
                                }
                            }
                        },
                        'active': {
                            'preferred_topology': 'hexagonal',
                            'thresholds': {
                                'memory_density': {'min': 0.3, 'max': 0.7},
                                'soliton_count': {'min': 500, 'max': 2000},
                                'activity_rate': {'min': 0.2, 'max': 0.6}
                            },
                            'transitions': {
                                'idle': {
                                    'all_of': ['memory_density < 0.2', 'activity_rate < 0.1']
                                },
                                'intensive': {
                                    'any_of': ['memory_density > 0.7', 'soliton_count > 2000']
                                }
                            }
                        },
                        'intensive': {
                            'preferred_topology': 'square',
                            'thresholds': {
                                'memory_density': {'min': 0.7, 'max': 1.0},
                                'soliton_count': {'min': 2000, 'max': 10000},
                                'activity_rate': {'min': 0.5, 'max': 1.0}
                            },
                            'transitions': {
                                'active': {
                                    'all_of': ['memory_density < 0.6', 'soliton_count < 1800']
                                },
                                'emergency': {
                                    'any_of': ['avg_comfort > 0.8', 'soliton_count > 9000']
                                }
                            }
                        },
                        'consolidating': {
                            'preferred_topology': 'small_world',
                            'thresholds': {},
                            'transitions': {
                                'idle': {'manual': True}
                            }
                        },
                        'emergency': {
                            'preferred_topology': 'kagome',  # Fall back to stable
                            'thresholds': {},
                            'transitions': {
                                'active': {
                                    'all_of': ['avg_comfort < 0.5', 'soliton_count < 5000']
                                }
                            }
                        }
                    },
                    
                    'comfort_overrides': {
                        'high_stress_threshold': 0.7,
                        'topology_downgrades': {
                            'square': 'hexagonal',
                            'hexagonal': 'kagome',
                            'small_world': 'kagome'
                        }
                    },
                    
                    'fallback_topology': 'kagome'
                }
            }
        }
    
    def _init_from_config(self):
        """Initialize policy engine from configuration"""
        policy_config = self.config.get('topology', {}).get('policy', {})
        
        self.enabled = policy_config.get('enabled', True)
        self.min_switch_interval = policy_config.get('min_switch_interval', 60)
        self.decision_window = policy_config.get('decision_window', 300)
        self.hysteresis_factor = policy_config.get('hysteresis_factor', 0.1)
        
        # Build state machine from config
        self.states = {}
        states_config = policy_config.get('states', {})
        
        for state_name, state_config in states_config.items():
            self.states[state_name] = {
                'preferred_topology': state_config.get('preferred_topology', 'kagome'),
                'thresholds': state_config.get('thresholds', {}),
                'transitions': self._build_transitions(state_config.get('transitions', {}))
            }
        
        # Comfort overrides
        comfort_config = policy_config.get('comfort_overrides', {})
        self.high_stress_threshold = comfort_config.get('high_stress_threshold', 0.7)
        self.topology_downgrades = comfort_config.get('topology_downgrades', {})
        
        self.fallback_topology = policy_config.get('fallback_topology', 'kagome')
        
        logger.info(f"Initialized topology policy with {len(self.states)} states")
    
    def _build_transitions(self, transitions_config: Dict) -> Dict:
        """Build transition conditions from config"""
        transitions = {}
        
        for target_state, conditions in transitions_config.items():
            if 'manual' in conditions and conditions['manual']:
                transitions[target_state] = {'type': 'manual'}
            elif 'all_of' in conditions:
                transitions[target_state] = {
                    'type': 'all_of',
                    'conditions': [self._parse_condition(c) for c in conditions['all_of']]
                }
            elif 'any_of' in conditions:
                transitions[target_state] = {
                    'type': 'any_of',
                    'conditions': [self._parse_condition(c) for c in conditions['any_of']]
                }
        
        return transitions
    
    def _parse_condition(self, condition_str: str) -> Callable:
        """Parse a condition string into a callable function"""
        # Simple parser for conditions like "memory_density > 0.3"
        parts = condition_str.strip().split()
        if len(parts) != 3:
            logger.warning(f"Invalid condition: {condition_str}")
            return lambda m: True
        
        metric, op, value = parts
        value = float(value)
        
        if op == '>':
            return lambda metrics: metrics.get(metric, 0) > value
        elif op == '<':
            return lambda metrics: metrics.get(metric, 0) < value
        elif op == '>=':
            return lambda metrics: metrics.get(metric, 0) >= value
        elif op == '<=':
            return lambda metrics: metrics.get(metric, 0) <= value
        elif op == '==':
            return lambda metrics: metrics.get(metric, 0) == value
        else:
            logger.warning(f"Unknown operator: {op}")
            return lambda m: True
    
    def update_metrics(self, lattice) -> Dict[str, float]:
        """Calculate current system metrics from lattice"""
        # Calculate metrics
        total_memories = len(lattice.memories)
        active_memories = len([m for m in lattice.memories.values() if m.access_count > 0])
        active_oscillators = len([o for o in getattr(lattice, 'oscillators', []) if o.get('active', True)])
        
        metrics = {
            'memory_density': total_memories / max(1, getattr(lattice, 'creation_count', 10000)),
            'soliton_count': active_oscillators,
            'activity_rate': active_memories / max(1, total_memories),
            'avg_comfort': self._calculate_avg_comfort(lattice),
            'total_memories': total_memories,
            'active_memories': active_memories
        }
        
        # Add to history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
        
        # Trim history to decision window
        cutoff = datetime.now() - timedelta(seconds=self.decision_window)
        self.metrics_history = [h for h in self.metrics_history if h['timestamp'] > cutoff]
        
        return metrics
    
    def _calculate_avg_comfort(self, lattice) -> float:
        """Calculate average comfort (stress) across all memories"""
        if not lattice.memories:
            return 0.5
        
        total_stress = sum(m.comfort_metrics.stress for m in lattice.memories.values())
        return total_stress / len(lattice.memories)
    
    def decide_topology(self, current_topology: str, metrics: Dict[str, float]) -> Optional[str]:
        """Decide if topology should change based on state and metrics"""
        if not self.enabled:
            return None
        
        # Check minimum interval
        if self.last_switch_time:
            elapsed = (datetime.now() - self.last_switch_time).total_seconds()
            if elapsed < self.min_switch_interval:
                return None
        
        # Check for state transitions
        new_state = self._check_state_transitions(metrics)
        if new_state != self.current_state:
            logger.info(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state
        
        # Get preferred topology for current state
        preferred = self.states[self.current_state]['preferred_topology']
        
        # Apply comfort overrides
        if metrics.get('avg_comfort', 0) > self.high_stress_threshold:
            if preferred in self.topology_downgrades:
                downgraded = self.topology_downgrades[preferred]
                logger.warning(f"High stress detected ({metrics['avg_comfort']:.2f}), "
                             f"downgrading topology: {preferred} -> {downgraded}")
                preferred = downgraded
        
        # Apply hysteresis to prevent flapping
        if preferred != current_topology:
            if self._should_switch_with_hysteresis(current_topology, preferred, metrics):
                logger.info(f"Topology change decided: {current_topology} -> {preferred}")
                self.last_switch_time = datetime.now()
                self.switch_count += 1
                
                # Record decision
                self.decision_history.append({
                    'timestamp': datetime.now(),
                    'from': current_topology,
                    'to': preferred,
                    'state': self.current_state,
                    'metrics': metrics.copy()
                })
                
                return preferred
        
        return None
    
    def _check_state_transitions(self, metrics: Dict[str, float]) -> str:
        """Check if we should transition to a new state"""
        current_transitions = self.states[self.current_state].get('transitions', {})
        
        for target_state, transition in current_transitions.items():
            if transition['type'] == 'manual':
                continue  # Skip manual transitions
            
            conditions = transition['conditions']
            
            if transition['type'] == 'all_of':
                if all(cond(metrics) for cond in conditions):
                    return target_state
            elif transition['type'] == 'any_of':
                if any(cond(metrics) for cond in conditions):
                    return target_state
        
        return self.current_state
    
    def _should_switch_with_hysteresis(self, current: str, preferred: str, 
                                      metrics: Dict[str, float]) -> bool:
        """Apply hysteresis to prevent rapid switching"""
        # Look at recent history
        recent_metrics = [h['metrics'] for h in self.metrics_history[-10:]]
        if not recent_metrics:
            return True
        
        # Check if metrics are stable
        for metric_name in ['memory_density', 'activity_rate', 'soliton_count']:
            values = [m.get(metric_name, 0) for m in recent_metrics]
            if values:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                if mean_val > 0 and std_dev / mean_val > self.hysteresis_factor:
                    logger.debug(f"Metric {metric_name} too volatile for switch "
                               f"(CV={std_dev/mean_val:.2f})")
                    return False
        
        return True
    
    def force_state(self, state: str):
        """Manually force a state transition (e.g., for consolidating)"""
        if state in self.states:
            logger.info(f"Manually forcing state transition to {state}")
            self.current_state = state
        else:
            logger.warning(f"Unknown state: {state}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current policy status"""
        return {
            'enabled': self.enabled,
            'current_state': self.current_state,
            'switch_count': self.switch_count,
            'last_switch': self.last_switch_time.isoformat() if self.last_switch_time else None,
            'metrics_history_size': len(self.metrics_history),
            'recent_decisions': self.decision_history[-5:]  # Last 5 decisions
        }
    
    def reload_config(self, config_path: Optional[str] = None):
        """Reload configuration without losing state"""
        logger.info("Reloading topology policy configuration")
        self.config = self._load_config(config_path)
        self._init_from_config()
