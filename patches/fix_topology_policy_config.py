#!/usr/bin/env python3
"""
Fix for topology_policy.py - Load configuration from YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class TopologyPolicy:
    """Policy engine for dynamic topology switching with config loading"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.current_state = "idle"
        self.metrics_history = []
        self.last_switch_time = None
        self.switch_count = 0
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._init_rules_from_config()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                Path("conf/soliton_memory_config.yaml"),
                Path("conf/lattice_config.yaml"),
                Path("../conf/soliton_memory_config.yaml"),
                Path("../conf/lattice_config.yaml"),
            ]
            
            for path in possible_paths:
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
        """Default configuration if no file is found"""
        return {
            'topology': {
                'policy': {
                    'min_switch_interval': 60,
                    'states': {
                        'idle': {
                            'preferred_topology': 'kagome',
                            'thresholds': {
                                'memory_density': 0.1,
                                'soliton_count': 100,
                                'activity_rate': 0.1
                            }
                        },
                        'active': {
                            'preferred_topology': 'hexagonal',
                            'thresholds': {
                                'memory_density': 0.7,
                                'soliton_count': 2000,
                                'activity_rate': 0.5
                            }
                        },
                        'intensive': {
                            'preferred_topology': 'square',
                            'thresholds': {
                                'memory_density': 0.9,
                                'soliton_count': 5000,
                                'activity_rate': 0.8
                            }
                        },
                        'consolidating': {
                            'preferred_topology': 'small_world'
                        }
                    },
                    'comfort_weight': 0.3,
                    'fallback_topology': 'kagome'
                }
            }
        }
    
    def _init_rules_from_config(self):
        """Initialize rules from loaded configuration"""
        self.rules = {}
        
        policy_config = self.config.get('topology', {}).get('policy', {})
        states_config = policy_config.get('states', {})
        
        # Build rules from config
        for state, state_config in states_config.items():
            thresholds = state_config.get('thresholds', {})
            
            self.rules[state] = {
                'check': self._create_check_function(thresholds),
                'preferred_topology': state_config.get('preferred_topology', 'kagome')
            }
        
        # Store other policy parameters
        self.min_switch_interval = policy_config.get('min_switch_interval', 60)
        self.comfort_weight = policy_config.get('comfort_weight', 0.3)
        self.fallback_topology = policy_config.get('fallback_topology', 'kagome')
        
        logger.info(f"Initialized topology policy with {len(self.rules)} states")
    
    def _create_check_function(self, thresholds: Dict[str, float]):
        """Create a check function based on thresholds"""
        def check(metrics):
            for metric, threshold in thresholds.items():
                if metric in metrics and metrics[metric] <= threshold:
                    return False
            return True
        return check
    
    def update_metrics(self, lattice) -> Dict[str, float]:
        """Calculate current system metrics"""
        metrics = {
            'memory_density': len(lattice.memories) / lattice.creation_count if lattice.creation_count > 0 else 0,
            'soliton_count': len([o for o in lattice.oscillators if o.get('active', True)]),
            'activity_rate': sum(1 for m in lattice.memories.values() if m.access_count > 0) / max(1, len(lattice.memories)),
            'avg_comfort': np.mean([m.comfort_metrics.stress for m in lattice.memories.values()]) if lattice.memories else 0.5
        }
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def decide_topology(self, current_topology: str, metrics: Dict[str, float]) -> Optional[str]:
        """Decide if topology should change based on state and metrics"""
        
        # Check if enough time has passed since last switch
        if self.last_switch_time:
            time_since_switch = (datetime.now() - self.last_switch_time).total_seconds()
            if time_since_switch < self.min_switch_interval:
                return None
        
        # Determine current state based on metrics
        new_state = self._determine_state(metrics)
        
        # Log state transition if changed
        if new_state != self.current_state:
            logger.info(f"Policy state transition: {self.current_state} -> {new_state}")
            logger.info(f"Metrics: {metrics}")
            self.current_state = new_state
        
        # Get preferred topology for current state
        preferred = self.rules[self.current_state]['preferred_topology']
        
        # Consider comfort metrics
        if metrics.get('avg_comfort', 0) > (1.0 - self.comfort_weight):
            # High stress - might override to a more relaxed topology
            logger.warning(f"High stress detected (comfort: {metrics['avg_comfort']})")
            if self.current_state == "intensive":
                preferred = "hexagonal"  # Step down from square
        
        # Return new topology if different from current
        if preferred != current_topology:
            logger.info(f"Topology change recommended: {current_topology} -> {preferred}")
            self.last_switch_time = datetime.now()
            self.switch_count += 1
            return preferred
        
        return None
    
    def _determine_state(self, metrics: Dict[str, float]) -> str:
        """Determine system state from metrics"""
        # Check states in order of intensity
        for state in ['intensive', 'active', 'idle']:
            if state in self.rules and self.rules[state]['check'](metrics):
                return state
        
        # Default to idle if no match
        return 'idle'
