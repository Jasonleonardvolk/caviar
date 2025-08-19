# python/core/topology_policy.py

import asyncio
import logging
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np

from .hot_swap_laplacian import HotSwappableLaplacian
from .oscillator_lattice import get_global_lattice

logger = logging.getLogger(__name__)

class TopologyState(Enum):
    """System states that influence topology choice"""
    IDLE = "idle"              # Low activity - optimize for stability
    ACTIVE = "active"          # Normal operation
    INTENSIVE = "intensive"    # Heavy memory operations
    CONSOLIDATING = "consolidating"  # Nightly maintenance

class TopologyPolicy:
    """
    Heuristic policy engine for dynamic topology decisions.
    Can be replaced with RL agent in future.
    """
    
    def __init__(self, hot_swap: HotSwappableLaplacian):
        self.hot_swap = hot_swap
        self.state = TopologyState.ACTIVE
        
        # Thresholds for state transitions
        self.thresholds = {
            "memory_density": 0.7,      # Switch if >70% capacity
            "loss_rate": 0.1,           # Switch if >10% loss/hour
            "access_rate": 100,         # Switch if >100 accesses/min
            "soliton_count": 2000,      # Switch topology at 2k solitons
        }
        
        # Metrics tracking
        self.metrics = {
            "soliton_count": 0,
            "loss_rate": 0.0,
            "access_rate": 0.0,
            "last_switch": None,
        }
        
        # Policy rules
        self.rules = self._init_rules()
        
    def _init_rules(self) -> Dict[str, Any]:
        """Initialize heuristic rules for topology selection"""
        return {
            TopologyState.IDLE: {
                "preferred": "kagome",
                "reason": "Maximum stability for long-term storage"
            },
            TopologyState.ACTIVE: {
                "preferred": "kagome",
                "fallback": "hexagonal",
                "switch_if": lambda m: m["soliton_count"] > self.thresholds["soliton_count"]
            },
            TopologyState.INTENSIVE: {
                "preferred": "hexagonal",
                "reason": "Better propagation for active recall"
            },
            TopologyState.CONSOLIDATING: {
                "preferred": "small_world",
                "reason": "Maximum interaction for consolidation"
            }
        }
        
    async def update_metrics(self):
        """Update system metrics from lattice"""
        lattice = get_global_lattice()
        
        self.metrics["soliton_count"] = len([o for o in lattice.oscillators if o.get("active", True)])
        
        # Estimate loss rate (simplified)
        stable_count = sum(1 for o in lattice.oscillators if o.get("stability", 0) > 0.5)
        self.metrics["loss_rate"] = 1.0 - (stable_count / max(1, self.metrics["soliton_count"]))
        
        # Update state based on metrics
        self._update_state()
        
    def _update_state(self):
        """Update system state based on metrics"""
        if self.metrics["access_rate"] > self.thresholds["access_rate"]:
            self.state = TopologyState.INTENSIVE
        elif self.metrics["soliton_count"] < 100:
            self.state = TopologyState.IDLE
        else:
            self.state = TopologyState.ACTIVE
            
    async def evaluate_topology(self) -> Optional[str]:
        """Evaluate if topology change is needed"""
        await self.update_metrics()
        
        current = self.hot_swap.current_topology
        rule = self.rules[self.state]
        
        # Check if switch is needed
        preferred = rule.get("preferred", current)
        
        if "switch_if" in rule:
            if rule["switch_if"](self.metrics):
                preferred = rule.get("fallback", preferred)
                
        # Avoid too frequent switches
        if self.metrics.get("last_switch"):
            time_since_switch = asyncio.get_event_loop().time() - self.metrics["last_switch"]
            if time_since_switch < 300:  # 5 minute cooldown
                return None
                
        if preferred != current:
            logger.info(f"Policy recommends switch: {current} → {preferred}")
            logger.info(f"Reason: {rule.get('reason', 'Metrics threshold')}")
            return preferred
            
        return None
        
    async def execute_switch(self, target_topology: str):
        """Execute topology switch with safety checks"""
        try:
            # Check energy budget
            if hasattr(self.hot_swap, 'ccl') and hasattr(self.hot_swap.ccl, 'energy_broker'):
                if not self.hot_swap.ccl.energy_broker.request("TOPOLOGY_SWITCH", 200, f"switch_to_{target_topology}"):
                    logger.warning("Insufficient energy for topology switch")
                    return False
                    
            # Perform switch
            await self.hot_swap.hot_swap_laplacian_with_safety(target_topology)
            
            self.metrics["last_switch"] = asyncio.get_event_loop().time()
            return True
            
        except Exception as e:
            logger.error(f"Topology switch failed: {e}")
            return False
            
    async def nightly_consolidation_mode(self):
        """Special mode for nightly consolidation"""
        old_state = self.state
        self.state = TopologyState.CONSOLIDATING
        
        # Switch to all-to-all for maximum interaction
        await self.execute_switch("small_world")
        
        # Return function to restore state
        async def restore():
            self.state = old_state
            preferred = self.rules[self.state]["preferred"]
            await self.execute_switch(preferred)
            
        return restore
        
    def get_status(self) -> Dict[str, Any]:
        """Get policy status and metrics"""
        return {
            "state": self.state.value,
            "current_topology": self.hot_swap.current_topology,
            "metrics": self.metrics.copy(),
            "thresholds": self.thresholds.copy(),
            "recommendation": self.rules[self.state].get("preferred", "none")
        }

# Singleton policy instance
_policy_instance: Optional[TopologyPolicy] = None

def get_topology_policy(hot_swap: Optional[HotSwappableLaplacian] = None) -> TopologyPolicy:
    """Get or create topology policy instance"""
    global _policy_instance
    
    if _policy_instance is None:
        if hot_swap is None:
            raise ValueError("Hot swap instance required for first initialization")
        _policy_instance = TopologyPolicy(hot_swap)
        
    return _policy_instance
