#!/usr/bin/env python3
"""
Energy Budget Broker - Token-bucket energy management for CCL access
Part of EigenSentry-2.0 "Symphony Conductor"
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnergyAllocation:
    """Energy allocation record"""
    module_id: str
    joule_tau: int
    timestamp: float
    purpose: str

class EnergyBudgetBroker:
    """
    Token-bucket energy broker for Chaos Control Layer access.
    Each module has credits (in arbitrary units joules·τ).
    """
    
    MAX_CREDITS = 1000
    REFILL_RATE = 10  # credits per second
    
    def __init__(self):
        self._credits = defaultdict(lambda: self.MAX_CREDITS // 10)
        self._last_refill = defaultdict(float)
        self._allocations = []
        self._total_energy_spent = 0
        
    def request(self, module_id: str, joule_tau: int, purpose: str = "") -> bool:
        """
        Request energy allocation for CCL entry.
        
        Args:
            module_id: Requesting module identifier
            joule_tau: Energy units requested
            purpose: Optional description of intended use
            
        Returns:
            True if allocation granted, False otherwise
        """
        # Refill credits based on time elapsed
        self._refill_credits(module_id)
        
        if self._credits[module_id] >= joule_tau:
            self._credits[module_id] -= joule_tau
            self._total_energy_spent += joule_tau
            
            # Record allocation
            self._allocations.append(EnergyAllocation(
                module_id=module_id,
                joule_tau=joule_tau,
                timestamp=time.time(),
                purpose=purpose
            ))
            
            logger.info(f"Energy granted: {module_id} allocated {joule_tau} units for {purpose}")
            return True
            
        logger.warning(f"Energy denied: {module_id} has {self._credits[module_id]} < {joule_tau}")
        return False
        
    def refund(self, module_id: str, joule_tau: int):
        """Refund unused energy to a module"""
        self._credits[module_id] = min(
            self._credits[module_id] + joule_tau, 
            self.MAX_CREDITS
        )
        self._total_energy_spent -= joule_tau
        logger.info(f"Energy refunded: {module_id} returned {joule_tau} units")
        
    def _refill_credits(self, module_id: str):
        """Refill credits based on elapsed time"""
        now = time.time()
        last = self._last_refill.get(module_id, now)
        elapsed = now - last
        
        if elapsed > 0:
            refill = int(elapsed * self.REFILL_RATE)
            self._credits[module_id] = min(
                self._credits[module_id] + refill,
                self.MAX_CREDITS
            )
            self._last_refill[module_id] = now
            
    def get_balance(self, module_id: str) -> int:
        """Get current credit balance for a module"""
        self._refill_credits(module_id)
        return self._credits[module_id]
        
    def get_status(self) -> Dict[str, any]:
        """Get broker status"""
        return {
            'total_energy_spent': self._total_energy_spent,
            'active_modules': len(self._credits),
            'recent_allocations': len([a for a in self._allocations 
                                      if time.time() - a.timestamp < 60]),
            'module_balances': dict(self._credits)
        }
