#!/usr/bin/env python3
"""
Energy Proxy Adapter for Unified Integration Hub (UIH)
Bridges metacognitive modules with the Energy Budget Broker
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnergyRequest:
    """Energy allocation request"""
    module: str
    amount: int
    purpose: str
    priority: int = 5  # 1-10 scale

class EnergyProxy:
    """
    Drop-in adapter that intercepts metacognitive energy requests
    and routes them through the Energy Budget Broker
    """
    
    def __init__(self, energy_broker, ccl):
        self.energy_broker = energy_broker
        self.ccl = ccl
        self._pending_requests: Dict[str, EnergyRequest] = {}
        self._active_allocations: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable] = {}
        
    async def request_energy(self, 
                           module: str,
                           amount: int,
                           purpose: str,
                           priority: int = 5) -> bool:
        """
        Request energy allocation for a metacognitive module
        
        Args:
            module: Module identifier (e.g., 'UIH', 'RFPE', 'SMP', 'DMON')
            amount: Energy units required
            purpose: Description of intended use
            priority: Request priority (1-10)
            
        Returns:
            True if allocation granted
        """
        request = EnergyRequest(
            module=module,
            amount=amount,
            purpose=purpose,
            priority=priority
        )
        
        # High priority requests get immediate processing
        if priority >= 8:
            return self.energy_broker.request(module, amount, purpose)
            
        # Queue lower priority requests
        request_id = f"{module}_{len(self._pending_requests)}"
        self._pending_requests[request_id] = request
        
        # Process in background
        asyncio.create_task(self._process_request(request_id))
        
        return True  # Optimistically return True
        
    async def _process_request(self, request_id: str):
        """Process queued energy request"""
        if request_id not in self._pending_requests:
            return
            
        request = self._pending_requests[request_id]
        
        # Try allocation with retries
        for attempt in range(3):
            if self.energy_broker.request(
                request.module, 
                request.amount, 
                request.purpose
            ):
                # Success - notify module
                if request.module in self._callbacks:
                    await self._callbacks[request.module]('allocated', request)
                    
                del self._pending_requests[request_id]
                return
                
            # Wait before retry
            await asyncio.sleep(0.1 * (attempt + 1))
            
        # Failed - notify module
        if request.module in self._callbacks:
            await self._callbacks[request.module]('denied', request)
            
        del self._pending_requests[request_id]
        
    async def enter_chaos_mode(self,
                             module: str,
                             energy_budget: int,
                             purpose: str) -> Optional[str]:
        """
        Enter chaos mode through CCL
        
        Returns:
            Session ID if successful
        """
        # Check if module already has active chaos session
        if module in self._active_allocations:
            logger.warning(f"{module} already in chaos mode")
            return None
            
        # Enter CCL
        session_id = await self.ccl.enter_chaos_session(
            module_id=module,
            purpose=purpose,
            required_energy=energy_budget
        )
        
        if session_id:
            self._active_allocations[module] = {
                'session_id': session_id,
                'energy_budget': energy_budget,
                'purpose': purpose
            }
            
        return session_id
        
    async def exit_chaos_mode(self, module: str) -> Optional[Dict[str, Any]]:
        """Exit chaos mode and return results"""
        if module not in self._active_allocations:
            return None
            
        allocation = self._active_allocations[module]
        results = await self.ccl.exit_chaos_session(allocation['session_id'])
        
        del self._active_allocations[module]
        
        return results
        
    def register_callback(self, module: str, callback: Callable):
        """Register callback for energy events"""
        self._callbacks[module] = callback
        
    def get_balance(self, module: str) -> int:
        """Get current energy balance for module"""
        return self.energy_broker.get_balance(module)
        
    def get_status(self) -> Dict[str, Any]:
        """Get proxy status"""
        return {
            'pending_requests': len(self._pending_requests),
            'active_chaos_sessions': list(self._active_allocations.keys()),
            'registered_modules': list(self._callbacks.keys())
        }

# Integration with UIH
class UIHEnergyAdapter:
    """
    Specific adapter for Unified Integration Hub
    """
    
    def __init__(self, energy_proxy):
        self.energy_proxy = energy_proxy
        self.module_id = "UIH"
        
        # Register callback
        self.energy_proxy.register_callback(
            self.module_id,
            self._handle_energy_event
        )
        
    async def _handle_energy_event(self, event_type: str, request: EnergyRequest):
        """Handle energy allocation events"""
        if event_type == 'allocated':
            logger.info(f"UIH energy allocated: {request.amount} units")
        elif event_type == 'denied':
            logger.warning(f"UIH energy denied: {request.amount} units")
            
    async def request_exploration_energy(self, amount: int = 100) -> bool:
        """Request energy for exploratory reasoning"""
        return await self.energy_proxy.request_energy(
            module=self.module_id,
            amount=amount,
            purpose="exploratory_reasoning",
            priority=6
        )
        
    async def enter_creative_chaos(self, budget: int = 500) -> Optional[str]:
        """Enter creative chaos mode"""
        return await self.energy_proxy.enter_chaos_mode(
            module=self.module_id,
            energy_budget=budget,
            purpose="creative_exploration"
        )
        
    async def exit_creative_chaos(self) -> Optional[Dict[str, Any]]:
        """Exit creative chaos and consolidate insights"""
        results = await self.energy_proxy.exit_chaos_mode(self.module_id)
        
        if results:
            # Process chaos results into insights
            logger.info(f"UIH chaos session generated {results['chaos_generated']} units")
            
        return results
