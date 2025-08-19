#!/usr/bin/env python
"""
Watchdog enhancements for agent_registry.py
Apply these changes to add supervisor/restart functionality
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WatchdogMixin:
    """
    Mixin to add watchdog functionality to AgentRegistry
    
    Add this to your agent_registry.py:
    
    class AgentRegistry(WatchdogMixin):
        # ... existing code ...
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_health: Dict[str, Dict[str, Any]] = {}
        self._restart_attempts: Dict[str, int] = {}
        self._max_restart_attempts = 3
        self._restart_cooldown = timedelta(minutes=5)
    
    async def execute_with_watchdog(self, agent_name: str, input_data: Any = None, 
                                   timeout: int = 60) -> Dict[str, Any]:
        """
        Execute an agent with watchdog protection
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            timeout: Timeout in seconds
            
        Returns:
            Execution result or error dict
        """
        agent = self.get(agent_name)
        if not agent:
            return {"status": "error", "error": f"Agent '{agent_name}' not found"}
        
        # Initialize health tracking
        if agent_name not in self._agent_health:
            self._agent_health[agent_name] = {
                "last_success": datetime.utcnow(),
                "consecutive_failures": 0,
                "total_executions": 0,
                "total_timeouts": 0
            }
        
        health = self._agent_health[agent_name]
        health["total_executions"] += 1
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(input_data),
                timeout=timeout
            )
            
            # Update health on success
            health["last_success"] = datetime.utcnow()
            health["consecutive_failures"] = 0
            
            return result
            
        except asyncio.TimeoutError:
            health["consecutive_failures"] += 1
            health["total_timeouts"] += 1
            
            logger.error(f"Agent '{agent_name}' timed out after {timeout}s")
            
            # Check if we should restart
            if await self._should_restart_agent(agent_name):
                await self._restart_agent(agent_name)
            
            return {
                "status": "timeout",
                "error": f"Agent execution timed out after {timeout} seconds",
                "agent": agent_name
            }
            
        except Exception as e:
            health["consecutive_failures"] += 1
            
            logger.error(f"Agent '{agent_name}' failed: {e}")
            
            # Check if we should restart
            if await self._should_restart_agent(agent_name):
                await self._restart_agent(agent_name)
            
            return {
                "status": "error",
                "error": str(e),
                "agent": agent_name
            }
    
    async def _should_restart_agent(self, agent_name: str) -> bool:
        """Determine if an agent should be restarted"""
        health = self._agent_health.get(agent_name, {})
        
        # Check consecutive failures
        if health.get("consecutive_failures", 0) < 3:
            return False
        
        # Check restart attempts
        attempts = self._restart_attempts.get(agent_name, 0)
        if attempts >= self._max_restart_attempts:
            logger.critical(f"Agent '{agent_name}' exceeded max restart attempts")
            return False
        
        # Check cooldown
        last_restart = health.get("last_restart")
        if last_restart:
            if datetime.utcnow() - last_restart < self._restart_cooldown:
                return False
        
        return True
    
    async def _restart_agent(self, agent_name: str):
        """Restart an agent"""
        logger.warning(f"Restarting agent '{agent_name}'...")
        
        try:
            # Get current agent
            old_agent = self.get(agent_name)
            if not old_agent:
                return
            
            # Shutdown old agent if it has the method
            if hasattr(old_agent, 'shutdown'):
                await old_agent.shutdown()
            
            # Try to reload the agent module
            success = self.reload_agent(agent_name)
            
            if success:
                # Update tracking
                self._restart_attempts[agent_name] = self._restart_attempts.get(agent_name, 0) + 1
                self._agent_health[agent_name]["last_restart"] = datetime.utcnow()
                self._agent_health[agent_name]["consecutive_failures"] = 0
                
                # Start the new agent if it has a start method
                new_agent = self.get(agent_name)
                if hasattr(new_agent, 'start'):
                    await new_agent.start()
                
                logger.info(f"Agent '{agent_name}' restarted successfully")
                
                # Log to psi_archive
                if self.psi_archive:
                    self.psi_archive.log_event("agent_restarted", {
                        "agent": agent_name,
                        "attempt": self._restart_attempts[agent_name]
                    })
            else:
                logger.error(f"Failed to restart agent '{agent_name}'")
                
        except Exception as e:
            logger.error(f"Error restarting agent '{agent_name}': {e}")
    
    def get_agent_health(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health information for agents"""
        if agent_name:
            return self._agent_health.get(agent_name, {})
        return self._agent_health
    
    async def monitor_agents(self, interval: int = 60):
        """
        Continuous monitoring task for all agents
        
        Args:
            interval: Check interval in seconds
        """
        while True:
            try:
                for agent_name in self.list_agents():
                    health = self._agent_health.get(agent_name, {})
                    
                    # Check if agent is unhealthy
                    last_success = health.get("last_success")
                    if last_success:
                        time_since_success = datetime.utcnow() - last_success
                        if time_since_success > timedelta(minutes=30):
                            logger.warning(
                                f"Agent '{agent_name}' hasn't succeeded in {time_since_success}"
                            )
                            
                            # Consider restarting if too long
                            if time_since_success > timedelta(hours=1):
                                if await self._should_restart_agent(agent_name):
                                    await self._restart_agent(agent_name)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(interval)


# Example enhancement to apply to existing agent_registry.py:
"""
# In agent_registry.py, modify the AgentRegistry class:

from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

class AgentRegistry(WatchdogMixin):  # Add mixin
    def __init__(self, psi_archive_instance=None):
        super().__init__()  # Initialize mixin
        self._agents: Dict[str, Agent] = {}
        self.psi_archive = psi_archive_instance or psi_archive
        # ... rest of existing __init__ ...
    
    # Add this method to start monitoring
    async def start_monitoring(self):
        '''Start the agent monitoring task'''
        asyncio.create_task(self.monitor_agents())

# Then in your main startup:
# await agent_registry.start_monitoring()
"""

# Standalone function to patch existing registry
def patch_agent_registry():
    """
    Apply watchdog enhancements to existing agent_registry
    
    Usage:
        from create.watchdog_enhancements import patch_agent_registry
        patch_agent_registry()
    """
    try:
        from ..core.agent_registry import AgentRegistry, agent_registry
        
        # Add methods dynamically
        AgentRegistry.execute_with_watchdog = WatchdogMixin.execute_with_watchdog
        AgentRegistry._should_restart_agent = WatchdogMixin._should_restart_agent
        AgentRegistry._restart_agent = WatchdogMixin._restart_agent
        AgentRegistry.get_agent_health = WatchdogMixin.get_agent_health
        AgentRegistry.monitor_agents = WatchdogMixin.monitor_agents
        
        # Initialize tracking dicts
        agent_registry._agent_health = {}
        agent_registry._restart_attempts = {}
        agent_registry._max_restart_attempts = 3
        agent_registry._restart_cooldown = timedelta(minutes=5)
        
        logger.info("Successfully patched agent_registry with watchdog functionality")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch agent_registry: {e}")
        return False

if __name__ == "__main__":
    print("Watchdog Enhancement Module")
    print("=" * 50)
    print("\nTo apply these enhancements:")
    print("1. Add WatchdogMixin to your AgentRegistry class")
    print("2. Or use patch_agent_registry() to apply dynamically")
    print("\nExample:")
    print("  from create.watchdog_enhancements import patch_agent_registry")
    print("  patch_agent_registry()")
