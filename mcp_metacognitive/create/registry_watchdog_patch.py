"""
Registry Watchdog Patch - Add supervisor restart capability

Apply this patch to mcp_metacognitive/core/agent_registry.py to add 
automatic restart for crashed agent tasks.

AUTOMATIC WIRING INSTRUCTIONS:
Add these two lines at the END of core/agent_registry.py:

from create.registry_watchdog_patch import add_supervisor_to_registry
add_supervisor_to_registry(agent_registry)    # idempotent
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

async def supervised_agent_start(self, agent_name: str):
    """
    Start an agent with supervisor protection
    
    Add this method to AgentRegistry class
    """
    agent = self.get(agent_name)
    if not agent:
        logger.error(f"Agent '{agent_name}' not found")
        return
    
    # Create supervisor task
    supervisor_task = asyncio.create_task(
        self._supervise_agent(agent_name)
    )
    
    # Store supervisor reference
    if not hasattr(self, '_supervisors'):
        self._supervisors = {}
    self._supervisors[agent_name] = supervisor_task
    
    logger.info(f"Started supervisor for agent '{agent_name}'")

async def _supervise_agent(self, agent_name: str, max_restarts: int = 3):
    """
    Supervisor loop for an agent - handles both crashes AND timeouts
    
    Add this method to AgentRegistry class
    """
    consecutive_failures = 0
    backoff_base = 60  # Base seconds for exponential backoff
    
    while consecutive_failures < max_restarts:
        try:
            agent = self.get(agent_name)
            if not agent:
                logger.error(f"Agent '{agent_name}' no longer exists")
                break
            
            # Start the agent if it has a start method
            if hasattr(agent, 'start'):
                logger.info(f"Starting agent '{agent_name}'")
                await agent.start()
                
                # Wait for agent task to complete (it shouldn't unless stopped)
                if hasattr(agent, '_task') and agent._task:
                    await agent._task
            else:
                # For agents without start(), monitor execute calls with timeout
                logger.info(f"Agent '{agent_name}' doesn't have start(), monitoring with timeout")
                while True:
                    # Check agent health every 5 minutes
                    await asyncio.sleep(300)
                    
                    # You could add health check logic here
                    if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'last_success'):
                        time_since_success = datetime.utcnow() - agent.metrics.last_success
                        if time_since_success > timedelta(hours=1):
                            logger.warning(f"Agent '{agent_name}' hasn't succeeded in {time_since_success}")
                            raise Exception("Agent unhealthy - no recent success")
                    
            consecutive_failures = 0  # Reset on clean exit
            
        except asyncio.CancelledError:
            logger.info(f"Supervisor for '{agent_name}' cancelled")
            break
            
        except asyncio.TimeoutError:
            consecutive_failures += 1
            logger.error(f"Agent '{agent_name}' timed out (attempt {consecutive_failures}/{max_restarts})")
            
            if consecutive_failures >= max_restarts:
                logger.critical(f"Agent '{agent_name}' exceeded max restarts due to timeouts")
                
                # Log to psi_archive
                if self.psi_archive:
                    self.psi_archive.log_event("agent_supervisor_timeout_failed", {
                        "agent": agent_name,
                        "max_restarts": max_restarts,
                        "error_type": "timeout"
                    })
                break
            
            # Exponential backoff
            backoff = min(backoff_base * (2 ** consecutive_failures), 3600)
            logger.info(f"Restarting agent '{agent_name}' after timeout in {backoff} seconds...")
            await asyncio.sleep(backoff)
            
            # Try to reload the agent
            if hasattr(self, 'reload_agent'):
                self.reload_agent(agent_name)
            
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Agent '{agent_name}' crashed (attempt {consecutive_failures}/{max_restarts}): {e}")
            
            if consecutive_failures >= max_restarts:
                logger.critical(f"Agent '{agent_name}' exceeded max restarts")
                
                # Log to psi_archive
                if self.psi_archive:
                    self.psi_archive.log_event("agent_supervisor_failed", {
                        "agent": agent_name,
                        "max_restarts": max_restarts,
                        "final_error": str(e)
                    })
                break
            
            # Exponential backoff
            backoff = min(backoff_base * (2 ** consecutive_failures), 3600)
            logger.info(f"Restarting agent '{agent_name}' in {backoff} seconds...")
            await asyncio.sleep(backoff)
            
            # Try to reload the agent
            if hasattr(self, 'reload_agent'):
                self.reload_agent(agent_name)

async def start_all_agents_supervised(self):
    """
    Start all registered agents with supervisor protection
    
    Add this method to AgentRegistry class
    """
    for agent_name in self.list_agents():
        agent = self.get(agent_name)
        if agent and hasattr(agent, '_metadata'):
            metadata = agent._metadata
            if metadata.get('auto_start', False):
                await self.supervised_agent_start(agent_name)

def add_supervisor_to_registry(registry):
    """
    Add supervisor methods to an existing registry instance (IDEMPOTENT)
    
    Call this function with your agent_registry instance:
    ```python
    from create.registry_watchdog_patch import add_supervisor_to_registry
    add_supervisor_to_registry(agent_registry)
    ```
    """
    # Check if already patched (idempotent)
    if hasattr(registry, 'supervised_agent_start'):
        logger.debug("Registry already has supervisor functionality")
        return registry
    
    import types
    
    # Add methods to the instance
    registry.supervised_agent_start = types.MethodType(supervised_agent_start, registry)
    registry._supervise_agent = types.MethodType(_supervise_agent, registry)
    registry.start_all_agents_supervised = types.MethodType(start_all_agents_supervised, registry)
    
    # Initialize supervisor storage
    if not hasattr(registry, '_supervisors'):
        registry._supervisors = {}
    
    logger.info("Registry patched with supervisor functionality")
    
    return registry

# Auto-initialization code for agent_registry.py:
"""
# Add this at the END of core/agent_registry.py:

# Auto-wire watchdog functionality
try:
    from create.registry_watchdog_patch import add_supervisor_to_registry
    add_supervisor_to_registry(agent_registry)    # idempotent
    
    # Optional: Auto-start supervised agents on module load
    # import asyncio
    # asyncio.create_task(agent_registry.start_all_agents_supervised())
    
except ImportError:
    logger.warning("Watchdog patch not available - agents running without supervisor")
"""
