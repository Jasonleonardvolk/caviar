"""
Agent Registry - Dynamic agent management with hot-swapping
Migrated from mcp_server_arch
"""

import sys
import importlib
import importlib.util
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .psi_archive import psi_archive

logger = logging.getLogger(__name__)

class Agent:
    """Base agent class"""
    def __init__(self, name: str, psi_archive=None):
        self.name = name
        self.psi_archive = psi_archive or psi_archive
        
    async def execute(self, *args, **kwargs):
        """Execute agent logic - override in subclasses"""
        raise NotImplementedError

class AgentRegistry:
    """Registry for managing agents with hot-swapping support"""
    
    def __init__(self, psi_archive_instance=None):
        self._agents: Dict[str, Agent] = {}
        self.psi_archive = psi_archive_instance or psi_archive
        
    def register(self, name: str, agent: Agent):
        """Register an agent"""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered")
        
        self._agents[name] = agent
        
        # Log registration
        if self.psi_archive:
            self.psi_archive.log_event("agent_registered", {"agent": name})
        
        logger.info(f"Agent '{name}' registered")
    
    def get(self, name: str) -> Optional[Agent]:
        """Get an agent by name"""
        return self._agents.get(name)
    
    def list_agents(self) -> list:
        """List all registered agents"""
        return list(self._agents.keys())
    
    def swap(self, name: str, new_agent: Agent) -> Optional[Agent]:
        """Hot-swap an agent"""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found for swap")
        
        old_agent = self._agents[name]
        self._agents[name] = new_agent
        
        # Log hot-swap
        if self.psi_archive:
            old_type = type(old_agent).__name__
            new_type = type(new_agent).__name__
            self.psi_archive.log_event("hotswap", {
                "agent": name,
                "old_type": old_type,
                "new_type": new_type
            })
        
        logger.info(f"Agent '{name}' hot-swapped")
        return old_agent
    
    def reload_agent(self, agent_name: str) -> bool:
        """Reload an agent module dynamically"""
        try:
            # Try to find the agent module
            module_name = f"agents.{agent_name}"
            
            if module_name in sys.modules:
                # Reload existing module
                module = importlib.reload(sys.modules[module_name])
            else:
                # Import new module
                module = importlib.import_module(module_name)
            
            # Look for agent class in module
            agent_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Agent) and attr != Agent:
                    agent_class = attr
                    break
            
            if agent_class:
                # Create new instance and swap
                new_agent = agent_class(agent_name, self.psi_archive)
                self.swap(agent_name, new_agent)
                return True
            
            logger.warning(f"No Agent subclass found in module {module_name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to reload agent '{agent_name}': {e}")
            if self.psi_archive:
                self.psi_archive.log_event("reload_error", {
                    "agent": agent_name,
                    "error": str(e)
                })
            return False

# Example agents
class DanielAgent(Agent):
    """Daniel - Workflow execution agent"""
    
    async def execute(self, input_data=None):
        """Run workflow"""
        if self.psi_archive:
            self.psi_archive.log_event("run_start", {"agent": self.name})
        
        # Workflow logic here
        result = {"status": "ok", "agent": self.name}
        
        if self.psi_archive:
            self.psi_archive.log_event("run_end", {
                "agent": self.name,
                "status": "completed"
            })
        
        return result

class TonkaAgent(Agent):
    """Tonka - Agent patching and modification"""
    
    async def propose_patch(self, target_agent: str, patch_code: str):
        """Propose a patch for another agent"""
        if self.psi_archive:
            self.psi_archive.log_event("patch_proposed", {
                "source": self.name,
                "target": target_agent
            })
        
        # In real implementation, this would modify the target agent
        return {"status": "patch_proposed", "target": target_agent}

# Global registry
agent_registry = AgentRegistry()

# Register default agents
agent_registry.register("daniel", DanielAgent("daniel"))
agent_registry.register("tonka", TonkaAgent("tonka"))
