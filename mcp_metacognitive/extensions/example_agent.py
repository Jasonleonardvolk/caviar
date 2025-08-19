"""
Example Agent Template
=====================

This is a template for creating new MCP servers that will be automatically discovered.
"""

from typing import Dict, Any, Optional
import logging
from ..core.agent_registry import Agent

logger = logging.getLogger(__name__)

class ExampleAgent(Agent):
    """
    Example agent that demonstrates how to create a new MCP server
    """
    
    # Metadata for dynamic discovery
    _metadata = {
        "name": "example",
        "description": "Example agent demonstrating the template structure",
        "enabled": False,  # Set to True to enable this agent
        "auto_start": False,  # Set to True to start automatically
        "endpoints": [
            {"path": "/api/example/hello", "method": "GET", "description": "Say hello"},
            {"path": "/api/example/process", "method": "POST", "description": "Process data"}
        ],
        "dependencies": [],  # List other agents this depends on
        "version": "0.1.0"
    }
    
    # Default configuration - can be overridden by environment variables
    _default_config = {
        "greeting": "Hello from Example Agent!",
        "max_processing_time": 30
    }
    
    def __init__(self, name: str = "example", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.config = config or self._default_config.copy()
        self.is_running = False
        
        logger.info(f"Example agent initialized with config: {self.config}")
    
    async def execute(self, command: str = "hello", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main execution method - implement your agent's logic here
        """
        if command == "hello":
            return {
                "status": "success",
                "message": self.config.get("greeting", "Hello!")
            }
        
        elif command == "process":
            # Example processing logic
            data = params.get("data", "") if params else ""
            processed = data.upper() if isinstance(data, str) else str(data)
            
            return {
                "status": "success",
                "original": data,
                "processed": processed
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown command: {command}"
            }
    
    async def start(self):
        """
        Start the agent - called when auto_start is True
        """
        self.is_running = True
        logger.info(f"{self.name} agent started")
    
    async def shutdown(self):
        """
        Graceful shutdown
        """
        self.is_running = False
        logger.info(f"{self.name} agent stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status - called by dynamic discovery
        """
        return {
            "running": self.is_running,
            "config": self.config
        }

# To create your own agent:
# 1. Copy this file to a new name (e.g., my_agent.py)
# 2. Update the class name and metadata
# 3. Set enabled=True in metadata
# 4. Implement your logic in the execute() method
# 5. The agent will be automatically discovered on next launch!
