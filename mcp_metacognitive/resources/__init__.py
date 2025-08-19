"""
Resource registration for TORI MCP server
"""

from .state_resources import register_state_resources
from .monitoring_resources import register_monitoring_resources
from .knowledge_resources import register_knowledge_resources
from .soliton_memory_resources import register_soliton_memory_resources


def register_resources(mcp, state_manager):
    """Register all resources with the MCP server."""
    register_state_resources(mcp, state_manager)
    register_monitoring_resources(mcp, state_manager)
    register_knowledge_resources(mcp, state_manager)
    register_soliton_memory_resources(mcp, state_manager)