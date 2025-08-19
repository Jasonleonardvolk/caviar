"""
Tool registration for TORI MCP server
"""

from .reflection_tools import register_reflection_tools
from .dynamics_tools import register_dynamics_tools
from .consciousness_tools import register_consciousness_tools
from .metacognitive_tools import register_metacognitive_tools
from .soliton_memory_tools import register_soliton_memory_tools


def register_tools(mcp, state_manager):
    """Register all tools with the MCP server."""
    register_reflection_tools(mcp, state_manager)
    register_dynamics_tools(mcp, state_manager)
    register_consciousness_tools(mcp, state_manager)
    register_metacognitive_tools(mcp, state_manager)
    register_soliton_memory_tools(mcp, state_manager)