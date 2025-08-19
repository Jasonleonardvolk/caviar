"""
Prompt registration for TORI MCP server
"""

from .cognitive_prompts import register_cognitive_prompts


def register_prompts(mcp, state_manager):
    """Register all prompts with the MCP server."""
    register_cognitive_prompts(mcp, state_manager)