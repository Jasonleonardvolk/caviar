"""
MCP Server Duplicate Registration Fix
Add this check to mcp_metacognitive/server_proper.py or server_simple.py
"""

# In the tool registration section, add existence checks:

def register_tool_safe(mcp_server, name, handler, description):
    """Register tool only if it doesn't already exist"""
    if hasattr(mcp_server, '_tools') and name in mcp_server._tools:
        logger.debug(f"Tool {name} already registered, skipping")
        return
    
    mcp_server.add_tool(
        name=name,
        handler=handler,
        description=description
    )

# Similarly for resources:

def register_resource_safe(mcp_server, uri, handler, description):
    """Register resource only if it doesn't already exist"""
    if hasattr(mcp_server, '_resources') and uri in mcp_server._resources:
        logger.debug(f"Resource {uri} already registered, skipping")
        return
    
    mcp_server.add_resource(
        uri=uri,
        handler=handler,
        description=description
    )

# And for prompts:

def register_prompt_safe(mcp_server, name, handler, description):
    """Register prompt only if it doesn't already exist"""
    if hasattr(mcp_server, '_prompts') and name in mcp_server._prompts:
        logger.debug(f"Prompt {name} already registered, skipping")
        return
    
    mcp_server.add_prompt(
        name=name,
        handler=handler,
        description=description
    )
