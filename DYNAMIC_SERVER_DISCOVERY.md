# Dynamic Server Discovery System 🚀

## Overview

The TORI MCP ecosystem now features a **dynamic server discovery system** that automatically detects and starts all available MCP servers. No more hardcoding servers - just drop them in and they're automatically integrated!

## How It Works

### 1. Automatic Discovery

The system automatically searches for servers in:
- `mcp_metacognitive/agents/` - Core agents (Daniel, Kaizen, etc.)
- `mcp_metacognitive/servers/` - Additional servers
- `mcp_metacognitive/extensions/` - Extension servers
- `mcp_metacognitive/mcp_servers/` - MCP-specific servers

It also loads from manifest files:
- `servers.json` - External server definitions
- `mcp_servers.json` - MCP server manifest
- `agent_manifest.json` - Agent configurations

### 2. Server Metadata

Each server must include metadata:

```python
class MyNewServer(Agent):
    _metadata = {
        "name": "my_server",
        "description": "What this server does",
        "enabled": True,  # Is it enabled?
        "auto_start": True,  # Start automatically?
        "endpoints": [  # API endpoints it provides
            {"path": "/api/my_server/action", "method": "POST", "description": "Do something"}
        ],
        "dependencies": [],  # Other servers it needs
        "version": "1.0.0"
    }
```

### 3. Automatic Integration

When you run `python enhanced_launcher.py`:
1. **Discovery** - Finds all servers
2. **Loading** - Loads enabled servers
3. **Dependency Resolution** - Starts servers in correct order
4. **Registration** - Registers with agent registry
5. **Auto-Start** - Starts servers with `auto_start=true`
6. **Status Reporting** - Shows all discovered servers

## Adding a New Server

### Method 1: Python Code

1. Create a new file in `extensions/` (e.g., `my_server.py`):

```python
from ..core.agent_registry import Agent

class MyServer(Agent):
    _metadata = {
        "name": "my_server",
        "description": "My custom MCP server",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/api/my_server/process", "method": "POST", "description": "Process data"}
        ],
        "dependencies": [],
        "version": "1.0.0"
    }
    
    async def execute(self, command: str, params: Dict[str, Any] = None):
        # Your logic here
        return {"status": "success", "result": "processed"}
```

2. That's it! The server will be discovered on next launch.

### Method 2: JSON Manifest

Add to `servers.json`:

```json
{
  "servers": [
    {
      "name": "external_api",
      "module": "extensions.external_api",
      "class": "ExternalAPIServer",
      "config": {
        "api_key": "${EXTERNAL_API_KEY}"
      },
      "metadata": {
        "description": "External API integration",
        "enabled": true,
        "auto_start": true,
        "endpoints": [
          {"path": "/api/external/query", "method": "GET", "description": "Query external API"}
        ]
      }
    }
  ]
}
```

## Environment Configuration

Servers can be configured via environment variables:

```bash
# For a server named "my_server"
set MY_SERVER_API_KEY=your-key
set MY_SERVER_TIMEOUT=60
set MY_SERVER_ENABLED=true
```

## Status and Monitoring

### System Status Endpoint

`GET http://localhost:8100/api/system/status`

Returns:
```json
{
  "servers": {
    "daniel": {
      "discovered": true,
      "enabled": true,
      "running": true,
      "description": "Main cognitive processing engine",
      "endpoints": [...]
    },
    "kaizen": {
      "discovered": true,
      "enabled": true,
      "running": true,
      "description": "Continuous improvement engine",
      "endpoints": [...]
    },
    "my_server": {
      "discovered": true,
      "enabled": true,
      "running": false,
      "description": "My custom server"
    }
  },
  "discovery": {
    "total_discovered": 3,
    "enabled": 3,
    "running": 2
  }
}
```

### Enhanced Launcher Output

The launcher now shows:
```
✅ Dynamic server discovery complete!
   🔍 Discovered: 5 servers
   ⚡ Running: 3 servers
   ✅ daniel: Main cognitive processing engine
   ✅ kaizen: Continuous improvement engine
   ✅ my_custom_server: My custom functionality
   ⏸️  example: Enabled but not running
   ⏸️  memory_server: Enabled but not running
```

## Benefits

1. **Extensibility** - Add servers without modifying core code
2. **Modularity** - Each server is self-contained
3. **Configuration** - Environment-based configuration
4. **Dependencies** - Automatic dependency resolution
5. **Discovery** - No manual registration needed
6. **Status** - Real-time visibility of all servers

## Example Use Cases

### 1. Add a Web Scraper
```python
class WebScraperServer(Agent):
    _metadata = {
        "name": "web_scraper",
        "description": "Web scraping and data extraction",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/api/scraper/extract", "method": "POST", "description": "Extract data from URL"}
        ]
    }
```

### 2. Add a Database Connector
```python
class DatabaseServer(Agent):
    _metadata = {
        "name": "database",
        "description": "Database query and management",
        "enabled": True,
        "auto_start": True,
        "dependencies": ["daniel"],  # Uses Daniel for query understanding
        "endpoints": [
            {"path": "/api/db/query", "method": "POST", "description": "Execute database query"}
        ]
    }
```

### 3. Add an Email Integration
```python
class EmailServer(Agent):
    _metadata = {
        "name": "email",
        "description": "Email sending and receiving",
        "enabled": False,  # Disabled by default
        "auto_start": False,
        "endpoints": [
            {"path": "/api/email/send", "method": "POST", "description": "Send email"}
        ]
    }
```

## Troubleshooting

### Server Not Discovered
- Check the file is in a search directory
- Ensure the class inherits from `Agent`
- Verify `_metadata` is defined

### Server Not Starting
- Check `enabled: true` in metadata
- Check dependencies are satisfied
- Look for errors in logs

### Configuration Issues
- Environment variables must be UPPERCASE
- Format: `{SERVER_NAME}_{CONFIG_KEY}`
- Check servers.json syntax

## Summary

The dynamic discovery system makes TORI truly extensible. Just drop in a new server file or add to the manifest, and it's automatically part of the ecosystem. No core code changes needed!
