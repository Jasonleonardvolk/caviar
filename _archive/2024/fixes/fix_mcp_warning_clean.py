#!/usr/bin/env python3
"""
Fix to suppress the repetitive FastMCP warning
"""

import re
from pathlib import Path

def fix_mcp_warning_clean():
    """Remove the warning log line from MCP server"""
    
    server_file = Path("mcp_metacognitive/server.py")
    
    if not server_file.exists():
        print("❌ mcp_metacognitive/server.py not found")
        return False
    
    # Backup
    backup = server_file.with_suffix('.py.backup_warning')
    if not backup.exists():
        server_file.rename(backup)
        backup.rename(server_file)
        
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the warning with a debug message (won't show in normal operation)
    content = content.replace(
        'logger.warning("FastMCP.run() doesn\'t accept host/port, using defaults")',
        'logger.debug("FastMCP.run() doesn\'t accept host/port, using defaults")'
    )
    
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Changed warning to debug - won't show in normal logs")
    return True

if __name__ == "__main__":
    fix_mcp_warning_clean()
