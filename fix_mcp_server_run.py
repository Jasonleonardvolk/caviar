#!/usr/bin/env python3
"""
Fix MCP server.py to handle FastMCP.run() properly
"""

import sys
from pathlib import Path

print("🔧 FIXING MCP SERVER.PY")
print("=" * 60)

server_path = Path("mcp_metacognitive/server.py")

if not server_path.exists():
    print("❌ server.py not found!")
    sys.exit(1)

# Read content
content = server_path.read_text(encoding='utf-8')

# Backup
backup_path = server_path.with_suffix('.py.backup_run_fix')
if not backup_path.exists():
    backup_path.write_text(content, encoding='utf-8')
    print(f"✅ Created backup: {backup_path}")

# Fix the mcp.run() call
print("\n📝 Fixing mcp.run() call...")

# Replace the problematic line
old_line = 'mcp.run(transport="sse", host=config.server_host, port=config.server_port)'
new_code = '''# Fix: FastMCP.run() doesn't accept host/port in the mcp package
        # We need to configure the server differently for SSE
        import uvicorn
        from mcp.server.sse import create_sse_transport
        
        # Create SSE app
        transport = create_sse_transport(mcp)
        
        # Run with uvicorn
        uvicorn.run(
            transport.app,
            host=config.server_host,
            port=config.server_port,
            log_level="info"
        )'''

if old_line in content:
    # Replace just that line with the new code block
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if old_line in line:
            # Get the indentation
            indent = len(line) - len(line.lstrip())
            # Add the new code with proper indentation
            for new_line in new_code.split('\n'):
                if new_line:  # Skip empty lines
                    new_lines.append(' ' * indent + new_line)
                else:
                    new_lines.append('')
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    print("✅ Fixed mcp.run() call to use uvicorn directly")
else:
    print("⚠️ mcp.run() line not found - checking if already fixed...")
    if "uvicorn.run" in content and "create_sse_transport" in content:
        print("✅ Already fixed!")
    else:
        print("❌ Could not find the line to fix")

# Write the fixed content
server_path.write_text(content, encoding='utf-8')

print("\n✅ Done! The MCP server should now start properly with SSE transport")
print("\nThe fix:")
print("  - Creates SSE transport explicitly")
print("  - Uses uvicorn.run() directly with host/port")
print("  - Avoids the FastMCP.run() parameter mismatch")
