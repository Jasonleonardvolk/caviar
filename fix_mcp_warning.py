#!/usr/bin/env python3
"""
Fix for FastMCP host/port warning
"""

import re
from pathlib import Path

def fix_mcp_run_warning():
    """Fix the FastMCP.run() host/port warning"""
    
    # Find MCP server files
    mcp_files = [
        Path("mcp_metacognitive/server_proper.py"),
        Path("mcp_metacognitive/server_simple.py"),
        Path("mcp_metacognitive/server.py")
    ]
    
    fixed_count = 0
    
    for mcp_file in mcp_files:
        if not mcp_file.exists():
            continue
            
        print(f"Checking {mcp_file}...")
        
        with open(mcp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for the problematic run() call
        # Pattern: mcp.run(host=..., port=...)
        pattern = r'(\w+\.run\s*\(\s*)host\s*=\s*[^,]+\s*,\s*port\s*=\s*[^)]+(\s*\))'
        
        if re.search(pattern, content):
            # Replace with just .run() since FastMCP doesn't accept these params
            new_content = re.sub(pattern, r'\1\2', content)
            
            # Backup original
            backup = mcp_file.with_suffix('.py.backup_mcp_warning')
            if not backup.exists():
                mcp_file.rename(backup)
                
            # Write fixed version
            with open(mcp_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Fixed {mcp_file}")
            fixed_count += 1
        else:
            # Try alternative pattern where host/port might be in transport_options
            alt_pattern = r'transport_options\s*=\s*\{[^}]*["\']host["\']\s*:[^}]+\}'
            
            if re.search(alt_pattern, content):
                print(f"⚠️  {mcp_file} uses transport_options - may need manual review")
    
    if fixed_count > 0:
        print(f"\n✅ Fixed {fixed_count} file(s)")
        print("The warning should be gone on next restart")
    else:
        print("\n📝 No direct mcp.run(host=, port=) calls found")
        print("The warning might be coming from the FastMCP framework itself")

if __name__ == "__main__":
    fix_mcp_run_warning()
