#!/usr/bin/env python3
"""
ASCII-SAFE MCP VERIFICATION
Test real MCP without Unicode crashes
"""

import subprocess
import sys


def test_ascii_safe_imports():
    """Test that imports work without Unicode crashes"""
    print("ASCII-SAFE MCP IMPORT TEST")
    print("=" * 40)
    
    imports_to_test = [
        "mcp_metacognitive",
        "mcp_metacognitive.config",
        "mcp_metacognitive.main", 
        "mcp_metacognitive.server_fixed"
    ]
    
    success_count = 0
    
    for import_name in imports_to_test:
        try:
            result = subprocess.run([
                sys.executable, '-c', f'''
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import {import_name}
print("IMPORT_SUCCESS")
'''
            ], capture_output=True, text=True, timeout=10)
            
            if "IMPORT_SUCCESS" in result.stdout:
                print(f"SUCCESS: {import_name}")
                success_count += 1
            else:
                print(f"FAILED: {import_name}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"ERROR: {import_name}: {e}")
    
    print(f"\\nImport Results: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)


def test_server_creation():
    """Test that we can create the real MCP server"""
    print("\\nREAL MCP SERVER TEST")
    print("=" * 30)
    
    try:
        result = subprocess.run([
            sys.executable, '-c', '''
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from mcp_metacognitive.server_fixed import create_real_mcp_server
server = create_real_mcp_server()
print("SERVER_CREATED")

# Test that server has tools
if hasattr(server, "_tools") and len(server._tools) > 0:
    print(f"TOOLS_FOUND: {len(server._tools)}")
else:
    print("NO_TOOLS")
'''
        ], capture_output=True, text=True, timeout=15)
        
        if "SERVER_CREATED" in result.stdout:
            print("SUCCESS: Real MCP server created")
            
            # Check for tools
            if "TOOLS_FOUND:" in result.stdout:
                tools_line = [line for line in result.stdout.split('\\n') if "TOOLS_FOUND:" in line][0]
                tools_count = tools_line.split(":")[1].strip()
                print(f"SUCCESS: Server has {tools_count} tools")
                return int(tools_count) > 0
            else:
                print("WARNING: No tools detected")
                return False
        else:
            print("FAILED: Could not create server")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Main ASCII-safe test"""
    print("TESTING REAL MCP (ASCII-SAFE)")
    print("No Unicode characters - Windows safe")
    print("=" * 50)
    
    imports_ok = test_ascii_safe_imports()
    server_ok = test_server_creation()
    
    print("\\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Imports Work: {'YES' if imports_ok else 'NO'}")
    print(f"Server Works: {'YES' if server_ok else 'NO'}")
    
    if imports_ok and server_ok:
        print("\\nSUCCESS: REAL MCP WORKING!")
        print("No Unicode crashes")
        print("Real FastMCP server with tools")
        print("\\nREADY TO LAUNCH:")
        print("  poetry run python enhanced_launcher.py")
        return True
    else:
        print("\\nISSUES DETECTED")
        print("Check errors above")
        return False


if __name__ == "__main__":
    main()
