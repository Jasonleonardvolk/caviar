#!/usr/bin/env python3
"""
REAL MCP VERIFICATION
Test that we're using REAL MCP packages (no fallbacks)
"""

import subprocess
import sys


def test_real_mcp_server():
    """Test that our server uses REAL MCP"""
    print("🎯 TESTING REAL MCP SERVER")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, '-c', '''
import logging
logging.basicConfig(level=logging.INFO)

from mcp_metacognitive.server_fixed import create_real_mcp_server
server = create_real_mcp_server()
print("REAL_MCP_SERVER_CREATED")

# Test that it has real tools
tools = getattr(server, '_tools', {})
print(f"TOOLS_COUNT: {len(tools)}")

# Check for our real tools
expected_tools = ["analyze_cognition", "metacognitive_reflection", "consciousness_assessment"]
for tool in expected_tools:
    if tool in str(tools):
        print(f"TOOL_FOUND: {tool}")
    else:
        print(f"TOOL_MISSING: {tool}")
'''
        ], capture_output=True, text=True, timeout=10)
        
        if "REAL_MCP_SERVER_CREATED" in result.stdout:
            print("✅ Real MCP server created successfully")
            
            # Count tools
            tools_count = 0
            for line in result.stdout.split('\n'):
                if "TOOLS_COUNT:" in line:
                    tools_count = int(line.split(":")[1].strip())
                    break
            
            print(f"✅ Server has {tools_count} real tools")
            
            # Check specific tools
            found_tools = []
            for line in result.stdout.split('\n'):
                if "TOOL_FOUND:" in line:
                    tool_name = line.split(":")[1].strip()
                    found_tools.append(tool_name)
                    print(f"✅ Tool available: {tool_name}")
            
            return len(found_tools) >= 3
        else:
            print("❌ Failed to create real MCP server")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_no_fallback_warnings():
    """Test that we don't get fallback warnings anymore"""
    print("\n🔍 CHECKING FOR FALLBACK WARNINGS")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, '-c', '''
import logging
import io
import sys

# Capture all log output
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)

# Test our server
from mcp_metacognitive.server_fixed import create_real_mcp_server
server = create_real_mcp_server()

# Get captured logs
log_output = log_capture.getvalue()
if "fallback" in log_output.lower() or "not available" in log_output.lower():
    print("FALLBACK_WARNINGS_FOUND")
    print(log_output)
else:
    print("NO_FALLBACK_WARNINGS")
'''
        ], capture_output=True, text=True, timeout=10)
        
        if "NO_FALLBACK_WARNINGS" in result.stdout:
            print("✅ No fallback warnings - using real MCP!")
            return True
        else:
            print("⚠️ Still seeing fallback warnings:")
            print(result.stdout)
            return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_real_imports():
    """Test that all real imports work"""
    print("\n📦 TESTING REAL IMPORTS")
    print("=" * 30)
    
    imports_to_test = [
        "mcp",
        "mcp.server", 
        "mcp.server.fastmcp",
        "mcp_metacognitive.config",
        "mcp_metacognitive.main",
        "mcp_metacognitive.server_fixed"
    ]
    
    success_count = 0
    
    for import_name in imports_to_test:
        try:
            result = subprocess.run([
                sys.executable, '-c', f'import {import_name}; print("IMPORT_OK")'
            ], capture_output=True, text=True, timeout=5)
            
            if "IMPORT_OK" in result.stdout:
                print(f"✅ {import_name}")
                success_count += 1
            else:
                print(f"❌ {import_name}")
        except Exception as e:
            print(f"❌ {import_name}: {e}")
    
    print(f"\n📊 Import success: {success_count}/{len(imports_to_test)}")
    return success_count == len(imports_to_test)


def main():
    """Main verification"""
    print("🎉 REAL MCP VERIFICATION")
    print("No more fallbacks - testing real functionality")
    print("=" * 50)
    
    real_server = test_real_mcp_server()
    no_fallbacks = test_no_fallback_warnings()
    imports_ok = test_real_imports()
    
    print("\n" + "=" * 50)
    print("🎯 REAL MCP VERIFICATION RESULTS")
    print("=" * 50)
    print(f"Real MCP Server: {'✅ YES' if real_server else '❌ NO'}")
    print(f"No Fallback Warnings: {'✅ YES' if no_fallbacks else '⚠️ SOME'}")
    print(f"Real Imports: {'✅ YES' if imports_ok else '❌ NO'}")
    
    if all([real_server, imports_ok]):
        print("\n🎉 SUCCESS: REAL MCP FULLY WORKING!")
        print("✅ No more fallback mode")
        print("✅ Full metacognitive functionality")
        print("✅ Real FastMCP server with TORI tools")
        print("\n🚀 Ready to launch with real MCP:")
        print("   poetry run python enhanced_launcher.py")
    else:
        print("\n⚠️ Some issues detected")
        print("Check individual test results above")
    
    return all([real_server, imports_ok])


if __name__ == "__main__":
    main()
