#!/usr/bin/env python3
"""
HONEST MCP STATUS CHECKER
No false positives - tells you the real truth about MCP status
"""

import subprocess
import sys


def honest_mcp_check():
    """Honest assessment of MCP package status"""
    print("🎯 HONEST MCP STATUS CHECK")
    print("=" * 40)
    print("No false positives - real truth only")
    print("=" * 40)
    
    # Test 1: Real MCP core import
    print("\n🔍 Testing REAL MCP packages...")
    
    mcp_tests = [
        ("mcp", "MCP core"),
        ("mcp.server", "MCP server"),
        ("mcp.server.fastmcp", "FastMCP"),
        ("mcp.types", "MCP types")
    ]
    
    real_mcp_working = True
    
    for module, name in mcp_tests:
        try:
            result = subprocess.run([
                sys.executable, '-c', f'''
try:
    import {module}
    print("REAL_SUCCESS")
except ImportError as e:
    print(f"REAL_FAIL: {{e}}")
'''
            ], capture_output=True, text=True, timeout=5)
            
            if "REAL_SUCCESS" in result.stdout:
                print(f"✅ {name}: REAL PACKAGE AVAILABLE")
            else:
                print(f"❌ {name}: NOT AVAILABLE")
                if "REAL_FAIL" in result.stdout:
                    error = result.stdout.split("REAL_FAIL: ")[1].strip()
                    print(f"   Error: {error}")
                real_mcp_working = False
        except Exception as e:
            print(f"❌ {name}: TEST FAILED - {e}")
            real_mcp_working = False
    
    # Test 2: Check what we actually have
    print("\n📋 WHAT WE ACTUALLY HAVE:")
    
    if real_mcp_working:
        print("✅ REAL MCP PACKAGES - Full functionality")
        status = "REAL"
    else:
        print("❌ NO REAL MCP PACKAGES")
        print("⚠️ Using fallback/mock implementations")
        print("💡 This means limited functionality")
        status = "FALLBACK"
    
    # Test 3: Check our server status
    print(f"\n🔍 OUR MCP SERVER STATUS:")
    try:
        result = subprocess.run([
            sys.executable, '-c', '''
import sys
import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

try:
    from mcp_metacognitive.server_fixed import main
    print("SERVER_IMPORT_OK")
except Exception as e:
    print(f"SERVER_IMPORT_FAIL: {e}")
'''
        ], capture_output=True, text=True, timeout=5)
        
        if "SERVER_IMPORT_OK" in result.stdout:
            print("✅ MCP server module loads")
            if "MCP packages not available" in result.stderr:
                print("⚠️ BUT server will use fallback mode")
                server_status = "FALLBACK"
            else:
                print("✅ Server should use real MCP")
                server_status = "REAL" if real_mcp_working else "FALLBACK"
        else:
            print("❌ MCP server module has issues")
            server_status = "BROKEN"
    except Exception as e:
        print(f"❌ Cannot test server: {e}")
        server_status = "UNKNOWN"
    
    print("\n" + "=" * 40)
    print("🎯 HONEST ASSESSMENT")
    print("=" * 40)
    print(f"MCP Packages: {status}")
    print(f"Server Status: {server_status}")
    
    if status == "REAL" and server_status == "REAL":
        print("\n🎉 PERFECT: Real MCP with full functionality")
        recommendation = "Launch normally - everything works"
    elif status == "FALLBACK" and server_status == "FALLBACK":
        print("\n⚠️ FALLBACK MODE: Limited mock functionality")
        recommendation = "Install real MCP packages for full features"
    elif server_status == "BROKEN":
        print("\n❌ BROKEN: Server has issues")
        recommendation = "Fix server imports first"
    else:
        print("\n🤔 MIXED STATUS: Partial functionality")
        recommendation = "Check individual components"
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    
    return status, server_status


if __name__ == "__main__":
    print("Looking for REAL MCP status (no false positives)...")
    print()
    
    mcp_status, server_status = honest_mcp_check()
    
    print(f"\n🎯 BOTTOM LINE:")
    if mcp_status == "REAL":
        print("✅ You have real MCP - launch away!")
    else:
        print("⚠️ You're using fallbacks - limited functionality")
        print("💡 Run: python install_real_mcp_packages.py")
    
    print(f"\n🚀 Launch command:")
    print(f"   poetry run python enhanced_launcher.py")
    if mcp_status == "FALLBACK":
        print("   (Will use fallback MCP server)")
