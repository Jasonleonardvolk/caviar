#!/usr/bin/env python3
"""
REAL MCP PACKAGES INSTALLER
Install the actual MCP dependencies instead of using fallbacks
"""

import subprocess
import sys
import time


def install_mcp_packages():
    """Install the real MCP packages"""
    print("🚀 INSTALLING REAL MCP PACKAGES")
    print("=" * 50)
    print("Installing actual MCP dependencies (not fallbacks)...")
    print()
    
    packages_to_install = [
        "mcp",
        # Add any other specific MCP packages needed
    ]
    
    success_count = 0
    
    for package in packages_to_install:
        print(f"📦 Installing {package}...")
        
        # Method 1: Poetry
        try:
            result = subprocess.run([
                'poetry', 'add', package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ {package} installed via poetry")
                success_count += 1
                continue
            else:
                print(f"⚠️ Poetry failed for {package}: {result.stderr[:100]}")
        except Exception as e:
            print(f"⚠️ Poetry not available: {e}")
        
        # Method 2: Pip fallback
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ {package} installed via pip")
                success_count += 1
            else:
                print(f"❌ Pip failed for {package}: {result.stderr[:100]}")
        except Exception as e:
            print(f"❌ Pip failed for {package}: {e}")
    
    return success_count, len(packages_to_install)


def verify_real_mcp_imports():
    """Verify REAL MCP packages are working (not fallbacks)"""
    print("\n🔍 VERIFYING REAL MCP PACKAGES")
    print("=" * 40)
    
    tests = [
        ("mcp", "MCP core package"),
        ("mcp.server", "MCP server module"),
        ("mcp.server.fastmcp", "FastMCP server")
    ]
    
    success_count = 0
    
    for module, description in tests:
        try:
            result = subprocess.run([
                sys.executable, '-c', f'import {module}; print("REAL IMPORT OK")'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "REAL IMPORT OK" in result.stdout:
                print(f"✅ {description}: REAL PACKAGE WORKING")
                success_count += 1
            else:
                print(f"❌ {description}: FAILED")
                if result.stderr:
                    print(f"   Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"❌ {description}: FAILED - {e}")
    
    return success_count == len(tests)


def check_mcp_availability():
    """Check what MCP options are actually available"""
    print("\n📋 MCP AVAILABILITY CHECK")
    print("=" * 30)
    
    # Check if MCP is available through different channels
    options = []
    
    # Option 1: PyPI
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'search', 'mcp'
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            options.append("PyPI (pip search worked)")
    except:
        pass
    
    # Option 2: Check if it's a private/GitHub package
    print("🔍 Checking MCP package sources...")
    print("   1. PyPI repository")
    print("   2. GitHub repositories") 
    print("   3. Private package indexes")
    print("   4. Local development packages")
    
    # Try to find the correct MCP package
    possible_packages = [
        "mcp-server",
        "python-mcp", 
        "fastmcp",
        "model-context-protocol",
        "anthropic-mcp"
    ]
    
    for pkg in possible_packages:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'show', pkg
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Found: {pkg}")
                return pkg
        except:
            pass
    
    print("⚠️ Standard MCP packages not found in current environment")
    return None


def main():
    """Main MCP package installer"""
    print("🎯 REAL MCP PACKAGES INSTALLER")
    print("Stop using fallbacks - install real MCP!")
    print("=" * 50)
    
    # Check what's available
    found_package = check_mcp_availability()
    
    if found_package:
        print(f"\n💡 Found MCP package: {found_package}")
        print("Try installing this specific package")
    
    # Try to install standard packages
    installed, total = install_mcp_packages()
    
    # Verify real imports work
    real_imports_work = verify_real_mcp_imports()
    
    print("\n" + "=" * 50)
    print("🎯 MCP INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"📦 Packages installed: {installed}/{total}")
    print(f"🔍 Real imports work: {'✅ YES' if real_imports_work else '❌ NO'}")
    
    if real_imports_work:
        print("\n🎉 SUCCESS: REAL MCP PACKAGES WORKING!")
        print("✅ No more fallback mode")
        print("✅ Full MCP functionality available")
        print("\n🚀 Ready to launch with real MCP:")
        print("   poetry run python enhanced_launcher.py")
    else:
        print("\n⚠️ MCP PACKAGES NOT PROPERLY INSTALLED")
        print("💡 Possible solutions:")
        print("   1. Check if MCP is available in your package repository")
        print("   2. Install from specific source (GitHub, private repo)")
        print("   3. Use TORI without MCP server (still functional)")
        print("\n🚀 TORI will still work with fallback mode:")
        print("   poetry run python enhanced_launcher.py")
        print("   (MCP server will use mock implementation)")


if __name__ == "__main__":
    main()
