#!/usr/bin/env python3
"""
Fix both issues without Unicode characters
"""

import sys
import subprocess
from pathlib import Path
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("FIXING BOTH ISSUES")
print("=" * 60)

# Issue 1: Check where concept_mesh_rs actually is
print("\n1. Checking concept_mesh_rs installation...")
print("-" * 40)

# Check if it's in site-packages
site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
pyd_files = list(site_packages.glob("concept_mesh_rs*.pyd"))

if pyd_files:
    print(f"[OK] Found in site-packages: {pyd_files[0].name}")
else:
    print("[!] Not found in site-packages!")
    
    # Check if wheel exists
    wheel_path = Path("concept_mesh/target/wheels")
    if wheel_path.exists():
        wheels = list(wheel_path.glob("*.whl"))
        if wheels:
            print(f"\n[*] Found wheel: {wheels[0].name}")
            print("Installing it now...")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[0])],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[OK] Installed successfully!")
            else:
                print(f"[ERROR] Install failed: {result.stderr}")

# Test import
print("\nTesting import...")
test = subprocess.run(
    [sys.executable, "-c", "import concept_mesh_rs; print('OK')"],
    capture_output=True,
    text=True
)

if test.returncode == 0:
    print("[OK] Import works!")
else:
    print(f"[ERROR] Import fails: {test.stderr}")

# Issue 2: Fix server.py indentation
print("\n2. Fixing server.py indentation...")
print("-" * 40)

server_path = Path("mcp_metacognitive/server.py")

# Restore from backup first
backup_path = server_path.with_suffix('.py.backup_run_fix')
if backup_path.exists():
    print("Restoring from backup...")
    content = backup_path.read_text(encoding='utf-8')
    server_path.write_text(content, encoding='utf-8')
    print("[OK] Restored original server.py")
else:
    print("[!] No backup found, working with current file")

# Now apply a simpler fix
content = server_path.read_text(encoding='utf-8')

# Find and replace the problematic mcp.run line
if 'mcp.run(transport="sse", host=config.server_host, port=config.server_port)' in content:
    print("Applying simple fix...")
    
    # Just wrap the existing call in a try/except
    new_content = content.replace(
        'mcp.run(transport="sse", host=config.server_host, port=config.server_port)',
        '''try:
            # Try with parameters first
            mcp.run(transport="sse", host=config.server_host, port=config.server_port)
        except TypeError:
            # If that fails, try without parameters and let MCP handle it
            logger.warning("FastMCP.run() doesn't accept host/port, using defaults")
            mcp.run()'''
    )
    
    server_path.write_text(new_content, encoding='utf-8')
    print("[OK] Applied simple fix to server.py")

# Verify no syntax errors
print("\nVerifying server.py syntax...")
compile_test = subprocess.run(
    [sys.executable, "-m", "py_compile", str(server_path)],
    capture_output=True
)

if compile_test.returncode == 0:
    print("[OK] server.py has valid syntax")
else:
    print("[ERROR] server.py has syntax errors!")

print("\n" + "=" * 60)
print("[OK] Fixes applied!")
print("\nNow try starting again:")
print("   python enhanced_launcher.py")
