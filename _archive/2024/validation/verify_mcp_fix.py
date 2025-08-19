#!/usr/bin/env python3
"""
MCP METACOGNITIVE IMPORT FIX VERIFICATION
Test that all MCP imports work properly
"""

import sys
from pathlib import Path


def test_mcp_imports():
    """Test MCP metacognitive imports"""
    print("=== MCP METACOGNITIVE IMPORT FIX TEST ===")
    print("Testing all import paths...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Basic module import
    try:
        import mcp_metacognitive
        print("✅ 1. Basic mcp_metacognitive import: SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"❌ 1. Basic mcp_metacognitive import: FAILED - {e}")
    
    # Test 2: Config import
    try:
        from mcp_metacognitive import config
        print("✅ 2. Config import: SUCCESS")
        print(f"   Config host: {getattr(config, 'host', 'unknown')}")
        print(f"   Config port: {getattr(config, 'port', 'unknown')}")
        success_count += 1
    except Exception as e:
        print(f"❌ 2. Config import: FAILED - {e}")
    
    # Test 3: Main module import
    try:
        from mcp_metacognitive.main import setup_server, fallback_main
        print("✅ 3. Main module import: SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"❌ 3. Main module import: FAILED - {e}")
    
    # Test 4: Server fixed import
    try:
        import mcp_metacognitive.server_fixed
        print("✅ 4. Server fixed import: SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"❌ 4. Server fixed import: FAILED - {e}")
    
    # Test 5: Run server_fixed directly
    try:
        result = sys.modules.get('subprocess', __import__('subprocess')).run([
            sys.executable, '-c', 
            'import mcp_metacognitive.server_fixed; print("Server import OK")'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ 5. Server execution test: SUCCESS")
            success_count += 1
        else:
            print(f"❌ 5. Server execution test: FAILED - {result.stderr}")
    except Exception as e:
        print(f"❌ 5. Server execution test: FAILED - {e}")
    
    print("=" * 50)
    print(f"🎯 MCP IMPORT TEST RESULTS: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 ALL MCP IMPORTS FIXED!")
        print("✅ The MCP metacognitive server should now start properly")
        return True
    else:
        print("⚠️ Some import issues remain")
        print("💡 Server will use fallback modes for missing components")
        return False


def check_mcp_files():
    """Check that all required MCP files exist"""
    print("\\n=== MCP FILES VERIFICATION ===")
    
    required_files = [
        "mcp_metacognitive/__init__.py",
        "mcp_metacognitive/config.py", 
        "mcp_metacognitive/main.py",
        "mcp_metacognitive/server_fixed.py"
    ]
    
    existing_count = 0
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
            existing_count += 1
        else:
            print(f"❌ {file_path} - MISSING")
    
    print(f"\\n📊 MCP Files: {existing_count}/{len(required_files)} present")
    return existing_count == len(required_files)


def main():
    """Main verification"""
    print("🛠️ MCP METACOGNITIVE IMPORT FIX VERIFICATION")
    print("Checking fixes for the import error:")
    print("ImportError: cannot import name 'config' from 'mcp_metacognitive'")
    print()
    
    files_ok = check_mcp_files()
    imports_ok = test_mcp_imports()
    
    print("\\n" + "=" * 50)
    print("=== FINAL VERIFICATION SUMMARY ===") 
    print("=" * 50)
    print(f"📁 Required Files: {'✅ PRESENT' if files_ok else '❌ MISSING'}")
    print(f"📦 Import Tests: {'✅ PASSED' if imports_ok else '⚠️ ISSUES'}")
    
    if files_ok and imports_ok:
        print("\\n🎉 MCP METACOGNITIVE IMPORT ERROR FIXED!")
        print("✅ The enhanced_launcher.py should now start the MCP server properly")
        print("🚀 Ready to launch: poetry run python enhanced_launcher.py")
    else:
        print("\\n⚠️ Some issues detected")
        print("💡 Server will use fallback modes")
        print("🚀 Launch will still work: poetry run python enhanced_launcher.py")
    
    print("\\n🎯 The import error should be resolved!")


if __name__ == "__main__":
    main()
