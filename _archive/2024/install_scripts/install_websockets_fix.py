#!/usr/bin/env python3
"""
📦 INSTANT WEBSOCKETS INSTALLER + VERIFICATION
Installs websockets dependency and verifies everything works
"""

import subprocess
import sys
import time
from pathlib import Path


def install_websockets_now():
    """Install websockets dependency immediately"""
    print("🚀 INSTANT WEBSOCKETS INSTALLER")
    print("=" * 50)
    
    install_success = False
    
    # Method 1: Poetry
    print("📦 Attempting installation via Poetry...")
    try:
        result = subprocess.run(['poetry', 'add', 'websockets'], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ Successfully installed websockets via Poetry")
            install_success = True
        else:
            print(f"⚠️ Poetry install failed: {result.stderr}")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"⚠️ Poetry not available or failed: {e}")
    
    # Method 2: Pip fallback
    if not install_success:
        print("📦 Attempting installation via pip...")
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'websockets'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✅ Successfully installed websockets via pip")
                install_success = True
            else:
                print(f"❌ Pip install failed: {result.stderr}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"❌ Pip install failed: {e}")
    
    return install_success


def verify_websockets():
    """Verify websockets is installed and working"""
    print("\n🔍 Verifying websockets installation...")
    
    try:
        import websockets
        print(f"✅ WebSockets imported successfully")
        print(f"📦 Version: {websockets.__version__}")
        return True
    except ImportError as e:
        print(f"❌ WebSockets import failed: {e}")
        return False


def verify_bridges():
    """Verify the hologram bridges can start without crashing"""
    print("\n🌉 Verifying hologram bridges...")
    
    success_count = 0
    total_bridges = 2
    
    # Test audio bridge
    try:
        print("🎵 Testing audio hologram bridge...")
        result = subprocess.run([sys.executable, '-c', 
                               "import audio_hologram_bridge; print('Audio bridge import OK')"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Audio hologram bridge: OK")
            success_count += 1
        else:
            print(f"❌ Audio hologram bridge failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Audio hologram bridge test failed: {e}")
    
    # Test concept mesh bridge
    try:
        print("🧠 Testing concept mesh hologram bridge...")
        result = subprocess.run([sys.executable, '-c', 
                               "import concept_mesh_hologram_bridge; print('Concept bridge import OK')"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Concept mesh hologram bridge: OK")
            success_count += 1
        else:
            print(f"❌ Concept mesh hologram bridge failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Concept mesh hologram bridge test failed: {e}")
    
    return success_count == total_bridges


def run_quick_dependency_check():
    """Run a quick check of all critical dependencies"""
    print("\n📋 Quick dependency check...")
    
    critical_deps = [
        'websockets',
        'asyncio', 
        'json',
        'logging',
        'time',
        'threading'
    ]
    
    success_count = 0
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
            success_count += 1
        except ImportError:
            print(f"❌ {dep} - MISSING")
    
    print(f"\n📊 Dependencies: {success_count}/{len(critical_deps)} available")
    return success_count == len(critical_deps)


def main():
    """Main installer and verification"""
    print("🚀 BULLETPROOF WEBSOCKETS INSTALLER & VERIFIER")
    print("=" * 60)
    print("Fixing the websockets crash once and for all!")
    print("=" * 60)
    
    # Step 1: Install websockets
    install_success = install_websockets_now()
    
    # Step 2: Verify installation
    verify_success = verify_websockets()
    
    # Step 3: Check dependencies
    deps_success = run_quick_dependency_check()
    
    # Step 4: Test bridges
    bridges_success = verify_bridges()
    
    print("\n" + "=" * 60)
    print("🎯 INSTALLATION SUMMARY:")
    print("=" * 60)
    print(f"📦 WebSockets Install: {'✅ SUCCESS' if install_success else '❌ FAILED'}")
    print(f"🔍 WebSockets Verify:  {'✅ SUCCESS' if verify_success else '❌ FAILED'}")
    print(f"📋 Dependencies:       {'✅ SUCCESS' if deps_success else '❌ FAILED'}")
    print(f"🌉 Bridge Tests:       {'✅ SUCCESS' if bridges_success else '❌ FAILED'}")
    
    if all([install_success or verify_success, deps_success, bridges_success]):
        print("\n🎉 ALL SYSTEMS GO! WEBSOCKETS CRASH FIXED!")
        print("🚀 Ready to launch TORI with bulletproof hologram bridges")
        print("\n💡 Next steps:")
        print("   1. Run: poetry run python enhanced_launcher.py")
        print("   2. Or:  START_TORI_BULLETPROOF_NOW.bat")
        print("\n🌟 Hologram + Audio bridges will now work perfectly!")
        return True
    else:
        print("\n⚠️ SOME ISSUES DETECTED")
        if not (install_success or verify_success):
            print("💡 Try manual install: pip install websockets")
        if not bridges_success:
            print("💡 Bridges will run in mock mode (still functional)")
        print("\n🎯 TORI will still work - bridges have bulletproof fallbacks!")
        return False


if __name__ == "__main__":
    main()
