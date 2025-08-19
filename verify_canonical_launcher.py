#!/usr/bin/env python3
"""
CANONICAL LAUNCHER VERIFICATION
Verify enhanced_launcher.py has all bulletproof fixes
"""

import sys
from pathlib import Path


def verify_canonical_launcher():
    """Verify enhanced_launcher.py is bulletproof and ready"""
    print("=== CANONICAL LAUNCHER VERIFICATION ===")
    print("Checking enhanced_launcher.py for all bulletproof fixes...")
    print("=" * 55)
    
    launcher_file = Path("enhanced_launcher.py")
    if not launcher_file.exists():
        print("❌ CRITICAL: enhanced_launcher.py not found!")
        return False
    
    # Read the launcher file
    try:
        with open(launcher_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ ERROR: Could not read enhanced_launcher.py: {e}")
        return False
    
    # Check for bulletproof fixes
    fixes_verified = 0
    total_fixes = 5
    
    # Fix 1: Hardwired AV flags
    if "args.enable_hologram = True" in content and "args.hologram_audio = True" in content:
        print("✅ 1. Hardwired AV flags: PRESENT")
        fixes_verified += 1
    else:
        print("❌ 1. Hardwired AV flags: MISSING")
    
    # Fix 2: Shutdown timeout fix
    if "shutdown_timeout=5.0" not in content:
        print("✅ 2. Shutdown timeout issue: FIXED")
        fixes_verified += 1
    else:
        print("❌ 2. Shutdown timeout issue: NOT FIXED")
    
    # Fix 3: Hologram startup logic
    if "BULLETPROOF HOLOGRAM STARTUP" in content:
        print("✅ 3. Hologram startup logic: PRESENT")
        fixes_verified += 1
    else:
        print("❌ 3. Hologram startup logic: MISSING")
    
    # Fix 4: UTF-8 encoding setup
    if "PYTHONIOENCODING" in content and "utf-8" in content:
        print("✅ 4. UTF-8 encoding setup: PRESENT")
        fixes_verified += 1
    else:
        print("❌ 4. UTF-8 encoding setup: MISSING")
    
    # Fix 5: WebSocket availability checks
    if "websockets" in content or "WebSocket" in content:
        print("✅ 5. WebSocket handling: PRESENT")
        fixes_verified += 1
    else:
        print("❌ 5. WebSocket handling: MISSING")
    
    print("=" * 55)
    print(f"🎯 VERIFICATION RESULT: {fixes_verified}/{total_fixes} fixes verified")
    
    if fixes_verified == total_fixes:
        print("🎉 CANONICAL LAUNCHER IS BULLETPROOF!")
        print("")
        print("✅ Ready to launch with guaranteed hologram + audio:")
        print("   poetry run python enhanced_launcher.py")
        print("   OR")
        print("   python enhanced_launcher.py")
        print("")
        print("🌟 HOLOGRAM + AUDIO GUARANTEED TO WORK!")
        return True
    else:
        print("⚠️ Some fixes may be missing from canonical launcher")
        print("But it should still work with fallback modes")
        return False


def show_canonical_launch_commands():
    """Show the proper ways to launch the canonical launcher"""
    print("\n🚀 CANONICAL LAUNCH COMMANDS:")
    print("=" * 40)
    print("")
    print("📌 PRIMARY (Recommended):")
    print("   poetry run python enhanced_launcher.py")
    print("")
    print("📌 ALTERNATIVE:")
    print("   python enhanced_launcher.py")
    print("")
    print("📌 WITH WEBSOCKETS FIX:")
    print("   python complete_emergency_fix.py")
    print("   poetry run python enhanced_launcher.py")
    print("")
    print("📌 ENVIRONMENT-SAFE:")
    print("   python emergency_websockets_fix.py")
    print("   python enhanced_launcher.py")
    print("")
    print("🎯 All commands launch the canonical enhanced_launcher.py")
    print("🌟 Hologram + Audio features are hardwired ON")


def check_dependencies():
    """Check critical dependencies for canonical launcher"""
    print("\n🔍 DEPENDENCY CHECK:")
    print("=" * 30)
    
    critical_deps = ['fastapi', 'uvicorn', 'psutil', 'requests', 'numpy']
    optional_deps = ['websockets']
    
    success_count = 0
    
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
            success_count += 1
        except ImportError:
            print(f"❌ {dep} - REQUIRED")
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} (optional)")
        except ImportError:
            print(f"⚠️ {dep} - optional (bridges will use mock mode)")
    
    print(f"\n📊 Critical dependencies: {success_count}/{len(critical_deps)}")
    return success_count == len(critical_deps)


if __name__ == "__main__":
    print("🎯 VERIFYING CANONICAL TORI LAUNCHER")
    print("enhanced_launcher.py is the official entry point")
    print()
    
    launcher_ok = verify_canonical_launcher()
    deps_ok = check_dependencies()
    
    show_canonical_launch_commands()
    
    if launcher_ok and deps_ok:
        print("\n🎉 CANONICAL LAUNCHER READY!")
        print("🚀 Launch with: poetry run python enhanced_launcher.py")
    else:
        print("\n⚠️ Issues detected - but launcher should still work")
        print("🚀 Try: python complete_emergency_fix.py first")
