#!/usr/bin/env python3
"""
TORI System Patch Verification Script
Checks if all required patches have been applied correctly
"""

import os
import sys
import json
import ast
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    return f"{color}{text}{Colors.ENDC}"

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def check_for_pattern(filepath, pattern):
    """Check if a pattern exists in file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return pattern in content
    except:
        return False

def verify_patches():
    """Verify all patches have been applied"""
    
    base_dir = Path(r"{PROJECT_ROOT}")
    results = []
    
    print(colored("\n" + "="*60, Colors.BOLD))
    print(colored("TORI SYSTEM PATCH VERIFICATION", Colors.BOLD))
    print(colored("="*60 + "\n", Colors.BOLD))
    
    # Check 1: Enhanced launcher has positional port passing
    test_name = "Enhanced Launcher - Dynamic Port Passing"
    filepath = base_dir / "enhanced_launcher.py"
    if check_for_pattern(filepath, "str(audio_port)  # Pass port as positional argument"):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - Port not passed as positional', Colors.RED)}")
    
    # Check 2: Audio bridge accepts positional port
    test_name = "Audio Bridge - Port Argument Support"
    filepath = base_dir / "audio_hologram_bridge.py"
    if check_for_pattern(filepath, "parser.add_argument('port', nargs='?'"):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - Missing positional port arg', Colors.RED)}")
    
    # Check 3: Audio bridge has correct handler signature
    test_name = "Audio Bridge - Handler Signature"
    if check_for_pattern(filepath, "async def handle_client(self, websocket, path)"):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - Missing path parameter', Colors.RED)}")
    
    # Check 4: Concept mesh bridge has fixes
    test_name = "Concept Mesh Bridge - Port & Handler"
    filepath = base_dir / "concept_mesh_hologram_bridge.py"
    has_port = check_for_pattern(filepath, "parser.add_argument('port', nargs='?'")
    has_handler = check_for_pattern(filepath, "async def handle_client(self, websocket, path)")
    if has_port and has_handler:
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        issues = []
        if not has_port: issues.append("port arg")
        if not has_handler: issues.append("handler signature")
        print(f"❌ {test_name}: {colored('FAIL - Missing: ' + ', '.join(issues), Colors.RED)}")
    
    # Check 5: API routes exist
    test_name = "API Routes - Concept Mesh Router"
    filepath = base_dir / "api" / "routes" / "concept_mesh.py"
    if check_file_exists(filepath):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - File not found', Colors.RED)}")
    
    # Check 6: Soliton routes exist
    test_name = "API Routes - Soliton Router"
    filepath = base_dir / "api" / "routes" / "soliton.py"
    if check_file_exists(filepath):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - File not found', Colors.RED)}")
    
    # Check 7: V1 router exists
    test_name = "API Routes - V1 Aggregator"
    filepath = base_dir / "api" / "routes" / "v1.py"
    if check_file_exists(filepath):
        has_concepts = check_for_pattern(filepath, "@api_v1_router.get(\"/concepts\")")
        if has_concepts:
            results.append((test_name, "PASS"))
            print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
        else:
            results.append((test_name, "PARTIAL"))
            print(f"⚠️ {test_name}: {colored('PARTIAL - Missing /concepts endpoint', Colors.YELLOW)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - File not found', Colors.RED)}")
    
    # Check 8: Test uses dynamic port
    test_name = "Test Script - Dynamic Port"
    filepath = base_dir / "test_concept_mesh_3_fixed.py"
    if check_for_pattern(filepath, "api_port.json"):
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - Still using hardcoded port', Colors.RED)}")
    
    # Check 9: __init__.py files exist
    test_name = "Python Package Structure"
    init_files = [
        base_dir / "prajna" / "__init__.py",
        base_dir / "prajna" / "api" / "__init__.py",
        base_dir / "api" / "__init__.py",
        base_dir / "api" / "routes" / "__init__.py"
    ]
    missing_inits = [f for f in init_files if not check_file_exists(f)]
    if not missing_inits:
        results.append((test_name, "PASS"))
        print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - Missing ' + str(len(missing_inits)) + ' __init__.py files', Colors.RED)}")
        for f in missing_inits:
            print(f"   Missing: {f}")
    
    # Check 10: Soliton endpoints
    test_name = "Soliton API Endpoints"
    filepath = base_dir / "api" / "routes" / "soliton_production.py"
    if check_file_exists(filepath):
        has_init = check_for_pattern(filepath, '@router.post("/init")')
        has_stats = check_for_pattern(filepath, '@router.get("/stats/')
        if has_init and has_stats:
            results.append((test_name, "PASS"))
            print(f"✅ {test_name}: {colored('PASS', Colors.GREEN)}")
        else:
            results.append((test_name, "PARTIAL"))
            missing = []
            if not has_init: missing.append("/init")
            if not has_stats: missing.append("/stats")
            print(f"⚠️ {test_name}: {colored('PARTIAL - Missing: ' + ', '.join(missing), Colors.YELLOW)}")
    else:
        results.append((test_name, "FAIL"))
        print(f"❌ {test_name}: {colored('FAIL - soliton_production.py not found', Colors.RED)}")
    
    # Summary
    print(colored("\n" + "="*60, Colors.BOLD))
    print(colored("VERIFICATION SUMMARY", Colors.BOLD))
    print(colored("="*60, Colors.BOLD))
    
    passed = sum(1 for _, status in results if status == "PASS")
    failed = sum(1 for _, status in results if status == "FAIL")
    partial = sum(1 for _, status in results if status == "PARTIAL")
    
    print(f"\n✅ Passed: {colored(str(passed), Colors.GREEN)}")
    print(f"⚠️ Partial: {colored(str(partial), Colors.YELLOW)}")
    print(f"❌ Failed: {colored(str(failed), Colors.RED)}")
    print(f"📊 Total: {len(results)}")
    
    if failed == 0 and partial == 0:
        print(colored("\n🎉 ALL PATCHES SUCCESSFULLY APPLIED!", Colors.GREEN + Colors.BOLD))
        print(colored("Your TORI system is ready to launch! 🚀", Colors.GREEN))
    elif failed == 0:
        print(colored("\n⚠️ MOSTLY READY - Some minor issues to fix", Colors.YELLOW + Colors.BOLD))
        print("Review the partial passes above and apply remaining patches.")
    else:
        print(colored(f"\n❌ {failed} CRITICAL PATCHES MISSING", Colors.RED + Colors.BOLD))
        print("Please apply the patches from TORI_PATCHES_COMPLETE.md")
    
    # Quick launch test
    print(colored("\n" + "="*60, Colors.BOLD))
    print(colored("QUICK LAUNCH COMMAND", Colors.BOLD))
    print(colored("="*60, Colors.BOLD))
    print("\nTo test your system after patches:")
    print(colored("python enhanced_launcher.py --api full --enable-hologram", Colors.GREEN))
    
    return passed, partial, failed

if __name__ == "__main__":
    try:
        passed, partial, failed = verify_patches()
        sys.exit(0 if failed == 0 else 1)
    except Exception as e:
        print(colored(f"\nERROR: {e}", Colors.RED))
        sys.exit(1)
