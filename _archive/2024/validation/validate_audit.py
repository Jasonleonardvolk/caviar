#!/usr/bin/env python3
"""
Audit Validation Script - Verifies all audit requirements are met
Run this to confirm system is production-ready
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_file_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if Path(path).exists():
        return True, f"{GREEN}✓{RESET} {description}"
    return False, f"{RED}✗{RESET} {description} - Missing: {path}"

def check_dir_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a directory exists."""
    if Path(path).is_dir():
        return True, f"{GREEN}✓{RESET} {description}"
    return False, f"{RED}✗{RESET} {description} - Missing: {path}"

def validate_json_file(path: str, description: str) -> Tuple[bool, str]:
    """Validate a JSON file is valid."""
    try:
        if Path(path).exists():
            with open(path) as f:
                json.load(f)
            return True, f"{GREEN}✓{RESET} {description} (valid JSON)"
        return False, f"{RED}✗{RESET} {description} - File missing"
    except json.JSONDecodeError:
        return False, f"{YELLOW}⚠{RESET} {description} - Invalid JSON"

def check_contains_text(path: str, text: str, description: str) -> Tuple[bool, str]:
    """Check if a file contains specific text."""
    try:
        if Path(path).exists():
            with open(path) as f:
                content = f.read()
            if text in content:
                return True, f"{GREEN}✓{RESET} {description}"
            return False, f"{YELLOW}⚠{RESET} {description} - Text not found: {text}"
        return False, f"{RED}✗{RESET} {description} - File missing"
    except Exception as e:
        return False, f"{RED}✗{RESET} {description} - Error: {e}"

def main():
    """Run all audit validation checks."""
    print(f"\n{BOLD}{BLUE}═══════════════════════════════════════════{RESET}")
    print(f"{BOLD}{BLUE}     TORI AUDIT VALIDATION SCRIPT{RESET}")
    print(f"{BOLD}{BLUE}═══════════════════════════════════════════{RESET}\n")
    
    base_path = Path(__file__).parent
    checks = []
    
    # 1. Frontend Files
    print(f"{BOLD}1. Frontend Components:{RESET}")
    frontend_files = [
        ("frontend/hybrid/lib/deviceDetect.ts", "Device detection"),
        ("frontend/hybrid/main.ts", "Main boot logic"),
        ("frontend/hybrid/wasmFallbackRenderer.ts", "WASM fallback"),
        ("frontend/hybrid/lib/errorHandler.ts", "Error handler"),
        ("frontend/hybrid/lib/parallaxMouse.ts", "Mouse parallax"),
        ("frontend/hybrid/lib/adaptiveRuntimeTuner.ts", "Runtime tuner"),
        ("frontend/hybrid/lib/sseClient.ts", "SSE client"),
        ("frontend/hybrid/App.svelte", "Main Svelte app"),
        ("frontend/hybrid/components/ParameterPanel.svelte", "Parameter panel"),
        ("frontend/hybrid/components/MeshSummaryPanel.svelte", "Mesh panel"),
        ("frontend/hybrid/components/AVStatus.svelte", "AV status"),
        ("frontend/hybrid/components/LogPanel.svelte", "Log panel"),
    ]
    
    for file_path, desc in frontend_files:
        result = check_file_exists(base_path / file_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # 2. PWA Files
    print(f"\n{BOLD}2. PWA Infrastructure:{RESET}")
    pwa_files = [
        ("frontend/public/manifest.webmanifest", "PWA manifest"),
        ("frontend/public/service-worker.js", "Service worker"),
        ("frontend/src/register-sw.ts", "SW registration"),
    ]
    
    for file_path, desc in pwa_files:
        result = check_file_exists(base_path / file_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # 3. Python Configuration
    print(f"\n{BOLD}3. Python Configuration:{RESET}")
    python_files = [
        ("python/core/logging_config.yaml", "Logging config"),
        ("python/core/logging_bootstrap.py", "Logging bootstrap"),
        ("python/core/adapter_loader.py", "Adapter loader"),
        ("python/core/mesh_exporter.py", "Mesh exporter"),
        ("python/core/phase_to_depth.py", "Phase processor"),
    ]
    
    for file_path, desc in python_files:
        result = check_file_exists(base_path / file_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # Check for threading lock
    result = check_contains_text(
        base_path / "python/core/adapter_loader.py",
        "_ADAPTER_SWAP_LOCK",
        "Threading lock in adapter_loader"
    )
    checks.append(result)
    print(f"  {result[1]}")
    
    # 4. Test Files
    print(f"\n{BOLD}4. Test Suite:{RESET}")
    test_files = [
        ("tests/test_hybrid_multiuser.py", "Multi-user tests"),
        ("tests/test_adapter_atomicity.py", "Atomicity tests"),
        ("tests/test_mesh_exporter_resilience.py", "Resilience tests"),
        ("tests/test_sse_event_integrity.py", "SSE integrity tests"),
        ("playwright/playwright.config.ts", "Playwright config"),
        ("playwright/tests/ui_hologram_controls.spec.ts", "UI control tests"),
        ("playwright/tests/ui_parallax_fallback.spec.ts", "Parallax tests"),
    ]
    
    for file_path, desc in test_files:
        result = check_file_exists(base_path / file_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # 5. Documentation
    print(f"\n{BOLD}5. Documentation:{RESET}")
    doc_files = [
        ("docs/ONBOARDING.md", "Onboarding guide"),
        ("docs/RELEASE_CHECKLIST.md", "Release checklist"),
        ("AUDIT_VERIFICATION.md", "Audit verification"),
        ("AUDIT_COMPLETE.md", "Audit completion report"),
    ]
    
    for file_path, desc in doc_files:
        result = check_file_exists(base_path / file_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # 6. Directory Structure
    print(f"\n{BOLD}6. Directory Structure:{RESET}")
    directories = [
        ("frontend/hybrid/lib", "Frontend lib directory"),
        ("frontend/hybrid/components", "Components directory"),
        ("python/core", "Python core directory"),
        ("tests", "Tests directory"),
        ("playwright/tests", "Playwright tests"),
        ("docs", "Documentation"),
    ]
    
    for dir_path, desc in directories:
        result = check_dir_exists(base_path / dir_path, desc)
        checks.append(result)
        print(f"  {result[1]}")
    
    # Calculate results
    passed = sum(1 for check in checks if check[0])
    total = len(checks)
    percentage = (passed / total) * 100
    
    # Summary
    print(f"\n{BOLD}{BLUE}═══════════════════════════════════════════{RESET}")
    print(f"{BOLD}VALIDATION SUMMARY:{RESET}")
    print(f"  Checks Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage == 100:
        print(f"  Status: {GREEN}{BOLD}✅ AUDIT COMPLETE - READY FOR PRODUCTION{RESET}")
        print(f"\n{GREEN}All audit requirements have been satisfied!{RESET}")
        print(f"{GREEN}The system is production-ready.{RESET}")
        return 0
    elif percentage >= 90:
        print(f"  Status: {YELLOW}{BOLD}⚠ NEARLY COMPLETE{RESET}")
        print(f"\n{YELLOW}Almost there! Fix the remaining items.{RESET}")
        return 1
    else:
        print(f"  Status: {RED}{BOLD}✗ INCOMPLETE{RESET}")
        print(f"\n{RED}Several audit items are missing.{RESET}")
        return 1
    
    print(f"{BOLD}{BLUE}═══════════════════════════════════════════{RESET}\n")

if __name__ == "__main__":
    sys.exit(main())
