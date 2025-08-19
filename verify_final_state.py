#!/usr/bin/env python3
"""
Final verification of all soliton fixes and repository state
"""

import os
import subprocess
import json
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = f"{Colors.GREEN}✅ EXISTS{Colors.END}" if exists else f"{Colors.RED}❌ MISSING{Colors.END}"
    print(f"{status} {description}: {filepath}")
    return exists

def check_file_contains(filepath, search_text, description):
    """Check if a file contains specific text"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            contains = search_text in content
            status = f"{Colors.GREEN}✅ FOUND{Colors.END}" if contains else f"{Colors.RED}❌ NOT FOUND{Colors.END}"
            print(f"{status} {description}")
            return contains
    except Exception as e:
        print(f"{Colors.RED}❌ ERROR{Colors.END} checking {filepath}: {e}")
        return False

def check_git_status():
    """Check for files that shouldn't be in git"""
    print(f"\n{Colors.BLUE}Checking for files that shouldn't be in git...{Colors.END}")
    
    bad_patterns = [
        "*.pyc", "*.log", "*.pkl", "*.egg-info", 
        "GREMLIN_*.ps1", "fix_soliton_*.ps1"
    ]
    
    issues = []
    for pattern in bad_patterns:
        try:
            result = subprocess.run(
                ["git", "ls-files", pattern],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                issues.append(f"{pattern}: {len(result.stdout.strip().split())} files")
        except:
            pass
    
    if issues:
        print(f"{Colors.YELLOW}⚠️  Found files that should be removed:{Colors.END}")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"{Colors.GREEN}✅ No problematic files in git{Colors.END}")
        return True

def main():
    print(f"{Colors.BOLD}{'='*60}")
    print("SOLITON FIXES FINAL VERIFICATION")
    print(f"{'='*60}{Colors.END}\n")
    
    results = {
        "production_code": 0,
        "test_infrastructure": 0,
        "repository_hygiene": 0,
        "total": 0
    }
    
    # Check production code fixes
    print(f"{Colors.BLUE}1. Production Code Fixes:{Colors.END}")
    
    if check_file_exists("api/routes/soliton.py", "Soliton API route"):
        results["production_code"] += 1
    results["total"] += 1
    
    if check_file_contains("api/routes/soliton.py", "diagnostic", "Diagnostic endpoint in soliton.py"):
        results["production_code"] += 1
    results["total"] += 1
    
    if check_file_exists("tori_ui_svelte/src/lib/services/solitonMemory.ts", "Frontend solitonMemory.ts"):
        results["production_code"] += 1
    results["total"] += 1
    
    if check_file_contains("tori_ui_svelte/src/lib/services/solitonMemory.ts", "STATS_RETRY_COOLDOWN", "Rate limiting in frontend"):
        results["production_code"] += 1
    results["total"] += 1
    
    # Check test infrastructure
    print(f"\n{Colors.BLUE}2. Test Infrastructure:{Colors.END}")
    
    if check_file_exists("tests/test_soliton_api.py", "Soliton API tests"):
        results["test_infrastructure"] += 1
    results["total"] += 1
    
    if check_file_exists(".github/workflows/build-concept-mesh.yml", "CI workflow"):
        results["test_infrastructure"] += 1
    results["total"] += 1
    
    if check_file_exists("requirements.lock", "Locked dependencies"):
        results["test_infrastructure"] += 1
    results["total"] += 1
    
    # Check repository hygiene
    print(f"\n{Colors.BLUE}3. Repository Hygiene:{Colors.END}")
    
    if check_file_exists(".gitignore", "Git ignore file"):
        results["repository_hygiene"] += 1
    results["total"] += 1
    
    if check_git_status():
        results["repository_hygiene"] += 1
    results["total"] += 1
    
    # Check for duplicate files
    duplicates = [
        "fixes/soliton_500_fixes/",
        "GREMLIN_HUNTER_MASTER.ps1",
        "fix_soliton_500_comprehensive.ps1"
    ]
    
    no_duplicates = True
    for dup in duplicates:
        if os.path.exists(dup):
            print(f"{Colors.YELLOW}⚠️  Duplicate found: {dup}{Colors.END}")
            no_duplicates = False
    
    if no_duplicates:
        print(f"{Colors.GREEN}✅ No duplicate fix scripts{Colors.END}")
        results["repository_hygiene"] += 1
    results["total"] += 1
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}{Colors.END}\n")
    
    total_passed = sum(v for k, v in results.items() if k != "total")
    
    print(f"Production Code Fixes: {results['production_code']}/4")
    print(f"Test Infrastructure: {results['test_infrastructure']}/3")
    print(f"Repository Hygiene: {results['repository_hygiene']}/3")
    print(f"\n{Colors.BOLD}Total: {total_passed}/{results['total']}{Colors.END}")
    
    if total_passed == results["total"]:
        print(f"\n{Colors.GREEN}✅ All fixes are properly applied!{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some fixes need attention.{Colors.END}")
        print(f"\nNext steps:")
        print("1. Run cleanup_repository.bat to remove unwanted files")
        print("2. Commit and push any missing files (.github/workflows/)")
        print("3. Re-run this verification")
    
    return 0 if total_passed == results["total"] else 1

if __name__ == "__main__":
    exit(main())
