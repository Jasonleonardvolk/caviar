#!/usr/bin/env python3
"""
Direct Python fix for NumPy ABI issues
This bypasses Poetry and fixes the issue directly
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def run_command(cmd, check=True, capture=True):
    """Run a command and return output"""
    print(f"{BLUE}Running: {cmd}{RESET}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=capture,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"{RED}Command failed: {result.stderr}{RESET}")
    return result

def main():
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}    DIRECT NUMPY ABI FIX - PYTHON VERSION{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")
    
    # Check if we're in venv
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"{YELLOW}WARNING: Not in virtual environment!{RESET}")
        print(f"{YELLOW}Please activate your venv first:{RESET}")
        print(f"  .venv\\Scripts\\Activate.ps1")
        return 1
    
    print(f"{GREEN}Virtual environment detected: {sys.prefix}{RESET}\n")
    
    # Step 1: Show current state
    print(f"{CYAN}[1/5] Current package versions:{RESET}")
    packages_to_check = ['numpy', 'spacy', 'thinc']
    current_versions = {}
    
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            current_versions[pkg] = version
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: not installed")
        except ValueError as e:
            if 'dtype size changed' in str(e):
                print(f"  {pkg}: ABI INCOMPATIBILITY ERROR")
            else:
                print(f"  {pkg}: error - {e}")
    
    # Check if NumPy 2.x is installed
    if 'numpy' in current_versions and current_versions['numpy'].startswith('2.'):
        print(f"\n{RED}{BOLD}PROBLEM DETECTED: NumPy {current_versions['numpy']} is incompatible with spaCy!{RESET}")
        print(f"{YELLOW}Must downgrade to NumPy 1.26.4{RESET}\n")
    
    # Step 2: Uninstall problematic packages
    print(f"\n{CYAN}[2/5] Uninstalling problematic packages...{RESET}")
    
    packages_to_remove = [
        'spacy',
        'spacy-legacy',
        'spacy-loggers',
        'thinc',
        'cymem',
        'murmurhash',
        'preshed',
        'numpy'
    ]
    
    for pkg in packages_to_remove:
        print(f"  Removing {pkg}...")
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}", check=False)
    
    # Step 3: Clear caches
    print(f"\n{CYAN}[3/5] Clearing caches...{RESET}")
    
    # Clear pip cache
    run_command(f"{sys.executable} -m pip cache purge", check=False)
    
    # Clear __pycache__ directories
    for pycache in Path('.').rglob('__pycache__'):
        try:
            shutil.rmtree(pycache)
            print(f"  Removed {pycache}")
        except:
            pass
    
    # Step 4: Install correct versions
    print(f"\n{CYAN}[4/5] Installing correct versions...{RESET}")
    
    # CRITICAL: Install NumPy 1.26.4 first
    print(f"\n{YELLOW}Installing NumPy 1.26.4 (THE CORRECT VERSION)...{RESET}")
    result = run_command(f"{sys.executable} -m pip install numpy==1.26.4 --no-cache-dir")
    
    if result.returncode != 0:
        print(f"{RED}Failed to install NumPy 1.26.4!{RESET}")
        return 1
    
    # Verify NumPy version
    try:
        import importlib
        import numpy
        importlib.reload(numpy)
        installed_version = numpy.__version__
        if installed_version != '1.26.4':
            print(f"{RED}ERROR: NumPy {installed_version} was installed instead of 1.26.4!{RESET}")
            return 1
        print(f"{GREEN}✓ NumPy 1.26.4 installed successfully{RESET}")
    except Exception as e:
        print(f"{RED}Error verifying NumPy: {e}{RESET}")
        return 1
    
    # Install other packages
    packages_to_install = [
        ('Cython', None),
        ('cymem', None),
        ('murmurhash', None),
        ('preshed', None),
        ('thinc', '8.2.4'),
        ('spacy', '3.7.5'),
    ]
    
    for pkg, version in packages_to_install:
        pkg_spec = f"{pkg}=={version}" if version else pkg
        print(f"\nInstalling {pkg_spec}...")
        result = run_command(f"{sys.executable} -m pip install {pkg_spec} --no-cache-dir")
        if result.returncode == 0:
            print(f"{GREEN}✓ {pkg} installed{RESET}")
        else:
            print(f"{RED}✗ {pkg} failed to install{RESET}")
    
    # Step 5: Final verification
    print(f"\n{CYAN}[5/5] Final verification...{RESET}")
    print("-" * 40)
    
    # Test imports
    test_passed = True
    
    # Test NumPy
    try:
        import numpy as np
        print(f"{GREEN}✓ NumPy {np.__version__} imports successfully{RESET}")
        if not np.__version__.startswith('1.26'):
            print(f"{RED}  WARNING: NumPy version is not 1.26.x{RESET}")
            test_passed = False
    except Exception as e:
        print(f"{RED}✗ NumPy import failed: {e}{RESET}")
        test_passed = False
    
    # Test spaCy
    try:
        import spacy
        print(f"{GREEN}✓ spaCy {spacy.__version__} imports successfully{RESET}")
        
        # Test functionality
        nlp = spacy.blank('en')
        doc = nlp("Test sentence")
        print(f"{GREEN}✓ spaCy functionality test passed{RESET}")
    except Exception as e:
        print(f"{RED}✗ spaCy failed: {e}{RESET}")
        test_passed = False
    
    # Test thinc
    try:
        import thinc
        print(f"{GREEN}✓ thinc {thinc.__version__} imports successfully{RESET}")
    except Exception as e:
        print(f"{RED}✗ thinc failed: {e}{RESET}")
        test_passed = False
    
    print("-" * 40)
    
    if test_passed:
        print(f"\n{GREEN}{BOLD}SUCCESS! All tests passed!{RESET}")
        print(f"\n{CYAN}Next steps:{RESET}")
        print(f"1. Download spaCy models:")
        print(f"   python -m spacy download en_core_web_sm")
        print(f"2. Test your application")
        return 0
    else:
        print(f"\n{RED}{BOLD}Some tests failed. See errors above.{RESET}")
        print(f"\n{YELLOW}If this didn't work, try:{RESET}")
        print(f"1. Delete the .venv folder completely")
        print(f"2. Create a new virtual environment:")
        print(f"   python -m venv .venv")
        print(f"3. Activate it and run this script again")
        return 1

if __name__ == "__main__":
    sys.exit(main())