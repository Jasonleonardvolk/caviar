#!/usr/bin/env python
"""
Update pyproject.toml to Pin Numpy Version
===========================================
This script updates the pyproject.toml file to pin numpy to a compatible version,
preventing future ABI compatibility issues.

Author: Enhanced Assistant
Date: 2025-08-06
"""

import sys
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import toml
from typing import Dict, Any, Optional

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

def get_current_numpy_version() -> Optional[str]:
    """Get the currently installed numpy version"""
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None

def get_recommended_numpy_version() -> str:
    """
    Get the recommended numpy version based on current environment.
    
    Returns the most stable, widely compatible version.
    """
    current_version = get_current_numpy_version()
    
    if current_version:
        # Parse major.minor version
        parts = current_version.split('.')
        if len(parts) >= 2:
            major = parts[0]
            minor = parts[1]
            
            # For numpy 1.x, recommend latest 1.26.x
            if major == "1":
                if int(minor) >= 26:
                    return "~1.26.0"
                elif int(minor) >= 24:
                    return "~1.24.0"
                else:
                    return "~1.23.0"
            # For numpy 2.x, recommend latest 2.0.x
            elif major == "2":
                return "~2.0.0"
    
    # Default recommendation
    return "~1.26.0"

def check_dependency_compatibility(numpy_version: str) -> Dict[str, Any]:
    """
    Check if other dependencies are compatible with the numpy version.
    """
    compatibility = {
        "compatible": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Parse numpy version
    version_match = re.match(r'[~^]?(\d+)\.(\d+)', numpy_version)
    if not version_match:
        compatibility["warnings"].append(f"Could not parse numpy version: {numpy_version}")
        return compatibility
    
    major = int(version_match.group(1))
    minor = int(version_match.group(2))
    
    # Check for known compatibility issues
    try:
        import importlib.metadata
        
        # Check spacy
        try:
            spacy_version = importlib.metadata.version('spacy')
            if spacy_version:
                spacy_major = int(spacy_version.split('.')[0])
                if spacy_major < 3 and major >= 2:
                    compatibility["warnings"].append(
                        f"spacy {spacy_version} may not be compatible with numpy 2.x"
                    )
                    compatibility["recommendations"].append(
                        "Consider upgrading spacy to 3.x or using numpy 1.26.x"
                    )
        except:
            pass
        
        # Check tensorflow
        try:
            tf_version = importlib.metadata.version('tensorflow')
            if tf_version:
                tf_major = int(tf_version.split('.')[0])
                if tf_major < 2 and major >= 2:
                    compatibility["warnings"].append(
                        f"tensorflow {tf_version} may not be compatible with numpy 2.x"
                    )
        except:
            pass
        
        # Check pandas
        try:
            pandas_version = importlib.metadata.version('pandas')
            if pandas_version:
                pandas_major = int(pandas_version.split('.')[0])
                pandas_minor = int(pandas_version.split('.')[1])
                if pandas_major < 2 and major >= 2:
                    compatibility["warnings"].append(
                        f"pandas {pandas_version} may require numpy < 2.0"
                    )
        except:
            pass
            
    except ImportError:
        compatibility["warnings"].append("Could not check package compatibility (importlib.metadata not available)")
    
    if compatibility["warnings"]:
        compatibility["compatible"] = False
    
    return compatibility

def update_pyproject_toml(numpy_version: str, force: bool = False) -> bool:
    """
    Update pyproject.toml with the specified numpy version.
    
    Args:
        numpy_version: The numpy version specification (e.g., "~1.26.0")
        force: Force update even if there are compatibility warnings
    
    Returns:
        True if successful, False otherwise
    """
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        print_error("pyproject.toml not found in current directory!")
        return False
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"pyproject.toml.backup_{timestamp}"
    
    print_info(f"Creating backup at: {backup_path}")
    shutil.copy2(pyproject_path, backup_path)
    
    try:
        # Load current pyproject.toml
        with open(pyproject_path, 'r') as f:
            data = toml.load(f)
        
        print_info("Current pyproject.toml loaded successfully")
        
        # Check if dependencies section exists
        if 'tool' not in data:
            data['tool'] = {}
        if 'poetry' not in data['tool']:
            data['tool']['poetry'] = {}
        if 'dependencies' not in data['tool']['poetry']:
            data['tool']['poetry']['dependencies'] = {}
        
        dependencies = data['tool']['poetry']['dependencies']
        
        # Check current numpy specification
        current_numpy = dependencies.get('numpy', None)
        if current_numpy:
            print_info(f"Current numpy specification: {current_numpy}")
        else:
            print_info("No numpy specification found in dependencies")
        
        # Update numpy version
        dependencies['numpy'] = numpy_version
        print_success(f"Updated numpy specification to: {numpy_version}")
        
        # Also check and update dev dependencies if they exist
        if 'dev-dependencies' in data['tool']['poetry']:
            dev_deps = data['tool']['poetry']['dev-dependencies']
            if 'numpy' in dev_deps:
                dev_deps['numpy'] = numpy_version
                print_success(f"Also updated numpy in dev-dependencies")
        
        # Check for group dependencies (newer Poetry format)
        if 'group' in data['tool']['poetry']:
            for group_name, group_data in data['tool']['poetry']['group'].items():
                if 'dependencies' in group_data and 'numpy' in group_data['dependencies']:
                    group_data['dependencies']['numpy'] = numpy_version
                    print_success(f"Also updated numpy in group '{group_name}' dependencies")
        
        # Save updated pyproject.toml
        with open(pyproject_path, 'w') as f:
            toml.dump(data, f)
        
        print_success("pyproject.toml updated successfully!")
        
        # Verify the file is valid
        try:
            with open(pyproject_path, 'r') as f:
                toml.load(f)
            print_success("Updated pyproject.toml is valid")
        except Exception as e:
            print_error(f"Updated pyproject.toml may be invalid: {e}")
            print_warning(f"Restoring from backup: {backup_path}")
            shutil.copy2(backup_path, pyproject_path)
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Failed to update pyproject.toml: {e}")
        print_warning(f"Backup available at: {backup_path}")
        return False

def main():
    """Main function to run the update process"""
    print_header("PYPROJECT.TOML NUMPY VERSION UPDATER")
    
    # Check if pyproject.toml exists
    if not Path("pyproject.toml").exists():
        print_error("pyproject.toml not found!")
        print_info("Please run this script from your project root directory.")
        return 1
    
    # Get current numpy version
    current_version = get_current_numpy_version()
    if current_version:
        print_info(f"Currently installed numpy version: {current_version}")
    else:
        print_warning("Numpy is not currently installed")
    
    # Get recommended version
    recommended_version = get_recommended_numpy_version()
    print_info(f"Recommended numpy version specification: {recommended_version}")
    
    # Allow user to specify version
    print("\nOptions:")
    print("1. Use recommended version:", recommended_version)
    print("2. Pin to current version:", f"^{current_version}" if current_version else "Not available")
    print("3. Use numpy 1.26.x (most compatible):", "~1.26.0")
    print("4. Use numpy 1.24.x (older, stable):", "~1.24.0")
    print("5. Use numpy 2.0.x (latest):", "~2.0.0")
    print("6. Enter custom version")
    print("7. Cancel")
    
    choice = input("\nSelect option (1-7): ").strip()
    
    if choice == "1":
        selected_version = recommended_version
    elif choice == "2":
        if current_version:
            selected_version = f"^{current_version}"
        else:
            print_error("No current version available")
            return 1
    elif choice == "3":
        selected_version = "~1.26.0"
    elif choice == "4":
        selected_version = "~1.24.0"
    elif choice == "5":
        selected_version = "~2.0.0"
    elif choice == "6":
        selected_version = input("Enter numpy version specification (e.g., ~1.26.0, ^1.25.0): ").strip()
        if not selected_version:
            print_error("No version specified")
            return 1
    elif choice == "7":
        print_info("Update cancelled")
        return 0
    else:
        print_error("Invalid choice")
        return 1
    
    print(f"\nSelected version: {selected_version}")
    
    # Check compatibility
    print("\nChecking compatibility...")
    compatibility = check_dependency_compatibility(selected_version)
    
    if compatibility["warnings"]:
        print_warning("Compatibility warnings detected:")
        for warning in compatibility["warnings"]:
            print(f"  - {warning}")
    
    if compatibility["recommendations"]:
        print_info("Recommendations:")
        for rec in compatibility["recommendations"]:
            print(f"  - {rec}")
    
    if not compatibility["compatible"]:
        response = input("\nThere are potential compatibility issues. Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print_info("Update cancelled")
            return 0
    
    # Update pyproject.toml
    print("\nUpdating pyproject.toml...")
    if update_pyproject_toml(selected_version, force=True):
        print_success("\nUpdate completed successfully!")
        print_info("\nNext steps:")
        print("1. Run: poetry lock --no-update")
        print("2. Run: poetry install")
        print("3. Verify with: poetry run python verify_numpy_abi.py")
        return 0
    else:
        print_error("\nUpdate failed!")
        return 1

if __name__ == "__main__":
    try:
        # Try to import toml
        import toml
    except ImportError:
        print_error("The 'toml' package is required but not installed.")
        print_info("Installing toml...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
        print_success("toml installed successfully. Please run the script again.")
        sys.exit(1)
    
    sys.exit(main())