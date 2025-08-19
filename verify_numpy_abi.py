#!/usr/bin/env python3
"""
NumPy ABI Compatibility Verification Script
Tests for binary compatibility issues between NumPy and compiled packages
Author: Enhanced Assistant
Date: 2025-08-06
"""

import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{title:^60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}[{title}]{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 40}{Colors.RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        return True, version_str
    else:
        return False, version_str

def check_package_import(package_name: str) -> Dict[str, Any]:
    """Test if a package can be imported and get its version"""
    result = {
        'name': package_name,
        'imported': False,
        'version': None,
        'error': None,
        'error_type': None
    }
    
    try:
        module = __import__(package_name)
        result['imported'] = True
        
        # Try to get version
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                result['version'] = str(getattr(module, attr))
                break
        
        if result['version'] is None:
            result['version'] = 'unknown'
            
    except ImportError as e:
        result['error'] = str(e)
        result['error_type'] = 'ImportError'
    except ValueError as e:
        # This is often the ABI compatibility error
        result['error'] = str(e)
        result['error_type'] = 'ValueError'
        if 'dtype size changed' in str(e) or 'ndarray size changed' in str(e):
            result['error_type'] = 'ABI_INCOMPATIBILITY'
    except Exception as e:
        result['error'] = str(e)
        result['error_type'] = type(e).__name__
    
    return result

def test_numpy_abi_compatibility() -> Dict[str, Any]:
    """Comprehensive test for NumPy ABI compatibility"""
    results = {
        'numpy_compatible': False,
        'dtype_sizes': {},
        'c_api_version': None,
        'abi_tests': []
    }
    
    try:
        import numpy as np
        
        # Get NumPy C API version
        results['c_api_version'] = np.version.version
        
        # Check dtype sizes
        dtype_tests = [
            ('float32', np.float32),
            ('float64', np.float64),
            ('int32', np.int32),
            ('int64', np.int64),
            ('complex64', np.complex64),
            ('complex128', np.complex128)
        ]
        
        for name, dtype in dtype_tests:
            try:
                arr = np.array([1], dtype=dtype)
                results['dtype_sizes'][name] = arr.dtype.itemsize
            except Exception as e:
                results['dtype_sizes'][name] = f"Error: {e}"
        
        # Test various NumPy operations
        abi_tests = [
            ('array_creation', lambda: np.array([1, 2, 3])),
            ('array_math', lambda: np.sum(np.array([1, 2, 3]))),
            ('array_reshape', lambda: np.array([1, 2, 3, 4]).reshape(2, 2)),
            ('array_indexing', lambda: np.array([1, 2, 3])[1]),
            ('random_generation', lambda: np.random.rand(3)),
        ]
        
        for test_name, test_func in abi_tests:
            try:
                test_func()
                results['abi_tests'].append({'test': test_name, 'passed': True})
            except Exception as e:
                results['abi_tests'].append({
                    'test': test_name, 
                    'passed': False, 
                    'error': str(e)
                })
        
        results['numpy_compatible'] = all(
            test['passed'] for test in results['abi_tests']
        )
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def test_spacy_compatibility() -> Dict[str, Any]:
    """Test spaCy specific compatibility"""
    results = {
        'spacy_functional': False,
        'nlp_creation': False,
        'tokenization': False,
        'error': None
    }
    
    try:
        import spacy
        
        # Try to create a blank pipeline
        try:
            nlp = spacy.blank('en')
            results['nlp_creation'] = True
            
            # Try tokenization
            doc = nlp("This is a test sentence.")
            tokens = [token.text for token in doc]
            if len(tokens) > 0:
                results['tokenization'] = True
                
            results['spacy_functional'] = True
            
        except Exception as e:
            results['error'] = f"spaCy operation failed: {e}"
            
    except ImportError as e:
        results['error'] = f"spaCy import failed: {e}"
    except ValueError as e:
        if 'dtype size changed' in str(e):
            results['error'] = f"ABI incompatibility: {e}"
        else:
            results['error'] = f"ValueError: {e}"
    except Exception as e:
        results['error'] = f"Unexpected error: {e}"
    
    return results

def main():
    """Main verification routine"""
    print_header("NUMPY ABI COMPATIBILITY VERIFICATION")
    
    print_info(f"Verification started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Working directory: {Path.cwd()}")
    
    # Check Python version
    print_section("Python Environment Check")
    py_compatible, py_version = check_python_version()
    if py_compatible:
        print_success(f"Python version: {py_version}")
    else:
        print_error(f"Python version {py_version} may have compatibility issues")
    
    print_info(f"Python executable: {sys.executable}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Running in virtual environment")
    else:
        print_warning("Not running in virtual environment")
    
    # Package import tests
    print_section("Package Import Tests")
    
    critical_packages = [
        'numpy', 'scipy', 'pandas', 'matplotlib',
        'thinc', 'spacy', 'numba', 'torch'
    ]
    
    import_results = []
    abi_issues = []
    
    for package in critical_packages:
        result = check_package_import(package)
        import_results.append(result)
        
        if result['imported']:
            print_success(f"{result['name']} {result['version']} imported successfully")
        else:
            if result['error_type'] == 'ABI_INCOMPATIBILITY':
                print_error(f"{result['name']} - ABI INCOMPATIBILITY: {result['error']}")
                abi_issues.append(result['name'])
            elif result['error_type'] == 'ImportError':
                print_warning(f"{result['name']} - Not installed")
            else:
                print_error(f"{result['name']} - {result['error_type']}: {result['error']}")
    
    # NumPy ABI tests
    print_section("NumPy ABI Detailed Check")
    numpy_results = test_numpy_abi_compatibility()
    
    if numpy_results.get('c_api_version'):
        print_success(f"NumPy version: {numpy_results['c_api_version']}")
        
        # Check if NumPy 2.x
        if numpy_results['c_api_version'].startswith('2.'):
            print_warning("NumPy 2.x detected - may have compatibility issues with some packages")
        
        print_info("Dtype sizes:")
        for dtype, size in numpy_results.get('dtype_sizes', {}).items():
            print(f"  - {dtype}: {size} bytes")
        
        if numpy_results['numpy_compatible']:
            print_success("NumPy operations test passed")
        else:
            print_error("NumPy operations test failed")
            for test in numpy_results.get('abi_tests', []):
                if not test['passed']:
                    print_error(f"  - {test['test']}: {test.get('error', 'Failed')}")
    else:
        print_error("NumPy not available or not functional")
    
    # spaCy specific tests
    print_section("spaCy Functionality Check")
    spacy_results = test_spacy_compatibility()
    
    if spacy_results['spacy_functional']:
        print_success("spaCy is fully functional")
        if spacy_results['nlp_creation']:
            print_success("  - Pipeline creation: OK")
        if spacy_results['tokenization']:
            print_success("  - Tokenization: OK")
    else:
        print_error(f"spaCy not functional: {spacy_results.get('error', 'Unknown error')}")
    
    # Generate report
    print_section("Generating Report")
    
    report_dir = Path("verification_reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"numpy_abi_check_{timestamp}.json"
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'python_version': py_version,
        'python_compatible': py_compatible,
        'packages': import_results,
        'abi_issues': abi_issues,
        'numpy_tests': numpy_results,
        'spacy_tests': spacy_results,
        'summary': {
            'total_packages': len(import_results),
            'successful_imports': sum(1 for r in import_results if r['imported']),
            'abi_incompatibilities': len(abi_issues),
            'numpy_functional': numpy_results.get('numpy_compatible', False),
            'spacy_functional': spacy_results.get('spacy_functional', False)
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print_success(f"Report saved to: {report_file}")
    
    # Final summary
    print_header("VERIFICATION COMPLETE")
    
    if abi_issues:
        print_error("Critical issues detected!")
        for package in abi_issues:
            print_error(f"  - {package}: ABI incompatibility")
        print_warning("\nPlease run the fix_numpy_abi.ps1 script to resolve these issues.")
    elif not spacy_results['spacy_functional']:
        print_warning("spaCy is not fully functional")
        print_warning("\nConsider running the fix_numpy_abi.ps1 script.")
    else:
        print_success("No critical issues detected!")
        print_success("All packages appear to be compatible.")
    
    return 0 if not abi_issues else 1

if __name__ == "__main__":
    sys.exit(main())