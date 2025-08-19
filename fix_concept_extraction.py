#!/usr/bin/env python
"""
Comprehensive fix script for concept_extraction import issues.
This script:
1. Cleans __pycache__ directories and .pyc files
2. Searches for duplicate concept_extraction implementations
3. Checks PYTHONPATH for module conflicts
4. Tests actual module imports to verify which one is being used
5. Provides a detailed report of findings
"""

import os
import sys
import shutil
import importlib
import logging
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("concept_extraction_fix")

def clean_pycache(base_dir: str) -> Tuple[int, int]:
    """Remove __pycache__ directories and .pyc files
    
    Returns:
        Tuple containing count of removed pycache dirs and pyc files
    """
    count_pycache = 0
    count_pyc = 0
    
    logger.info(f"Cleaning __pycache__ directories and .pyc files in {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            logger.info(f"Removing {pycache_path}")
            try:
                shutil.rmtree(pycache_path)
                count_pycache += 1
            except Exception as e:
                logger.error(f"Failed to remove {pycache_path}: {e}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith(".pyc"):
                pyc_path = os.path.join(root, file)
                logger.info(f"Removing {pyc_path}")
                try:
                    os.remove(pyc_path)
                    count_pyc += 1
                except Exception as e:
                    logger.error(f"Failed to remove {pyc_path}: {e}")
    
    logger.info(f"Removed {count_pycache} __pycache__ directories and {count_pyc} .pyc files")
    return count_pycache, count_pyc

def find_duplicate_implementations(base_dir: str) -> List[str]:
    """Search for duplicate implementations of concept_extraction
    
    Returns:
        List of file paths containing extract_concepts_from_text function definitions
    """
    results = []
    logger.info(f"Searching for extract_concepts_from_text implementations in {base_dir}...")
    
    try:
        # Use grep to search for function definitions
        grep_cmd = ["grep", "-r", "def extract_concepts_from_text", base_dir, "--include=*.py"]
        
        # On Windows, use findstr instead
        if os.name == 'nt':
            grep_cmd = ["findstr", "/s", "/i", "/m", "def extract_concepts_from_text", f"{base_dir}\\*.py"]
        
        process = subprocess.run(grep_cmd, capture_output=True, text=True)
        
        if process.returncode == 0 or process.returncode == 1:  # grep returns 1 if no matches
            for line in process.stdout.splitlines():
                if line.strip():
                    # Extract file path from the output
                    if os.name == 'nt':
                        # Windows findstr format is typically just the filename
                        results.append(line.split(':', 1)[0])
                    else:
                        # Unix grep format is "file:match"
                        results.append(line.split(':', 1)[0])
        
        if not results:
            logger.info("No duplicate function definitions found.")
        else:
            for path in results:
                logger.info(f"Found implementation in: {path}")
    
    except Exception as e:
        logger.error(f"Error searching for duplicate implementations: {e}")
    
    return results

def find_import_references(base_dir: str) -> List[str]:
    """Search for import references to concept_extraction
    
    Returns:
        List of file paths containing imports of extract_concepts_from_text
    """
    results = []
    logger.info(f"Searching for import references to extract_concepts_from_text in {base_dir}...")
    
    try:
        # Use grep to search for imports
        grep_cmd = ["grep", "-r", "import.*extract_concepts_from_text", base_dir, "--include=*.py"]
        
        # On Windows, use findstr instead
        if os.name == 'nt':
            grep_cmd = ["findstr", "/s", "/i", "/m", "import.*extract_concepts_from_text", f"{base_dir}\\*.py"]
        
        process = subprocess.run(grep_cmd, capture_output=True, text=True)
        
        if process.returncode == 0 or process.returncode == 1:  # grep returns 1 if no matches
            for line in process.stdout.splitlines():
                if line.strip():
                    # Extract file path from the output
                    if os.name == 'nt':
                        # Windows findstr format
                        file_path = line.split(':', 1)[0]
                    else:
                        # Unix grep format
                        file_path = line.split(':', 1)[0]
                    
                    if file_path not in results:
                        results.append(file_path)
        
        if not results:
            logger.info("No import references found.")
        else:
            for path in results:
                logger.info(f"Found import reference in: {path}")
    
    except Exception as e:
        logger.error(f"Error searching for import references: {e}")
    
    return results

def check_pythonpath() -> List[str]:
    """Check PYTHONPATH for module conflicts
    
    Returns:
        List of paths in sys.path
    """
    logger.info("Checking Python path...")
    
    paths = []
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")
        paths.append(path)
    
    # Also check environment variable
    env_path = os.environ.get('PYTHONPATH', '')
    if env_path:
        logger.info(f"PYTHONPATH environment variable: {env_path}")
        for p in env_path.split(os.pathsep):
            if p not in paths:
                paths.append(p)
                logger.info(f"  Additional path from env: {p}")
    
    return paths

def find_module_in_pythonpath(module_name: str, paths: List[str]) -> List[str]:
    """Find module in Python path
    
    Args:
        module_name: Name of the module to find (e.g., 'concept_extraction')
        paths: List of paths to search in
        
    Returns:
        List of found module paths
    """
    logger.info(f"Searching for {module_name} module in Python path...")
    
    found_modules = []
    
    # Convert module name to possible file paths
    possible_paths = [
        f"{module_name}.py",
        os.path.join(module_name, "__init__.py"),
        os.path.join("ingest_pdf", "extraction", f"{module_name}.py")
    ]
    
    for path in paths:
        if not os.path.exists(path):
            continue
            
        for possible_path in possible_paths:
            full_path = os.path.join(path, possible_path)
            if os.path.exists(full_path):
                logger.info(f"Found module at: {full_path}")
                found_modules.append(full_path)
    
    if not found_modules:
        logger.info(f"No {module_name} modules found in Python path.")
    
    return found_modules

def test_module_import(module_name: str) -> Optional[str]:
    """Test importing a module
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Path to the imported module file, or None if import failed
    """
    logger.info(f"Testing import of {module_name}...")
    
    try:
        # Try to import the module
        module = importlib.import_module(module_name)
        
        # Check module file path
        if hasattr(module, "__file__"):
            module_path = module.__file__
            logger.info(f"Successfully imported {module_name} from {module_path}")
            
            # Check for key attributes
            if hasattr(module, "extract_concepts_from_text"):
                logger.info(f"Module has extract_concepts_from_text function")
            else:
                logger.warning(f"Module does not have extract_concepts_from_text function")
            
            if hasattr(module, "Concept"):
                logger.info(f"Module has Concept class")
            else:
                logger.warning(f"Module does not have Concept class")
            
            return module_path
        else:
            logger.warning(f"Imported {module_name} but it has no __file__ attribute")
            return None
    
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error testing import of {module_name}: {e}")
        return None

def generate_report(
    cleaned_pycache: Tuple[int, int],
    implementations: List[str],
    import_refs: List[str],
    python_paths: List[str],
    found_modules: List[str],
    import_test_results: Dict[str, Optional[str]]
) -> str:
    """Generate a report of findings
    
    Returns:
        Report as a string
    """
    report = []
    report.append("=" * 80)
    report.append("CONCEPT EXTRACTION MODULE FIX REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Cleaning results
    report.append("1. CACHE CLEANING RESULTS")
    report.append("-" * 40)
    report.append(f"Removed {cleaned_pycache[0]} __pycache__ directories")
    report.append(f"Removed {cleaned_pycache[1]} .pyc files")
    report.append("")
    
    # Implementation findings
    report.append("2. CONCEPT EXTRACTION IMPLEMENTATIONS")
    report.append("-" * 40)
    if implementations:
        report.append(f"Found {len(implementations)} implementation(s):")
        for impl in implementations:
            report.append(f"  - {impl}")
    else:
        report.append("No implementations found (this is unexpected)")
    report.append("")
    
    # Import references
    report.append("3. IMPORT REFERENCES")
    report.append("-" * 40)
    if import_refs:
        report.append(f"Found {len(import_refs)} file(s) importing extract_concepts_from_text:")
        for ref in import_refs:
            report.append(f"  - {ref}")
    else:
        report.append("No import references found")
    report.append("")
    
    # Python path
    report.append("4. PYTHON PATH")
    report.append("-" * 40)
    report.append(f"Found {len(python_paths)} directories in Python path")
    report.append("")
    
    # Found modules
    report.append("5. MODULES FOUND IN PYTHON PATH")
    report.append("-" * 40)
    if found_modules:
        report.append(f"Found {len(found_modules)} module file(s):")
        for module in found_modules:
            report.append(f"  - {module}")
    else:
        report.append("No modules found in Python path")
    report.append("")
    
    # Import test results
    report.append("6. IMPORT TEST RESULTS")
    report.append("-" * 40)
    for module_name, result in import_test_results.items():
        if result:
            report.append(f"{module_name}: Successfully imported from {result}")
        else:
            report.append(f"{module_name}: Import failed")
    report.append("")
    
    # Diagnosis and recommendations
    report.append("7. DIAGNOSIS AND RECOMMENDATIONS")
    report.append("-" * 40)
    
    # Case 1: Multiple implementations found
    if len(implementations) > 1:
        report.append("⚠️ ISSUE: Multiple implementations of extract_concepts_from_text found")
        report.append("RECOMMENDATION: Keep only the canonical dataclass-based implementation")
        report.append("and remove or rename other versions to avoid conflicts.")
    
    # Case 2: Module import failures
    if any(result is None for result in import_test_results.values()):
        report.append("⚠️ ISSUE: Some module imports failed")
        report.append("RECOMMENDATION: Check your project structure and ensure the module")
        report.append("is in the correct location with proper __init__.py files.")
    
    # Case 3: Module found in unexpected location
    canonical_path = os.path.abspath("ingest_pdf/extraction/concept_extraction.py")
    non_canonical_modules = [m for m in found_modules if canonical_path not in m]
    
    if non_canonical_modules:
        report.append("⚠️ ISSUE: Found concept_extraction module in non-canonical location(s):")
        for module in non_canonical_modules:
            report.append(f"  - {module}")
        report.append("RECOMMENDATION: Remove these copies or ensure they're not in Python path.")
    
    # Case 4: All tests pass
    if (
        len(implementations) == 1 and
        all(result is not None for result in import_test_results.values()) and
        not non_canonical_modules
    ):
        report.append("✅ Good news! No obvious issues found after cleanup.")
        report.append("The module should now import correctly.")
    
    # General recommendations
    report.append("")
    report.append("GENERAL RECOMMENDATIONS:")
    report.append("1. Restart your application/environment to apply changes")
    report.append("2. Use explicit imports: from ingest_pdf.extraction.concept_extraction import extract_concepts_from_text")
    report.append("3. If using notebooks or scripts, verify they have the correct import paths")
    report.append("4. Consider adding __init__.py files if missing in package directories")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function"""
    # Get the base directory (assuming script is run from project root)
    base_dir = os.getcwd()
    
    logger.info(f"Starting concept_extraction fix script in {base_dir}")
    
    # 1. Clean __pycache__ directories and .pyc files
    cleaned_pycache = clean_pycache(base_dir)
    
    # 2. Search for duplicate implementations
    implementations = find_duplicate_implementations(base_dir)
    
    # 3. Search for import references
    import_refs = find_import_references(base_dir)
    
    # 4. Check Python path
    python_paths = check_pythonpath()
    
    # 5. Find modules in Python path
    found_modules = find_module_in_pythonpath("concept_extraction", python_paths)
    
    # 6. Test imports
    import_test_results = {}
    
    # Try different import paths
    modules_to_test = [
        "concept_extraction",
        "ingest_pdf.extraction.concept_extraction"
    ]
    
    for module_name in modules_to_test:
        import_test_results[module_name] = test_module_import(module_name)
    
    # 7. Generate and print report
    report = generate_report(
        cleaned_pycache,
        implementations,
        import_refs,
        python_paths,
        found_modules,
        import_test_results
    )
    
    print("\n\n" + report)
    
    # Save report to file
    report_path = os.path.join(base_dir, "concept_extraction_fix_report.txt")
    try:
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    logger.info("Fix script completed")

if __name__ == "__main__":
    main()
