"""
MCP Stack Import Scanner

This script scans Python files for potential import issues by checking for
module usage before imports and missing type annotations.
"""

import os
import sys
import re
from pathlib import Path
import ast
from typing import List, Dict, Set, Tuple, Optional

def scan_file_for_import_issues(file_path: str) -> List[str]:
    """Scan a Python file for potential import issues."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for modules used before import
        try:
            tree = ast.parse(content)
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imported_names.add(name.name)
                    else:  # ImportFrom
                        for name in node.names:
                            if name.name == '*':
                                # Can't track * imports precisely
                                continue
                            imported_names.add(name.name)
                            if node.module:
                                imported_names.add(f"{node.module}.{name.name}")
                
                # Check for attribute access before import
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    module_name = node.value.id
                    if (module_name not in imported_names and 
                        module_name not in ['self', 'cls'] and
                        not any(node.lineno > imp_node.lineno for imp_node in ast.walk(tree) 
                                if isinstance(imp_node, (ast.Import, ast.ImportFrom))
                                and any(n.name == module_name for n in imp_node.names))):
                        issues.append(f"Potential module use before import: {module_name}.{node.attr} at line {node.lineno}")
        
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        # Check for typing usage without imports
        type_annotations = re.findall(r':\s*(List|Dict|Set|Tuple|Optional|Union|Callable|Any|TypeVar)\[', content)
        if type_annotations:
            type_imports = []
            if 'from typing import' in content:
                for match in re.finditer(r'from\s+typing\s+import\s+([^#\n]+)', content):
                    type_imports.extend([t.strip() for t in match.group(1).split(',')])
            
            for annotation in set(type_annotations):
                if annotation not in type_imports:
                    issues.append(f"Type annotation used without import: {annotation}")
        
        # Check for common stdlib usage without imports
        common_modules = {
            'os': [r'\bos\.\w+'],
            'sys': [r'\bsys\.\w+'],
            'json': [r'\bjson\.\w+'],
            'logging': [r'\blogging\.\w+'],
            'time': [r'\btime\.\w+', r'\btime\(\)'],
            'asyncio': [r'\basyncio\.\w+'],
            'pathlib': [r'\bPath\('],
        }
        
        for module, patterns in common_modules.items():
            if not any(re.search(rf'import\s+{module}\b', content) or 
                     re.search(rf'from\s+{module}\s+import', content)):
                for pattern in patterns:
                    if re.search(pattern, content):
                        issues.append(f"Potential use of {module} without import: {pattern}")
    
    except Exception as e:
        issues.append(f"Error scanning file: {e}")
    
    return issues

def scan_directory(directory: Path, extension: str = '.py') -> Dict[str, List[str]]:
    """Recursively scan directory for Python files with import issues."""
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                issues = scan_file_for_import_issues(file_path)
                if issues:
                    results[file_path] = issues
    
    return results

def main():
    """Main function."""
    # Get the root directory (adjust as needed)
    root_dir = Path(__file__).parent
    
    print(f"Scanning directory: {root_dir}")
    print("This may take a moment...")
    
    # Directories to scan
    dirs_to_scan = [
        root_dir / "ingest_pdf" / "pipeline",
        root_dir / "mcp_metacognitive" / "core",
    ]
    
    all_results = {}
    for directory in dirs_to_scan:
        if directory.exists():
            print(f"Scanning {directory}...")
            results = scan_directory(directory)
            all_results.update(results)
    
    # Print results
    if all_results:
        print("\nPotential import issues found:")
        for file_path, issues in all_results.items():
            print(f"\n{file_path}:")
            for issue in issues:
                print(f"  - {issue}")
        
        print(f"\nTotal files with issues: {len(all_results)}")
    else:
        print("\nNo import issues found.")

if __name__ == "__main__":
    main()
