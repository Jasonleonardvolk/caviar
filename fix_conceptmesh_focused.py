#!/usr/bin/env python3
"""
Fix ConceptMesh Initialization Issues - Focused Version
This script finds and fixes ConceptMesh initialization problems in project files only
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

# Directories to exclude from search
EXCLUDE_DIRS = {
    '.venv', '__pycache__', 'venv', '.git', 'node_modules',
    'venv_tori_prod', '.yarn', 'build', 'dist', 'eggs',
    '*.egg-info', 'htmlcov', '.tox', '.pytest_cache',
    '.mypy_cache', 'site-packages', 'Lib', 'Scripts',
    'bin', 'lib', 'lib64', 'include', 'share'
}

# File patterns to exclude
EXCLUDE_FILES = {
    'test_golden_e2e.py',  # Already fixed
}

def backup_file(filepath):
    """Create a backup of the file before modifying"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up: {filepath} -> {backup_path}")
    return backup_path

def fix_conceptmesh_initialization(filepath):
    """Fix ConceptMesh initialization in a file"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    content = None
    used_encoding = None
    
    # Try different encodings
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        return None, "Failed to decode file with any encoding"
    
    original_content = content
    changes_made = []
    
    # Skip if this is a class definition file
    if 'class ConceptMesh' in content:
        return [], None
    
    # Pattern 1: ConceptMesh with url parameter
    pattern1 = r'ConceptMesh\s*\(\s*url\s*=\s*[^)]+\)'
    if re.search(pattern1, content):
        content = re.sub(pattern1, 'ConceptMesh()', content)
        changes_made.append("Fixed ConceptMesh(url=...) -> ConceptMesh()")
    
    # Pattern 2: ConceptMesh with config parameter
    pattern2 = r'ConceptMesh\s*\(\s*config\s*=\s*[^)]+\)'
    if re.search(pattern2, content):
        content = re.sub(pattern2, 'ConceptMesh()', content)
        changes_made.append("Fixed ConceptMesh(config=...) -> ConceptMesh()")
    
    # Pattern 3: Any ConceptMesh with keyword arguments
    pattern3 = r'ConceptMesh\s*\(\s*\w+\s*=\s*[^)]+\)'
    if re.search(pattern3, content):
        content = re.sub(pattern3, 'ConceptMesh()', content)
        changes_made.append("Fixed ConceptMesh with keyword args -> ConceptMesh()")
    
    if content != original_content:
        backup_file(filepath)
        with open(filepath, 'w', encoding=used_encoding) as f:
            f.write(content)
        return changes_made, None
    
    return [], None

def is_project_file(filepath, project_root):
    """Check if file is part of the project (not in dependencies)"""
    try:
        rel_path = Path(filepath).relative_to(project_root)
        parts = rel_path.parts
        
        # Check if any part of the path is in exclude dirs
        for part in parts:
            if part in EXCLUDE_DIRS:
                return False
        
        # Check specific file exclusions
        if rel_path.name in EXCLUDE_FILES:
            return False
            
        return True
    except ValueError:
        return False

def find_project_python_files(directory):
    """Find Python files in the project (excluding dependencies)"""
    project_root = Path(directory)
    python_files = []
    
    # Key project directories to check
    project_dirs = [
        'mcp_metacognitive',
        'python',
        'core',
        'api',
        'routes',
        'tests',
        'scripts',
        'tools',
        'ingest_pdf',
        'services',
        'utils'
    ]
    
    # Search in root directory files
    for file in project_root.glob('*.py'):
        if file.name not in EXCLUDE_FILES:
            python_files.append(str(file))
    
    # Search in project directories
    for proj_dir in project_dirs:
        dir_path = project_root / proj_dir
        if dir_path.exists():
            for file in dir_path.rglob('*.py'):
                if is_project_file(file, project_root):
                    python_files.append(str(file))
    
    return python_files

def main():
    """Main execution function"""
    print("🔍 ConceptMesh Initialization Fixer - Focused Version")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    
    # Find project Python files
    print("\n🔎 Searching for project Python files...")
    python_files = find_project_python_files(project_root)
    print(f"📊 Found {len(python_files)} project Python files")
    
    # Check each file for ConceptMesh issues
    files_with_issues = []
    total_changes = []
    errors = []
    
    print("\n🔧 Checking files for ConceptMesh initialization issues...")
    for filepath in python_files:
        try:
            # Read file and check for ConceptMesh
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'ConceptMesh' in content and 'class ConceptMesh' not in content:
                # This file uses ConceptMesh but doesn't define it
                changes, error = fix_conceptmesh_initialization(filepath)
                if error:
                    errors.append((filepath, error))
                elif changes:
                    files_with_issues.append(filepath)
                    total_changes.extend([(filepath, change) for change in changes])
                    print(f"✅ Fixed: {filepath}")
                    for change in changes:
                        print(f"   - {change}")
        except Exception as e:
            errors.append((filepath, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total files checked: {len(python_files)}")
    print(f"Files with issues fixed: {len(files_with_issues)}")
    print(f"Total changes made: {len(total_changes)}")
    print(f"Files with errors: {len(errors)}")
    
    if files_with_issues:
        print("\n📝 Files modified:")
        for file in files_with_issues:
            print(f"   - {file}")
    
    if errors:
        print("\n⚠️ Files with errors (skipped):")
        for file, error in errors[:5]:  # Show only first 5 errors
            print(f"   - {file}: {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    
    print("\n✅ ConceptMesh initialization fix complete!")
    
    # Create a summary file
    summary_path = project_root / "CONCEPTMESH_FIX_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# ConceptMesh Initialization Fix Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Statistics\n")
        f.write(f"- Total project files checked: {len(python_files)}\n")
        f.write(f"- Files modified: {len(files_with_issues)}\n")
        f.write(f"- Total changes: {len(total_changes)}\n")
        f.write(f"- Files with errors: {len(errors)}\n\n")
        
        if total_changes:
            f.write("## Changes Made\n\n")
            for filepath, change in total_changes:
                f.write(f"### {filepath}\n")
                f.write(f"- {change}\n\n")
        
        f.write("\n## Next Steps\n")
        f.write("1. Run `poetry run python enhanced_launcher.py` to test the fixes\n")
        f.write("2. Check the backup files if you need to restore any changes\n")
        f.write("3. Remove backup files once you've verified everything works\n")
    
    print(f"\n📄 Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
