#!/usr/bin/env python3
"""
Fix ConceptMesh Initialization Issues
This script finds and fixes all ConceptMesh initialization problems
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file before modifying"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up: {filepath} -> {backup_path}")
    return backup_path

def fix_conceptmesh_initialization(filepath):
    """Fix ConceptMesh initialization in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
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
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes_made
    
    return []

def find_python_files(directory):
    """Find all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in {'.venv', '__pycache__', 'venv', '.git', 'node_modules'}]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    """Main execution function"""
    print("🔍 ConceptMesh Initialization Fixer")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    
    # Find all Python files
    print("\n🔎 Searching for Python files...")
    python_files = find_python_files(project_root)
    print(f"📊 Found {len(python_files)} Python files")
    
    # Check each file for ConceptMesh issues
    files_with_issues = []
    total_changes = []
    
    print("\n🔧 Checking files for ConceptMesh initialization issues...")
    for filepath in python_files:
        try:
            # Read file and check for ConceptMesh
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'ConceptMesh' in content and 'class ConceptMesh' not in content:
                # This file uses ConceptMesh but doesn't define it
                changes = fix_conceptmesh_initialization(filepath)
                if changes:
                    files_with_issues.append(filepath)
                    total_changes.extend([(filepath, change) for change in changes])
                    print(f"✅ Fixed: {filepath}")
                    for change in changes:
                        print(f"   - {change}")
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total files checked: {len(python_files)}")
    print(f"Files with issues fixed: {len(files_with_issues)}")
    print(f"Total changes made: {len(total_changes)}")
    
    if files_with_issues:
        print("\n📝 Files modified:")
        for file in files_with_issues:
            print(f"   - {file}")
    
    print("\n✅ ConceptMesh initialization fix complete!")
    
    # Create a summary file
    summary_path = project_root / "CONCEPTMESH_FIX_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# ConceptMesh Initialization Fix Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Statistics\n")
        f.write(f"- Total files checked: {len(python_files)}\n")
        f.write(f"- Files modified: {len(files_with_issues)}\n")
        f.write(f"- Total changes: {len(total_changes)}\n\n")
        
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
