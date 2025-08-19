#!/usr/bin/env python3
"""
Quick fix for remaining Pydantic import issues
"""

import os
import re
from pathlib import Path

def fix_pydantic_syntax_errors(root_dir):
    """Fix common Pydantic syntax errors"""
    
    print("🔍 Scanning for Pydantic syntax errors...")
    
    # Common patterns to fix
    fixes = [
        # Missing comma between imports
        (r'from pydantic import ([A-Za-z]+)\s+([A-Za-z]+)', r'from pydantic import \1, \2'),
        # Multiple spaces between imports
        (r'from pydantic import\s+(\w+)\s+(\w+)\s+(\w+)', r'from pydantic import \1, \2, \3'),
    ]
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv_tori_prod'}]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply fixes
                    for pattern, replacement in fixes:
                        content = re.sub(pattern, replacement, content)
                    
                    # Fix specific validator import issues
                    if 'from pydantic import' in content and 'validator' in content:
                        # Ensure validator is properly imported
                        content = re.sub(
                            r'from pydantic import (.*?)validator(?!,)',
                            r'from pydantic import \1validator,',
                            content
                        )
                        # Clean up trailing commas
                        content = re.sub(r',\s*\n', '\n', content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Fixed: {file_path}")
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    return fixed_count

def check_remaining_issues():
    """Check for any remaining import issues"""
    
    print("\n🔍 Checking for remaining issues...")
    
    problem_files = []
    
    # Check specific files that commonly have issues
    check_files = [
        "ingest_pdf/pipeline/config.py",
        "mcp_bridge_real_tori.py",
        "enhanced_launcher.py"
    ]
    
    for file_path in check_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common issues
                if re.search(r'from pydantic import \w+\s+\w+(?!,)', content):
                    problem_files.append(file_path)
                    print(f"⚠️  Potential issue in: {file_path}")
                    
            except Exception as e:
                print(f"❌ Could not check {file_path}: {e}")
    
    return problem_files

def main():
    print("🚀 Quick Pydantic Syntax Fix")
    print("=" * 50)
    
    # Get project root
    project_root = Path.cwd()
    
    # Fix syntax errors
    fixed = fix_pydantic_syntax_errors(project_root)
    print(f"\n✅ Fixed {fixed} files")
    
    # Check for remaining issues
    problems = check_remaining_issues()
    
    if problems:
        print(f"\n⚠️  {len(problems)} files may still have issues")
    else:
        print("\n✅ No remaining issues found")
    
    print("\n🎯 Try running enhanced_launcher.py again!")

if __name__ == "__main__":
    main()
