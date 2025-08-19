#!/usr/bin/env python3
"""
Fix Soliton Init Typo
====================

This script searches for and fixes the user_id typo in the soliton init handler.
"""

import os
import re
from pathlib import Path

def search_and_fix_typo(directory):
    """Search for user_id typo in Python files"""
    
    found_issues = []
    fixed_issues = []
    
    # Patterns to search for
    patterns = [
        (r'user_id', 'user_id'),
        (r'\.user_id', '.user_id'),
        (r'request\.user_id', 'request.user_id'),
        (r'init_request\.user_id', 'init_request.user_id')
    ]
    
    # Search in Python files
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv_tori_prod', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        original_content = content
                    
                    # Check for any of the patterns
                    for pattern, replacement in patterns:
                        if re.search(pattern, content):
                            found_issues.append((filepath, pattern))
                            # Fix the issue
                            content = re.sub(pattern, replacement, content)
                    
                    # Write back if changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_issues.append(filepath)
                        print(f"✅ Fixed typo in: {filepath}")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {filepath}: {e}")
    
    return found_issues, fixed_issues

def check_specific_files():
    """Check specific files that might have the issue"""
    specific_files = [
        "api/routes/soliton.py",
        "api/routes/soliton_production.py",
        "api/routes/soliton_router.py",
        "alan_backend/routes/soliton.py",
        "api/routes/fallback_soliton.py"
    ]
    
    for file_path in specific_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"\n📄 Checking {file_path}...")
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for the init endpoint
                if "/init" in content:
                    print("  Found /init endpoint")
                    
                    # Check for user_id usage
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'user_id' in line and 'request' in line:
                            print(f"  Line {i+1}: {line.strip()}")
                            if 'user_id' in line:
                                print(f"  ⚠️ FOUND TYPO at line {i+1}!")
                                
            except Exception as e:
                print(f"  Error reading file: {e}")

def main():
    """Main function"""
    print("🔍 Searching for user_id typo in Soliton routes...")
    
    # First check specific files
    check_specific_files()
    
    # Then do a full search
    print("\n🔍 Performing full search...")
    found, fixed = search_and_fix_typo(".")
    
    if found:
        print(f"\n📊 Summary:")
        print(f"  Found {len(found)} instances of the typo")
        print(f"  Fixed {len(fixed)} files")
        
        print(f"\n📄 Files with issues:")
        for filepath, pattern in found:
            print(f"  - {filepath} (pattern: {pattern})")
    else:
        print("\n✅ No instances of user_id typo found!")
        print("\n🤔 The error might be coming from:")
        print("  1. A request body with incorrect field name")
        print("  2. Frontend code sending wrong field name")
        print("  3. A middleware or request parser issue")

if __name__ == "__main__":
    main()
