#!/usr/bin/env python3
"""
Fix Pydantic v2 BaseSettings import errors in TORI/KHA project
"""

import os
import re
from pathlib import Path

def fix_pydantic_imports(file_path):
    """Fix Pydantic imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file imports BaseSettings from pydantic
        if 'from pydantic import' in content and 'BaseSettings' in content:
            # Replace the import
            original_content = content
            
            # Pattern to match various import styles
            patterns = [
                # from pydantic import
from pydantic_settings import BaseSettings
                (r'from pydantic import ([^;\n]*?), ([^;\n]*?)(?=\n|$)'
                 lambda m: f'from pydantic import {m.group(1).replace("" "").strip()}{m.group(2).replace("" "").strip()}'.strip().rstrip('') + '\nfrom pydantic_settings import')
                
                # from pydantic import other_stuff
                (r'from pydantic import (.*?)(.*?)(?=\n|$)'
                 lambda m: handle_mixed_import(m)),
            ]
            
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            # Remove duplicate imports and clean up
            lines = content.split('\n')
            cleaned_lines = []
            seen_imports = set()
            
            for line in lines:
                if line.startswith('from pydantic import'):
                    # Remove empty imports
                    if line.strip() == 'from pydantic import':
                        continue
                    # Remove trailing commas and clean up
                    line = re.sub(r',\s*$', '', line)
                    line = re.sub(r'import\s+,', 'import', line)
                    
                # Track imports to avoid duplicates
                import_sig = line.strip()
                if import_sig.startswith('from pydantic'):
                    if import_sig in seen_imports:
                        continue
                    seen_imports.add(import_sig)
                
                cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {file_path}")
                return True
        
        # Also check for standalone BaseSettings usage
        elif re.search(r'\bBaseSettings\b', content) and 'pydantic' in content:
            # File uses BaseSettings but might import it differently
            print(f"Check needed: {file_path} - uses BaseSettings")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def handle_mixed_import(match):
    """Handle imports that have BaseSettings mixed with other imports"""
    before = match.group(1).strip()
    after = match.group(2).strip()
    
    # Collect all other imports
    other_imports = []
    if before and before != ',':
        other_imports.extend([i.strip() for i in before.rstrip(',').split(',') if i.strip()])
    if after and after != ',':
        other_imports.extend([i.strip() for i in after.lstrip(',').split(',') if i.strip()])
    
    result = []
    if other_imports:
        result.append(f"from pydantic import {', '.join(other_imports)}")
    result.append("from pydantic_settings import BaseSettings")
    
    return '\n'.join(result)

def find_python_files(root_dir):
    """Find all Python files in the project"""
    python_files = []
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv_tori_prod', 'venv', '.pytest_cache'}
    
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from the search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    # Get the project root
    project_root = Path("C:\\Users\\jason\\Desktop\\tori\\kha")
    
    print(f"Scanning for Python files in: {project_root}")
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_pydantic_imports(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    # Also update requirements
    req_files = [
        project_root / "requirements.txt",
        project_root / "requirements_production.txt",
        project_root / "requirements_nodb.txt"
    ]
    
    for req_file in req_files:
        if req_file.exists():
            update_requirements(req_file)

def update_requirements(req_file):
    """Add pydantic-settings to requirements file if needed"""
    try:
        with open(req_file, 'r') as f:
            content = f.read()
        
        if 'pydantic' in content and 'pydantic-settings' not in content:
            lines = content.strip().split('\n')
            
            # Find where to insert pydantic-settings
            for i, line in enumerate(lines):
                if line.strip().startswith('pydantic'):
                    # Insert pydantic-settings after pydantic
                    lines.insert(i + 1, 'pydantic-settings>=2.0.0')
                    break
            else:
                # If pydantic not found explicitly, add at the end
                if 'fastapi' in content:  # FastAPI depends on pydantic
                    lines.append('pydantic>=2.0.0')
                    lines.append('pydantic-settings>=2.0.0')
            
            with open(req_file, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            
            print(f"Updated requirements file: {req_file}")
    except Exception as e:
        print(f"Error updating {req_file}: {e}")

if __name__ == "__main__":
    main()
