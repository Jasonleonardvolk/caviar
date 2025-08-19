#!/usr/bin/env python3
"""
Update all imports from psi_archive to psi_archive_extended
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace various import patterns
        patterns = [
            # from x import psi_archive
            (r'from\s+(\S+)\s+import\s+psi_archive\b', r'from \1 import psi_archive_extended as psi_archive'),
            # from x.psi_archive import Y
            (r'from\s+(\S+)\.psi_archive\s+import', r'from \1.psi_archive_extended import'),
            # import x.psi_archive
            (r'import\s+(\S+)\.psi_archive\b', r'import \1.psi_archive_extended'),
            # Direct references to psi_archive module
            (r'(\s+)psi_archive\.', r'\1psi_archive_extended.'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Special case for PSI_ARCHIVER import
        content = re.sub(
            r'from\s+core\.psi_archive\s+import\s+PSI_ARCHIVER',
            'from core.psi_archive_extended import PSI_ARCHIVER',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in the project"""
    root_dir = Path(__file__).parent.parent
    
    # Directories to search
    search_dirs = ['python', 'api', 'tools', 'ingest_pdf', 'integrations']
    
    updated_files = []
    
    for dir_name in search_dirs:
        dir_path = root_dir / dir_name
        if not dir_path.exists():
            continue
            
        for py_file in dir_path.rglob('*.py'):
            # Skip the archive files themselves
            if 'psi_archive' in py_file.name:
                continue
                
            if update_imports_in_file(py_file):
                updated_files.append(py_file)
                print(f"✅ Updated: {py_file.relative_to(root_dir)}")
    
    print(f"\n📊 Summary: Updated {len(updated_files)} files")
    
    # Also update any config files
    config_files = [
        root_dir / 'config' / 'ingestion_config.yaml',
        root_dir / 'config' / 'pipeline_config.yaml'
    ]
    
    for config_file in config_files:
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                if 'psi_archive' in content and 'psi_archive_extended' not in content:
                    content = content.replace('psi_archive', 'psi_archive_extended')
                    with open(config_file, 'w') as f:
                        f.write(content)
                    print(f"✅ Updated config: {config_file.name}")
            except Exception as e:
                print(f"⚠️  Error updating {config_file}: {e}")

if __name__ == "__main__":
    main()
