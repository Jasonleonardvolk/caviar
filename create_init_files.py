#!/usr/bin/env python3
"""
Create missing __init__.py files for Python modules
"""

from pathlib import Path

def create_init_files():
    """Create __init__.py files in Python directories"""
    
    # Define directories that need __init__.py
    directories = [
        "python",
        "python/core",
        "python/stability",
    ]
    
    created = 0
    for dir_path in directories:
        directory = Path(dir_path)
        if directory.exists() and directory.is_dir():
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
                print(f"✅ Created {init_file}")
                created += 1
            else:
                print(f"ℹ️  Already exists: {init_file}")
        else:
            print(f"⚠️  Directory not found: {directory}")
    
    print(f"\n✨ Created {created} __init__.py files")

if __name__ == "__main__":
    create_init_files()
