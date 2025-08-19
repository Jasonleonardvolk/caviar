#!/usr/bin/env python3
"""
PyProject.toml Dependency Manager
Easily add dependencies to your pyproject.toml files
"""

import sys
from pathlib import Path
import toml
import argparse

def load_pyproject(path):
    """Load pyproject.toml file"""
    with open(path, 'r') as f:
        return toml.load(f)

def save_pyproject(path, data):
    """Save pyproject.toml file"""
    with open(path, 'w') as f:
        toml.dump(data, f)
    print(f"✅ Updated: {path}")

def add_dependency(pyproject_path, dep_string):
    """Add a dependency to pyproject.toml"""
    data = load_pyproject(pyproject_path)
    
    # Ensure project.dependencies exists
    if 'project' not in data:
        data['project'] = {}
    if 'dependencies' not in data['project']:
        data['project']['dependencies'] = []
    
    # Add dependency if not already present
    if dep_string not in data['project']['dependencies']:
        data['project']['dependencies'].append(dep_string)
        save_pyproject(pyproject_path, data)
        print(f"✅ Added: {dep_string}")
    else:
        print(f"⚠️  Already exists: {dep_string}")

def list_dependencies(pyproject_path):
    """List all dependencies"""
    data = load_pyproject(pyproject_path)
    deps = data.get('project', {}).get('dependencies', [])
    
    if deps:
        print(f"\n📦 Dependencies in {pyproject_path.name}:")
        for dep in deps:
            print(f"   - {dep}")
    else:
        print(f"\n📦 No dependencies in {pyproject_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Manage pyproject.toml dependencies')
    parser.add_argument('path', help='Path to pyproject.toml')
    parser.add_argument('--add', '-a', help='Add a dependency (e.g., "numpy>=1.20.0")')
    parser.add_argument('--list', '-l', action='store_true', help='List dependencies')
    
    args = parser.parse_args()
    
    pyproject_path = Path(args.path)
    if not pyproject_path.exists():
        print(f"❌ File not found: {pyproject_path}")
        sys.exit(1)
    
    if args.add:
        add_dependency(pyproject_path, args.add)
    elif args.list:
        list_dependencies(pyproject_path)
    else:
        # Interactive mode
        print(f"📝 Managing: {pyproject_path}")
        list_dependencies(pyproject_path)
        
        print("\n🔧 Common dependencies for TORI:")
        common_deps = [
            "numpy>=1.20.0",
            "scipy>=1.7.0", 
            "networkx>=2.6.0",
            "pydantic>=2.0.0",
            "aiofiles>=0.8.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "requests>=2.28.0",
            "websockets>=10.0",
            "PyYAML>=6.0",
        ]
        
        for i, dep in enumerate(common_deps, 1):
            print(f"   {i}. {dep}")
        
        print("\n💡 Enter dependency to add (or number from list, or 'q' to quit):")
        
        while True:
            choice = input("> ").strip()
            
            if choice.lower() == 'q':
                break
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(common_deps):
                    add_dependency(pyproject_path, common_deps[idx])
                else:
                    print("❌ Invalid number")
            elif choice:
                add_dependency(pyproject_path, choice)

if __name__ == "__main__":
    # Handle both pip and poetry style toml parsing
    try:
        import tomli as toml
    except ImportError:
        try:
            import tomllib as toml
        except ImportError:
            import toml
    
    main()
