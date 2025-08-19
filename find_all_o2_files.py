#!/usr/bin/env python3
"""
Find all files created or modified since July 3rd, 2025
"""

import os
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json

def get_modified_files(root_dir, start_date):
    """Find all files modified since start_date"""
    
    # Directories to exclude
    exclude_dirs = {
        '.git', 'node_modules', '.yarn', '__pycache__', 
        '.pytest_cache', '.turbo', 'dist', 'build', '.next',
        '.vscode', '.idea', 'coverage', '.nyc_output'
    }
    
    # Extensions to include
    include_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.rs', 
        '.yaml', '.yml', '.json', '.md', '.bat', '.sh', 
        '.ps1', '.html', '.css', '.svelte', '.vue',
        '.toml', '.txt', '.sql', '.proto', '.graphql'
    }
    
    modified_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Skip if current path contains excluded directory
        if any(exc in root for exc in exclude_dirs):
            continue
            
        for file in files:
            file_path = Path(root) / file
            
            # Skip if not a programming file
            if file_path.suffix.lower() not in include_extensions:
                continue
                
            try:
                # Get file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime >= start_date:
                    relative_path = file_path.relative_to(root_dir)
                    modified_files.append({
                        'path': str(relative_path).replace('\\', '/'),
                        'modified': mtime.isoformat(),
                        'size': file_path.stat().st_size
                    })
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                
    return sorted(modified_files, key=lambda x: x['modified'], reverse=True)

def categorize_files(files):
    """Categorize files by type and directory"""
    categories = {
        'python_core': [],
        'python_tests': [],
        'rust': [],
        'typescript': [],
        'javascript': [],
        'config': [],
        'docs': [],
        'scripts': [],
        'other': []
    }
    
    for file in files:
        path = file['path']
        
        if path.endswith('.py'):
            if 'test' in path or 'tests/' in path:
                categories['python_tests'].append(path)
            else:
                categories['python_core'].append(path)
        elif path.endswith('.rs'):
            categories['rust'].append(path)
        elif path.endswith(('.ts', '.tsx')):
            categories['typescript'].append(path)
        elif path.endswith(('.js', '.jsx')):
            categories['javascript'].append(path)
        elif path.endswith(('.yaml', '.yml', '.json', '.toml')):
            categories['config'].append(path)
        elif path.endswith('.md'):
            categories['docs'].append(path)
        elif path.endswith(('.bat', '.sh', '.ps1')):
            categories['scripts'].append(path)
        else:
            categories['other'].append(path)
            
    return categories

if __name__ == "__main__":
    root_dir = Path(r"{PROJECT_ROOT}")
    start_date = datetime(2025, 7, 3)
    
    print(f"Searching for files modified since {start_date.date()}...")
    print(f"Root directory: {root_dir}")
    print("-" * 80)
    
    # Find all modified files
    modified_files = get_modified_files(root_dir, start_date)
    
    print(f"\nFound {len(modified_files)} files modified since July 3rd, 2025\n")
    
    # Categorize files
    categories = categorize_files(modified_files)
    
    # Print summary by category
    for category, files in categories.items():
        if files:
            print(f"\n{category.upper().replace('_', ' ')} ({len(files)} files):")
            for file in sorted(files)[:10]:  # Show first 10
                print(f"  - {file}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
    
    # Save full list to file
    output = {
        'summary': {
            'total_files': len(modified_files),
            'start_date': start_date.isoformat(),
            'categories': {k: len(v) for k, v in categories.items()}
        },
        'files': modified_files,
        'categorized': categories
    }
    
    output_file = root_dir / "O2_COMPLETE_FILE_LIST.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFull list saved to: {output_file}")
    
    # Generate markdown report
    md_content = f"""# Complete O2 Files List - All Files Modified Since July 3rd, 2025

## Summary
- **Total Files**: {len(modified_files)}
- **Search Date**: Since {start_date.date()}
- **Generated**: {datetime.now().isoformat()}

## Files by Category

"""
    
    for category, files in categories.items():
        if files:
            md_content += f"### {category.upper().replace('_', ' ')} ({len(files)} files)\n\n"
            for file in sorted(files):
                md_content += f"- `{file}`\n"
            md_content += "\n"
    
    md_file = root_dir / "O2_COMPLETE_FILE_LIST.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Markdown report saved to: {md_file}")
