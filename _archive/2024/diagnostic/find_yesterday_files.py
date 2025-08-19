#!/usr/bin/env python3
"""
Find all files created or modified yesterday (July 15, 2025)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import json

print("FINDING FILES MODIFIED YESTERDAY (July 15, 2025)")
print("=" * 60)

# Get yesterday's date
today = datetime(2025, 7, 16)
yesterday = datetime(2025, 7, 15)
tomorrow = datetime(2025, 7, 17)

# Root directory to search
root_dir = Path(".")

# Lists to store results
modified_files = []
created_files = []

# Extensions to focus on (code files)
code_extensions = {'.py', '.rs', '.toml', '.json', '.md', '.yaml', '.yml', '.js', '.ts', '.jsx', '.tsx'}

print(f"\nSearching for files modified on: {yesterday.strftime('%Y-%m-%d')}")
print("-" * 40)

# Walk through directory
for root, dirs, files in os.walk(root_dir):
    # Skip certain directories
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'target', '.pytest_cache', 'build', 'dist'}
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    
    for file in files:
        file_path = Path(root) / file
        
        # Skip if not a code file
        if file_path.suffix.lower() not in code_extensions:
            continue
            
        try:
            # Get file stats
            stats = file_path.stat()
            
            # Get modification time
            mtime = datetime.fromtimestamp(stats.st_mtime)
            
            # Check if modified yesterday
            if yesterday <= mtime < tomorrow:
                modified_files.append({
                    'path': str(file_path),
                    'modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'size': stats.st_size
                })
                
            # Get creation time (on Windows)
            if hasattr(stats, 'st_birthtime'):
                ctime = datetime.fromtimestamp(stats.st_birthtime)
            else:
                ctime = datetime.fromtimestamp(stats.st_ctime)
                
            # Check if created yesterday
            if yesterday <= ctime < tomorrow:
                created_files.append({
                    'path': str(file_path),
                    'created': ctime.strftime('%Y-%m-%d %H:%M:%S'),
                    'size': stats.st_size
                })
                
        except Exception as e:
            pass

# Sort by modification time
modified_files.sort(key=lambda x: x['modified'])
created_files.sort(key=lambda x: x['created'])

# Display results
print(f"\n📝 FILES MODIFIED YESTERDAY ({len(modified_files)} files):")
print("-" * 60)

for file in modified_files:
    print(f"{file['modified']} | {file['size']:>8} bytes | {file['path']}")

if created_files:
    print(f"\n✨ FILES CREATED YESTERDAY ({len(created_files)} files):")
    print("-" * 60)
    
    for file in created_files:
        print(f"{file['created']} | {file['size']:>8} bytes | {file['path']}")

# Save results to JSON
results = {
    'search_date': yesterday.strftime('%Y-%m-%d'),
    'modified_files': modified_files,
    'created_files': created_files,
    'total_modified': len(modified_files),
    'total_created': len(created_files)
}

output_file = 'yesterday_files_report.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Full report saved to: {output_file}")

# Group by directory
print("\n📁 MODIFIED FILES BY DIRECTORY:")
print("-" * 60)

dir_groups = {}
for file in modified_files:
    dir_name = str(Path(file['path']).parent)
    if dir_name not in dir_groups:
        dir_groups[dir_name] = []
    dir_groups[dir_name].append(file['path'])

for dir_name in sorted(dir_groups.keys()):
    print(f"\n{dir_name}/")
    for file_path in dir_groups[dir_name]:
        print(f"  - {Path(file_path).name}")

print("\n" + "=" * 60)
print(f"Summary: {len(modified_files)} files modified, {len(created_files)} files created on July 15, 2025")
