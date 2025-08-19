#!/usr/bin/env python3
"""
Simple script to find yesterday's files without using subprocess
"""

import os
from pathlib import Path
from datetime import datetime
import json

# Since we can't execute Python directly, let me search for recent files
# by checking some key directories

print("SEARCHING FOR RECENTLY MODIFIED FILES")
print("=" * 60)

# Key directories to check
key_dirs = [
    ".",
    "concept_mesh",
    "mcp_metacognitive", 
    "python/core",
    "albert",
    "ingest_pdf"
]

recent_files = []

# Function to check if file was modified recently (within last 2 days)
def is_recent(file_path):
    try:
        stats = file_path.stat()
        mtime = datetime.fromtimestamp(stats.st_mtime)
        # Check if modified in last 48 hours
        days_ago = (datetime.now() - mtime).days
        return days_ago <= 2, mtime, stats.st_size
    except:
        return False, None, 0

# Search through key directories
for dir_name in key_dirs:
    dir_path = Path(dir_name)
    if not dir_path.exists():
        continue
        
    print(f"\nChecking {dir_name}...")
    
    # Look for Python, Rust, and config files
    patterns = ['*.py', '*.rs', '*.toml', '*.json', '*.md']
    
    for pattern in patterns:
        for file_path in dir_path.rglob(pattern):
            # Skip certain directories
            skip_parts = {'.git', '__pycache__', 'target', '.venv', 'node_modules'}
            if any(skip in file_path.parts for skip in skip_parts):
                continue
                
            is_recent_file, mtime, size = is_recent(file_path)
            if is_recent_file:
                recent_files.append({
                    'path': str(file_path),
                    'modified': mtime.strftime('%Y-%m-%d %H:%M:%S') if mtime else 'Unknown',
                    'size': size,
                    'extension': file_path.suffix
                })

# Sort by modification time
recent_files.sort(key=lambda x: x['modified'], reverse=True)

# Display results
print("\n" + "=" * 60)
print("RECENTLY MODIFIED FILES (Last 48 hours):")
print("-" * 60)

for file in recent_files[:50]:  # Show top 50 most recent
    print(f"{file['modified']} | {file['extension']:>5} | {file['path']}")

# Save as JSON
with open('recent_files_scan.json', 'w') as f:
    json.dump(recent_files, f, indent=2)

print(f"\nTotal recent files found: {len(recent_files)}")
print("Full list saved to: recent_files_scan.json")
