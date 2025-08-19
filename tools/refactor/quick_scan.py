#!/usr/bin/env python3
"""Quick dry-run script to find files with the old path"""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

old_path = r"{PROJECT_ROOT}"
root = Path(r"D:\Dev\kha")

# Directories to skip
skip_dirs = {'.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '.cache', '__pycache__', '.pytest_cache'}
# Extensions to check
check_exts = {'.py', '.ts', '.tsx', '.js', '.jsx', '.svelte', '.json', '.md', '.txt', '.yaml', '.yml'}

found_files = []
total_scanned = 0
print(f"Scanning for files containing: {old_path}")
print(f"In directory: {root}")
print("-" * 60)

for dirpath, dirnames, filenames in os.walk(root):
    # Skip unwanted directories
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    
    # Show progress
    if total_scanned % 100 == 0:
        print(f"Scanned {total_scanned} files... Found {len(found_files)} with old path")
    
    for filename in filenames:
        filepath = Path(dirpath) / filename
        
        # Skip files without target extensions
        if filepath.suffix.lower() not in check_exts:
            continue
            
        # Skip large files (>2MB)
        try:
            if filepath.stat().st_size > 2_000_000:
                continue
        except:
            continue
        
        total_scanned += 1
        
        # Check if file contains the old path
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if old_path in content:
                count = content.count(old_path)
                found_files.append((str(filepath), count))
                print(f"FOUND: {filepath.relative_to(root)} ({count} occurrences)")
        except Exception as e:
            pass

print("-" * 60)
print(f"\nSummary:")
print(f"Total files scanned: {total_scanned}")
print(f"Files with old path: {len(found_files)}")
print(f"\nTop 10 files with most occurrences:")
for filepath, count in sorted(found_files, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {count:3d} - {Path(filepath).relative_to(root)}")

# Save to CSV
csv_path = root / "tools" / "refactor" / "quick_scan_results.csv"
with open(csv_path, 'w', newline='') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['File', 'Occurrences'])
    for filepath, count in found_files:
        writer.writerow([filepath, count])

print(f"\nFull results saved to: {csv_path}")
print(f"\nTo refactor these files, run:")
print(f"  .\\tools\\refactor\\Run-MassRefactor.ps1 -BackupDir D:\\Backups\\KhaRefactor")
