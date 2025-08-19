import os
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Quick check of recent files
root_dir = Path(r"{PROJECT_ROOT}")
start_date = datetime(2025, 7, 3)
count = 0
files_list = []

exclude_dirs = {'.git', 'node_modules', '.yarn', '__pycache__', '.pytest_cache', '.turbo', 'dist', 'build'}
include_ext = {'.py', '.js', '.ts', '.jsx', '.tsx', '.rs', '.yaml', '.yml', '.json', '.md', '.bat', '.sh', '.ps1', '.html', '.css', '.svelte'}

for root, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    if any(exc in str(root) for exc in exclude_dirs):
        continue
        
    for file in files[:5]:  # Sample first 5 files per dir
        file_path = Path(root) / file
        if file_path.suffix.lower() in include_ext:
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime >= start_date:
                    count += 1
                    rel_path = file_path.relative_to(root_dir)
                    files_list.append(str(rel_path).replace('\\', '/'))
                    if count <= 20:
                        print(f"{mtime.date()}: {rel_path}")
            except:
                pass

print(f"\nTotal files found (sample): {count}+")
print("\nNote: This is just a sample. Running full search...")
