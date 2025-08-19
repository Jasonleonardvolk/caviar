#!/usr/bin/env python3
"""
mass_refactor_simple.py - Simple, reliable path refactoring based on working quick_scan
"""
import os
import sys
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
OLD_PATH = r"{PROJECT_ROOT}"
ROOT = Path(r"D:\Dev\kha")
BACKUP_DIR = None  # Will be set from command line
DRY_RUN = False
RESUME = False

# Directories to skip
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '.cache', '__pycache__', 
             '.pytest_cache', 'target', '.idea', '.vscode', 'tools\\dawn', 'tools/dawn'}

# Extensions to process
PROCESS_EXTS = {'.py', '.ts', '.tsx', '.js', '.jsx', '.svelte', '.wgsl', '.json', '.md', 
                '.txt', '.yaml', '.yml'}

# Python header to inject
PYTHON_HEADER = "from pathlib import Path\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\n"

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Refactor absolute paths to relative')
    parser.add_argument('--root', default=str(ROOT), help='Repository root')
    parser.add_argument('--old', default=OLD_PATH, help='Old path to replace')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying')
    parser.add_argument('--backup-dir', help='Directory to backup original files')
    parser.add_argument('--resume', action='store_true', help='Resume from previous state')
    parser.add_argument('--text-token', default='IRIS_ROOT', help='Token for non-Python files')
    return parser.parse_args()

def load_resume_state(state_file):
    """Load list of already processed files"""
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_resume_state(state_file, processed_files):
    """Save list of processed files"""
    try:
        with open(state_file, 'w') as f:
            json.dump(list(processed_files), f)
    except:
        pass

def patch_python_file(content, old_path):
    """Replace paths in Python files and add header if needed"""
    if old_path not in content:
        return content, 0
    
    new_content = content
    
    # Add header if not present
    if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in new_content:
        if "from pathlib import Path" in new_content:
            # Replace existing Path import with our header
            new_content = new_content.replace("from pathlib import Path\n", PYTHON_HEADER, 1)
        else:
            # Add header at the beginning
            new_content = PYTHON_HEADER + new_content
    
    # Replace all occurrences of old path
    count = new_content.count(old_path)
    new_content = new_content.replace(old_path, "{PROJECT_ROOT}")
    
    return new_content, count

def patch_text_file(content, old_path, token='IRIS_ROOT'):
    """Replace paths in non-Python text files"""
    if old_path not in content:
        return content, 0
    
    count = content.count(old_path)
    new_content = content.replace(old_path, f"${{{token}}}")
    
    return new_content, count

def backup_file(filepath, backup_dir, root):
    """Create backup of file before modification"""
    if not backup_dir:
        return
    
    try:
        rel_path = Path(filepath).relative_to(root)
        backup_path = Path(backup_dir) / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not backup_path.exists():
            shutil.copy2(filepath, backup_path)
    except Exception as e:
        print(f"Warning: Could not backup {filepath}: {e}")

def process_file(filepath, old_path, root, backup_dir, dry_run, token='IRIS_ROOT'):
    """Process a single file"""
    try:
        # Read file
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        
        if old_path not in content:
            return 'skipped', 0
        
        # Determine how to patch based on extension
        if filepath.suffix.lower() == '.py':
            new_content, count = patch_python_file(content, old_path)
        else:
            new_content, count = patch_text_file(content, old_path, token)
        
        if count == 0:
            return 'nochange', 0
        
        if not dry_run:
            # Backup original
            if backup_dir:
                backup_file(filepath, backup_dir, root)
            
            # Write modified content
            filepath.write_text(new_content, encoding='utf-8')
        
        return 'patched', count
        
    except Exception as e:
        return f'error: {e}', 0

def main():
    args = parse_args()
    
    root = Path(args.root)
    old_path = args.old
    dry_run = args.dry_run
    backup_dir = args.backup_dir
    token = args.text_token
    
    # Setup directories
    refactor_dir = root / "tools" / "refactor"
    refactor_dir.mkdir(parents=True, exist_ok=True)
    
    state_file = refactor_dir / "refactor_state.json"
    log_file = refactor_dir / f"refactor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    csv_file = refactor_dir / "refactor_plan.csv"
    
    # Load resume state
    processed = load_resume_state(state_file) if args.resume else set()
    
    # Create backup directory if specified
    if backup_dir and not dry_run:
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'DRY RUN: ' if dry_run else ''}Refactoring paths in {root}")
    print(f"Replacing: {old_path}")
    print(f"With: {{PROJECT_ROOT}} (Python) or ${{{token}}} (other files)")
    if backup_dir and not dry_run:
        print(f"Backing up to: {backup_dir}")
    print("-" * 60)
    
    # Scan and process files
    results = []
    total_scanned = 0
    total_patched = 0
    total_errors = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip unwanted directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for filename in filenames:
                filepath = Path(dirpath) / filename
                
                # Skip if already processed (resume mode)
                if str(filepath) in processed:
                    continue
                
                # Skip files without target extensions
                if filepath.suffix.lower() not in PROCESS_EXTS:
                    continue
                
                # Skip large files (>2MB)
                try:
                    if filepath.stat().st_size > 2_000_000:
                        continue
                except:
                    continue
                
                total_scanned += 1
                
                # Show progress
                if total_scanned % 100 == 0:
                    print(f"Processed {total_scanned} files... ({total_patched} changed)")
                
                # Process the file
                status, count = process_file(filepath, old_path, root, backup_dir, dry_run, token)
                
                if status == 'patched':
                    total_patched += 1
                    rel_path = filepath.relative_to(root)
                    print(f"{'[DRY] ' if dry_run else ''}PATCHED: {rel_path} ({count} occurrences)")
                    log.write(f"{filepath} :: {status} :: {count}\n")
                    results.append((str(filepath), count))
                elif status.startswith('error'):
                    total_errors += 1
                    log.write(f"{filepath} :: {status}\n")
                
                # Update resume state periodically
                if not dry_run and total_scanned % 500 == 0:
                    processed.add(str(filepath))
                    save_resume_state(state_file, processed)
                
                # Mark as processed
                if not dry_run:
                    processed.add(str(filepath))
    
    # Save final state
    if not dry_run:
        save_resume_state(state_file, processed)
    
    # Save results to CSV
    if dry_run and results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Occurrences'])
            for filepath, count in results:
                writer.writerow([filepath, count])
        print(f"\nDry run plan saved to: {csv_file}")
    
    # Print summary
    print("-" * 60)
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"Total files scanned: {total_scanned}")
    print(f"Files {'would be ' if dry_run else ''}patched: {total_patched}")
    if total_errors:
        print(f"Errors: {total_errors}")
    
    if dry_run and total_patched > 0:
        print(f"\nTo apply these changes, run without --dry-run:")
        print(f"  python {Path(__file__).name} --backup-dir D:\\Backups\\KhaRefactor")

if __name__ == "__main__":
    main()
