#!/usr/bin/env python3
"""
refactor_continue.py - Continue refactoring from where it crashed
Skips files that were already successfully modified
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuration
OLD_PATH = r"{PROJECT_ROOT}"
ROOT = Path(r"D:\Dev\kha")

# Directories to skip
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '.cache', '__pycache__', 
             '.pytest_cache', 'target', '.idea', '.vscode', 'tools\\dawn', 'tools/dawn'}

# Extensions to process
PROCESS_EXTS = {'.py', '.ts', '.tsx', '.js', '.jsx', '.svelte', '.wgsl', '.json', '.md', 
                '.txt', '.yaml', '.yml'}

# Python header
PYTHON_HEADER = "from pathlib import Path\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\n"

def get_already_processed():
    """Get list of files already processed by checking which files no longer contain the old path"""
    processed = set()
    
    # Quick check: files that have {PROJECT_ROOT} or ${IRIS_ROOT} are likely already processed
    print("Checking for already processed files...")
    count = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.suffix.lower() not in PROCESS_EXTS:
                continue
            try:
                if filepath.stat().st_size > 2_000_000:
                    continue
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                # If file has our markers but not the old path, it's been processed
                if ('{PROJECT_ROOT}' in content or '${IRIS_ROOT}' in content) and OLD_PATH not in content:
                    processed.add(str(filepath))
                    count += 1
                    if count % 100 == 0:
                        print(f"Found {count} already processed files...", end='\r')
            except:
                pass
    
    print(f"Found {count} already processed files.          ")
    return processed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Continue refactoring after interruption')
    parser.add_argument('--backup-dir', help='Backup directory')
    parser.add_argument('--skip-check', action='store_true', help='Skip checking for already processed files')
    args = parser.parse_args()
    
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    
    # Get already processed files
    if not args.skip_check:
        already_processed = get_already_processed()
    else:
        already_processed = set()
    
    # Setup
    log_dir = ROOT / "tools" / "refactor"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"refactor_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Backing up to: {backup_dir}")
    
    print(f"Continuing refactoring of {ROOT}")
    print(f"Replacing: {OLD_PATH}")
    print(f"Skipping {len(already_processed)} already processed files")
    print("-" * 40)
    
    # Process files
    total = 0
    skipped = 0
    patched = 0
    errors = 0
    last_report = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:  # UTF-8 encoding!
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Skip unwanted directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for filename in filenames:
                filepath = Path(dirpath) / filename
                
                # Skip if already processed
                if str(filepath) in already_processed:
                    skipped += 1
                    continue
                
                # Skip wrong extensions
                if filepath.suffix.lower() not in PROCESS_EXTS:
                    continue
                
                # Skip large files
                try:
                    if filepath.stat().st_size > 2_000_000:
                        continue
                except:
                    continue
                
                total += 1
                
                # Progress report every 100 files
                if total - last_report >= 100:
                    print(f"Processed: {total} new files, {patched} modified, {skipped} skipped", end='\r')
                    last_report = total
                
                try:
                    # Read file
                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    
                    if OLD_PATH not in content:
                        continue
                    
                    # Backup if needed
                    if backup_dir:
                        try:
                            rel = filepath.relative_to(ROOT)
                            backup_path = backup_dir / rel
                            backup_path.parent.mkdir(parents=True, exist_ok=True)
                            if not backup_path.exists():
                                import shutil
                                shutil.copy2(filepath, backup_path)
                        except:
                            pass
                    
                    # Patch content
                    if filepath.suffix.lower() == '.py':
                        # Python file
                        new_content = content
                        if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in new_content:
                            if "from pathlib import Path" in new_content:
                                new_content = new_content.replace("from pathlib import Path\n", PYTHON_HEADER, 1)
                            else:
                                new_content = PYTHON_HEADER + new_content
                        new_content = new_content.replace(OLD_PATH, "{PROJECT_ROOT}")
                    else:
                        # Other files
                        new_content = content.replace(OLD_PATH, "${IRIS_ROOT}")
                    
                    # Write file
                    filepath.write_text(new_content, encoding='utf-8')
                    
                    patched += 1
                    count = content.count(OLD_PATH)
                    log.write(f"{str(filepath)} :: patched :: {count} occurrences\n")
                    
                    if patched % 10 == 0:
                        rel = filepath.relative_to(ROOT)
                        print(f"Modified: {rel} ({count} changes)", " " * 30)
                        
                except Exception as e:
                    errors += 1
                    log.write(f"{str(filepath)} :: error :: {str(e)}\n")
    
    # Final report
    print(" " * 80)  # Clear the progress line
    print("-" * 40)
    print(f"Complete!")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Newly scanned: {total}")
    print(f"  Newly modified: {patched}")
    print(f"  Errors: {errors}")
    print(f"Log saved to: {log_file.relative_to(ROOT)}")
    
    if errors > 0:
        print(f"Warning: {errors} files had errors - check the log file")

if __name__ == "__main__":
    main()
