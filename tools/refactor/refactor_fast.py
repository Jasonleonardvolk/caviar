#!/usr/bin/env python3
"""
refactor_fast.py - Optimized for large repositories (23GB+)
No dry run by default, minimal output, maximum speed
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast path refactoring for large repos')
    parser.add_argument('--backup-dir', help='Backup directory (recommended!)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()
    
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    quiet = args.quiet
    
    # Setup
    log_dir = ROOT / "tools" / "refactor"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"Backing up to: {backup_dir}")
    
    if not quiet:
        print(f"Refactoring {ROOT}")
        print(f"Replacing: {OLD_PATH}")
        print("-" * 40)
    
    # Process files
    total = 0
    patched = 0
    errors = 0
    last_report = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Skip unwanted directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for filename in filenames:
                filepath = Path(dirpath) / filename
                
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
                if not quiet and total - last_report >= 100:
                    print(f"Processed: {total} files, {patched} modified", end='\r')
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
                    log.write(f"{filepath} :: patched :: {count} occurrences\n")
                    
                    if not quiet and patched % 10 == 0:
                        rel = filepath.relative_to(ROOT)
                        print(f"Modified: {rel} ({count} changes)", " " * 20)
                        
                except Exception as e:
                    errors += 1
                    log.write(f"{filepath} :: error :: {e}\n")
    
    # Final report
    print(" " * 80)  # Clear the progress line
    print("-" * 40)
    print(f"Complete! Scanned: {total}, Modified: {patched}, Errors: {errors}")
    print(f"Log saved to: {log_file.relative_to(ROOT)}")
    
    if errors > 0:
        print(f"Warning: {errors} files had errors - check the log file")

if __name__ == "__main__":
    main()
