#!/usr/bin/env python3
"""
Duplicate-finder
================
Walks a codebase and reports:

1. **Exact duplicates** – files whose SHA-1 hashes match.
2. **Name clashes** – same filename but different content.

Usage
-----
    python tools/dup_finder.py
    python tools/dup_finder.py --root kha --ext .py,.js,.ts,.rs,.elfin
    python tools/dup_finder.py --root . --skip-dir node_modules --skip-dir .venv
"""

import argparse
import collections
import hashlib
import os
import pathlib
import sys
import textwrap
from typing import Iterable, Tuple, Dict, List, Set

# Directories to skip by default
SKIP_DIRS: Set[str] = {'.git', '.venv', 'node_modules', '__pycache__', 'dist', 'build'}

def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield every file under root, quietly ignoring directories we can't enter."""
    for dpath, dnames, fnames in os.walk(root, onerror=lambda e: None, followlinks=False):
        # Filter out directories we want to skip
        dnames[:] = [d for d in dnames if d not in SKIP_DIRS]
        for fname in fnames:
            yield pathlib.Path(dpath) / fname

def sha1(path: pathlib.Path, chunk: int = 65536) -> str:
    """Calculate SHA-1 hash of a file."""
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except (FileNotFoundError, PermissionError, OSError) as e:
        # Return None or raise to let caller handle
        raise

def scan(root: pathlib.Path, exts: set[str]) -> Tuple[Dict[str, List[pathlib.Path]], Dict[str, List[Tuple[pathlib.Path, str]]]]:
    """Scan directory tree for files, collecting hashes and names."""
    by_hash: Dict[str, List[pathlib.Path]] = collections.defaultdict(list)
    by_name: Dict[str, List[Tuple[pathlib.Path, str]]] = collections.defaultdict(list)

    for p in iter_files(root):
        # Skip files without the right extension
        if exts and p.suffix.lower() not in exts:
            continue
        
        try:
            digest = sha1(p)
        except (FileNotFoundError, PermissionError, OSError):
            # File vanished or is unreadable – skip
            print(f"[Warning] Skipping unreadable file: {p}", file=sys.stderr)
            continue
        
        by_hash[digest].append(p)
        by_name[p.name].append((p, digest))
    
    return by_hash, by_name

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect duplicate files by hash and by filename."
    )
    parser.add_argument("--root", default=".", help="Directory to scan")
    parser.add_argument(
        "--ext",
        default=".py,.js,.ts,.rs,.elfin",
        help="Comma-separated extensions to include",
    )
    parser.add_argument(
        "--skip-dir", 
        action='append', 
        default=[],
        help='Directory name to ignore (can be given multiple times)'
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help='Show more detailed output'
    )
    args = parser.parse_args()

    # Update SKIP_DIRS with any additional directories from command line
    if args.skip_dir:
        SKIP_DIRS.update(args.skip_dir)

    root = pathlib.Path(args.root).resolve()
    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}

    if not root.exists():
        sys.exit(f"[dup_finder] Root path does not exist: {root}")

    if args.verbose:
        print(f"[dup_finder] Scanning: {root}")
        print(f"[dup_finder] Extensions: {', '.join(sorted(exts))}")
        print(f"[dup_finder] Skipping dirs: {', '.join(sorted(SKIP_DIRS))}")

    by_hash, by_name = scan(root, exts)

    # Report results
    duplicate_count = 0
    
    print("\n=== Perfect duplicates (same SHA-1) ===")
    for digest, paths in by_hash.items():
        if len(paths) > 1:
            duplicate_count += len(paths) - 1
            print(f"\nSHA1 {digest[:10]}…")
            for p in paths:
                try:
                    rel_path = p.relative_to(root)
                except ValueError:
                    rel_path = p
                print(f"    {rel_path}")

    name_clash_count = 0
    
    print("\n=== Same filename, different content ===")
    for name, lst in by_name.items():
        hashes = {d for _, d in lst}
        if len(lst) > 1 and len(hashes) > 1:
            name_clash_count += 1
            print(f"\n{name}")
            for p, d in lst:
                try:
                    rel_path = p.relative_to(root)
                except ValueError:
                    rel_path = p
                print(f"    {rel_path}  ({d[:10]}…)")

    print(f"\n[dup_finder] Scan complete.")
    print(f"[dup_finder] Found {duplicate_count} duplicate files and {name_clash_count} name clashes.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
