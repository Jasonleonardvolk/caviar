#!/usr/bin/env python3
"""
Lightweight CI Runner
====================
Runs all hygiene tools in sequence and reports results.

Usage
-----
    python tools/ci_runner.py
    python tools/ci_runner.py --skip-dir node_modules
    python tools/ci_runner.py --verbose
"""

import subprocess
import pathlib
import sys
import argparse
import time
from typing import List, Tuple

def run_command(cmd: List[str], verbose: bool = False) -> Tuple[int, float]:
    """Run a command and return exit code and execution time."""
    print('▶', ' '.join(cmd))
    
    start_time = time.time()
    
    if verbose:
        # Show output in real-time
        result = subprocess.call(cmd)
    else:
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ❌ Failed with exit code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")
        else:
            print(f"  ✓ Completed successfully")
            
        return result.returncode, time.time() - start_time
    
    return result if verbose else result.returncode, time.time() - start_time

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all hygiene tools for the project."
    )
    parser.add_argument(
        "--skip-dir",
        action='append',
        default=[],
        help='Directory name to skip in all tools (can be given multiple times)'
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help='Show detailed output from each tool'
    )
    parser.add_argument(
        "--root",
        help='Override root directory (default: parent of tools directory)'
    )
    args = parser.parse_args()
    
    # Determine root directory
    if args.root:
        root = pathlib.Path(args.root).resolve()
    else:
        root = pathlib.Path(__file__).resolve().parents[1]
    
    print(f"[CI Runner] Working directory: {root}")
    print(f"[CI Runner] Running hygiene checks...\n")
    
    # Build skip-dir arguments
    skip_args = []
    for skip_dir in args.skip_dir:
        skip_args.extend(['--skip-dir', skip_dir])
    
    # Define tools to run
    steps = [
        {
            'name': 'Duplicate Finder',
            'cmd': ['python', 'tools/dup_finder.py', '--root', str(root)] + skip_args,
            'outputs': []
        },
        {
            'name': 'Dependency Tree',
            'cmd': ['python', 'tools/deps_tree.py', '--root', str(root)] + skip_args,
            'outputs': ['deps_current.md']
        },
        {
            'name': 'Filesystem Map',
            'cmd': ['python', 'tools/gen_map.py', '--root', str(root)] + skip_args,
            'outputs': ['filesystem_map.md']
        },
    ]
    
    # Add verbose flag if requested
    if args.verbose:
        for step in steps:
            step['cmd'].append('--verbose')
    
    # Track results
    results = []
    total_time = 0
    
    # Run each tool
    for step in steps:
        print(f"\n{'='*60}")
        print(f"Running: {step['name']}")
        print('='*60)
        
        exit_code, exec_time = run_command(step['cmd'], args.verbose)
        total_time += exec_time
        
        results.append({
            'name': step['name'],
            'success': exit_code == 0,
            'time': exec_time,
            'outputs': step['outputs']
        })
        
        if exit_code != 0:
            print(f"\n❌ {step['name']} failed!")
            print(f"\n[CI Runner] Stopping due to failure.")
            return 1
    
    # Summary report
    print(f"\n{'='*60}")
    print("CI Summary")
    print('='*60)
    
    for result in results:
        status = "✓" if result['success'] else "❌"
        print(f"{status} {result['name']:<20} ({result['time']:.1f}s)")
        if result['outputs']:
            for output in result['outputs']:
                if pathlib.Path(output).exists():
                    print(f"    → Generated: {output}")
    
    print(f"\nTotal execution time: {total_time:.1f}s")
    print(f"\n✅ All CI checks passed!")
    
    # List generated files
    generated_files = []
    for result in results:
        for output in result['outputs']:
            if pathlib.Path(output).exists():
                generated_files.append(output)
    
    if generated_files:
        print(f"\nGenerated reports:")
        for f in generated_files:
            print(f"  - {f}")
        print(f"\nYou can now commit these files:")
        print(f"  git add {' '.join(generated_files)}")
        print(f"  git commit -m \"Update hygiene reports\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
