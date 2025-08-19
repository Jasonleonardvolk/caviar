#!/usr/bin/env python3
"""
ELFIN Circular Reference Analyzer CLI

Command-line interface for the ELFIN static analysis tools.
"""

import argparse
import sys
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the analyzer
try:
    from alan_backend.elfin.analyzer.circular_analyzer import CircularReferenceAnalyzer, analyze_file
except ImportError:
    # Fallback for local development
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from alan_backend.elfin.analyzer.circular_analyzer import CircularReferenceAnalyzer, analyze_file


def check_command(args) -> int:
    """
    Run the circular reference and syntax check.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    exit_code = 0
    file_count = 0
    error_count = 0
    warning_count = 0
    
    # Process all files
    for file_pattern in args.files:
        for file_path in glob.glob(file_pattern):
            if not file_path.endswith('.elfin'):
                continue
                
            file_count += 1
            print(f"Checking {file_path}...")
            
            try:
                # Use the analyze_file function directly
                output_file = None
                if args.output:
                    output_dir = Path(args.output)
                    output_dir.mkdir(exist_ok=True)
                    output_file = str(output_dir / f"{Path(file_path).stem}_analysis.json")
                
                # Run analysis
                analyzer = CircularReferenceAnalyzer(file_path)
                analyzer.run_analysis()
                
                # Count issues
                file_errors = sum(1 for issue in analyzer.issues if issue['severity'] == 'error')
                file_warnings = sum(1 for issue in analyzer.issues if issue['severity'] == 'warning')
                
                error_count += file_errors
                warning_count += file_warnings
                
                # Print results
                analyzer.print_summary()
                
                # Export results if requested
                if output_file:
                    analyzer.export_results(output_file)
                
                # Update exit code
                has_errors = analyzer.has_circular_references()
                if has_errors:
                    exit_code = 1
                elif file_warnings > 0 and args.warnings_as_errors:
                    exit_code = 1
                    
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                exit_code = 2
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Files checked: {file_count}")
    print(f"Total errors: {error_count}")
    print(f"Total warnings: {warning_count}")
    
    return exit_code


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="ELFIN static analysis tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check ELFIN files for issues")
    check_parser.add_argument("files", nargs="+", help="ELFIN files to check (glob patterns supported)")
    check_parser.add_argument("--output", "-o", help="Directory to save analysis results")
    check_parser.add_argument("--warnings-as-errors", "-W", action="store_true", 
                             help="Treat warnings as errors (exit non-zero)")
    
    args = parser.parse_args()
    
    if args.command == "check":
        sys.exit(check_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
