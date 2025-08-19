#!/usr/bin/env python
"""
ELFIN Command Line Interface

This module provides a command-line interface for ELFIN operations, including:
- Running dimensional checks on ELFIN files
- Generating code from ELFIN files in various target languages
- Validating ELFIN files against formal specifications
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional

from alan_backend.elfin.units.checker import check_elfin_file, DimensionError
from alan_backend.elfin.codegen.rust import generate_rust_code


def check_command(args: argparse.Namespace) -> None:
    """
    Run dimensional checks on an ELFIN file.
    
    Args:
        args: Command-line arguments
    """
    print(f"Checking ELFIN file: {args.file}")
    
    try:
        errors = check_elfin_file(args.file)
        
        if errors:
            print("Found dimensional errors:")
            for expr, error in errors:
                print(f"  {expr}: {error}")
            sys.exit(1)
        else:
            print("No dimensional errors found!")
            sys.exit(0)
    except Exception as e:
        print(f"Error checking file: {e}")
        sys.exit(1)


def check_units_command(args: argparse.Namespace) -> None:
    """
    Run dimensional checks on ELFIN files but only emit warnings (don't fail build).
    
    Args:
        args: Command-line arguments
    """
    import os
    from pathlib import Path
    from elfin.standalone_parser import parse_file
    from elfin.compiler.passes.dim_checker import DimChecker
    
    # Determine what to check
    files_to_check = []
    if os.path.isdir(args.path):
        # Check all .elfin files in the directory
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.endswith(".elfin"):
                    files_to_check.append(os.path.join(root, file))
    else:
        # Check the specified file
        files_to_check.append(args.path)
    
    # Initialize the dimension checker
    checker = DimChecker()
    
    # Run the checker on all files, using the file hash cache
    start_time = time.time()
    all_diagnostics = checker.run(files_to_check)
    elapsed = time.time() - start_time
    
    # Track total warnings
    total_warnings = 0
    
    # Print diagnostics
    for file_path, diagnostics in all_diagnostics.items():
        # Count warnings
        total_warnings += len(diagnostics)
        
        # Print diagnostics for this file
        if diagnostics:
            for diag in diagnostics:
                if args.plain:
                    # Output in the format expected by VS Code problem matcher:
                    # path:line:column [severity CODE] message
                    code_str = f" {diag.code}" if diag.code else ""
                    location = f"{file_path}:{diag.line}:{diag.column}" if diag.line else f"{file_path}:1:1"
                    print(f"{location} [{diag.severity}{code_str}] {diag.message}")
                else:
                    # Standard human-readable format
                    location = f"{file_path}:{diag.line}:{diag.column}" if diag.line else file_path
                    print(f"{location} {diag.severity.upper()}: {diag.message}")
        elif not args.plain:
            print(f"No dimensional issues found in {file_path}")
    
    # Report final count but don't fail the build (only in non-plain mode)
    if not args.plain:
        if total_warnings > 0:
            print(f"Found {total_warnings} dimensional warnings across {len(files_to_check)} files")
        else:
            print(f"No dimensional issues found in {len(files_to_check)} files")
        
        # Report timing
        print(f"Check completed in {elapsed:.3f} seconds")


def generate_command(args: argparse.Namespace) -> None:
    """
    Generate code from an ELFIN file.
    
    Args:
        args: Command-line arguments
    """
    print(f"Generating {args.language} code from ELFIN file: {args.file}")
    
    try:
        # Set up options for the compiler pipeline
        options = {
            'fold': not args.no_fold,
            'use_units': not args.no_units
        }
        
        if args.language == "rust":
            output_file = generate_rust_code(
                args.file,
                args.output_dir,
                options=options
            )
            print(f"Generated Rust code: {output_file}")
        else:
            print(f"Language {args.language} not supported yet")
            sys.exit(1)
    except Exception as e:
        print(f"Error generating code: {e}")
        sys.exit(1)


def verify_command(args: argparse.Namespace) -> None:
    """
    Verify an ELFIN file against formal specifications.
    
    Args:
        args: Command-line arguments
    """
    print(f"Verifying ELFIN file: {args.file}")
    print("Verification not implemented yet")
    sys.exit(1)


def fmt_command(args: argparse.Namespace) -> None:
    """
    Format ELFIN files to enforce coding style.
    
    Args:
        args: Command-line arguments
    """
    import os
    import glob
    from pathlib import Path
    from alan_backend.elfin.formatting.elffmt import ELFINFormatter
    
    paths = []
    
    # Process each provided path (file or directory)
    for path_str in args.path:
        path = Path(path_str)
        
        if path.is_dir():
            # Find all .elfin files in the directory and subdirectories
            for elfin_file in path.glob('**/*.elfin'):
                paths.append(elfin_file)
        elif path.is_file() and path.suffix.lower() == '.elfin':
            # Single file
            paths.append(path)
        else:
            print(f"Warning: Skipping non-ELFIN file: {path}")
    
    if not paths:
        print("No ELFIN files found to format.")
        sys.exit(0)
    
    formatter = ELFINFormatter()
    check_only = args.check
    
    processed = 0
    changed = 0
    unchanged = 0
    
    for file_path in paths:
        processed += 1
        try:
            original_content = file_path.read_text()
            formatted_content = formatter.format_string(original_content)
            
            # Check if content changed
            if original_content != formatted_content:
                changed += 1
                if check_only:
                    print(f"Would format (not formatted due to --check): {file_path}")
                else:
                    file_path.write_text(formatted_content)
                    print(f"Formatted: {file_path}")
            else:
                unchanged += 1
                print(f"Already formatted: {file_path}")
                
        except Exception as e:
            print(f"Error formatting {file_path}: {e}", file=sys.stderr)
    
    print(f"Processed {processed} files: {unchanged} already formatted, {changed} would be formatted")
    
    # In check mode, exit with non-zero status if any files would be formatted
    if check_only and changed > 0:
        sys.exit(1)


def lsp_command(args: argparse.Namespace) -> None:
    """
    Start the ELFIN language server.
    
    Args:
        args: Command-line arguments
    """
    try:
        from alan_backend.elfin.lsp.server import start_server
        print("Starting ELFIN language server...")
        start_server()
    except ImportError:
        print("Error: pygls package not found. Install it with:")
        print("  pip install pygls")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for the ELFIN CLI.
    
    Args:
        argv: Command-line arguments (optional)
    """
    parser = argparse.ArgumentParser(description="ELFIN Command Line Interface")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run",
        required=True
    )
    
    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check ELFIN file for dimensional consistency"
    )
    check_parser.add_argument(
        "file",
        help="Path to ELFIN file"
    )
    check_parser.set_defaults(func=check_command)
    
    # Check-units command (warning only)
    check_units_parser = subparsers.add_parser(
        "check-units",
        help="Check ELFIN files for dimensional consistency (warnings only)"
    )
    check_units_parser.add_argument(
        "path",
        help="Path to ELFIN file or directory"
    )
    check_units_parser.add_argument(
        "--plain",
        action="store_true",
        help="Output warnings in a simple format suitable for tools"
    )
    check_units_parser.set_defaults(func=check_units_command)
    
    # Format command
    fmt_parser = subparsers.add_parser(
        "fmt",
        help="Format ELFIN files according to style guidelines"
    )
    fmt_parser.add_argument(
        "path",
        nargs="+",
        help="Path(s) to ELFIN file or directory"
    )
    fmt_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are formatted correctly without making changes"
    )
    fmt_parser.set_defaults(func=fmt_command)
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate code from ELFIN file"
    )
    generate_parser.add_argument(
        "file",
        help="Path to ELFIN file"
    )
    generate_parser.add_argument(
        "--language",
        "-l",
        default="rust",
        choices=["rust", "c", "cpp", "python"],
        help="Target language (default: rust)"
    )
    generate_parser.add_argument(
        "--output-dir",
        "-o",
        default="generated",
        help="Output directory (default: generated)"
    )
    generate_parser.add_argument(
        "--no-units",
        action="store_true",
        help="Generate code without unit safety (default: false)"
    )
    generate_parser.add_argument(
        "--no-fold",
        action="store_true",
        help="Disable constant folding optimization (default: enabled)"
    )
    generate_parser.set_defaults(func=generate_command)
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify ELFIN file against formal specifications"
    )
    verify_parser.add_argument(
        "file",
        help="Path to ELFIN file"
    )
    verify_parser.set_defaults(func=verify_command)
    
    # LSP command
    lsp_parser = subparsers.add_parser(
        "lsp",
        help="Start the ELFIN language server"
    )
    lsp_parser.set_defaults(func=lsp_command)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
