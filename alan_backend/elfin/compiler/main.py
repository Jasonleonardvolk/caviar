#!/usr/bin/env python3
"""
ELFIN Compiler - Kia Magic Compiler

Main entry point for the ELFIN compiler. This module provides the command-line interface
and the main compilation pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Use relative imports
from .grammar.parser import parse_file, ELFINSyntaxError
from .ast.megatron import ELFINMegatron


def compile_file(file_path, output_path=None, target_language='python', verbose=False):
    """
    Compile an ELFIN file to the target language.
    
    Args:
        file_path: Path to the ELFIN file to compile
        output_path: Path to write the output file (optional)
        target_language: Target language (python or cpp)
        verbose: Whether to print verbose output
        
    Returns:
        The path to the output file
    """
    if verbose:
        print(f"Compiling {file_path} to {target_language}...")
    
    try:
        # Parse the file into an AST
        ast = parse_file(file_path)
        
        if verbose:
            print("Parsing successful!")
            print(f"AST root: {type(ast).__name__}")
        
        # TODO: Add code generation step
        # For now, we'll just return success without actually generating code
        
        if output_path:
            if verbose:
                print(f"Output would be written to {output_path}")
        
        return True
    except ELFINSyntaxError as e:
        print(f"Syntax error: {e}")
        return False
    except Exception as e:
        print(f"Compilation error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for the compiler."""
    parser = argparse.ArgumentParser(
        description="ELFIN Compiler - Kia Magic Compiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", help="Input ELFIN file to compile")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-t", "--target", choices=["python", "cpp"], default="python",
                       help="Target language")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    # If output is not specified, derive it from the input
    if not args.output:
        input_path = Path(args.input)
        output_dir = input_path.parent
        
        if args.target == "python":
            output_ext = ".py"
        else:
            output_ext = ".cpp"
        
        args.output = output_dir / (input_path.stem + output_ext)
    
    # Compile the file
    success = compile_file(args.input, args.output, args.target, args.verbose)
    
    # Return the appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
