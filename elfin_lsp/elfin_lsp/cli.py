"""
Command line interface for the ELFIN language server.

This module provides a CLI entry point for the ELFIN language server.
"""

import argparse
import logging
import sys
from typing import List, Optional

from elfin_lsp.server import ELFIN_LS, run as run_server


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments. If None, sys.argv[1:] is used.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ELFIN Language Server")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the language server")
    run_parser.add_argument("--log-file", help="Log file path")
    run_parser.add_argument("--log-level", 
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           default="INFO",
                           help="Log level")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments. If None, sys.argv[1:] is used.
    
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "run":
        # Configure logging
        log_level = getattr(logging, parsed_args.log_level)
        log_file = parsed_args.log_file or "elfin_lsp.log"
        
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        
        # Run the server
        run_server()
        return 0
    
    elif parsed_args.command == "version":
        from elfin_lsp import __version__
        print(f"ELFIN Language Server v{__version__}")
        return 0
    
    else:
        print("No command specified. Run with --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
