#!/usr/bin/env python3
"""
Error documentation generator for the ELFIN framework.

This script provides utilities for generating and managing error
documentation for the ELFIN framework.
"""

import argparse
import os
import pathlib
import sys
from typing import List, Optional

import logging
logger = logging.getLogger(__name__)


# Error documentation template
ERROR_DOC_TEMPLATE = """# {error_code} – "{error_title}"

## What the warning means

SUMMARY_PLACEHOLDER

## Typical reasons & quick fixes

| Cause | How to fix |
|-------|------------|
| CAUSE_1 | FIX_1 |
| CAUSE_2 | FIX_2 |
| CAUSE_3 | FIX_3 |

## What to do right now

```python
# Example remediation code
REMEDIATION_CODE_PLACEHOLDER
```

## Docs link

Full explanation, mathematical derivation, and debugging checklist are available at
https://elfin.dev/errors/{error_code}
"""


def get_docs_dir() -> pathlib.Path:
    """
    Get the docs directory for error documentation.
    
    Returns:
        Path to the docs directory
    """
    # First, try to find ELFIN_DOCS_DIR environment variable
    docs_dir = os.environ.get("ELFIN_DOCS_DIR")
    if docs_dir:
        return pathlib.Path(docs_dir) / "errors"
        
    # Otherwise, try to find docs directory relative to project root
    # Start with this script's location and work upward
    current_dir = pathlib.Path(__file__).resolve().parent
    
    # Look for docs/errors in parent directories
    for _ in range(5):  # Limit search depth to avoid infinite loop
        docs_path = current_dir / "docs" / "errors"
        if docs_path.exists() and docs_path.is_dir():
            return docs_path
            
        # Check if we've reached the project root
        if (current_dir / "setup.py").exists() or (current_dir / "pyproject.toml").exists():
            # Create docs directory
            docs_path = current_dir / "docs" / "errors"
            docs_path.mkdir(parents=True, exist_ok=True)
            return docs_path
            
        # Move up to parent directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # We've reached the root
            break
            
        current_dir = parent_dir
    
    # Last resort: use current directory
    return pathlib.Path.cwd() / "docs" / "errors"


def normalize_error_code(error_code: str) -> str:
    """
    Normalize an error code to E-XXX_YYY format.
    
    Args:
        error_code: Error code to normalize
        
    Returns:
        Normalized error code
    """
    # Strip "E-" prefix if present
    if error_code.startswith("E-"):
        error_code = error_code[2:]
        
    # Add "E-" prefix back
    return f"E-{error_code}"


def get_error_title(error_code: str) -> str:
    """
    Get a default title for an error code.
    
    Args:
        error_code: Error code
        
    Returns:
        Default title for the error code
    """
    # Strip "E-" prefix if present
    if error_code.startswith("E-"):
        error_code = error_code[2:]
    
    # Look up known error codes
    from alan_backend.elfin.errors.error_handler import VerificationError
    
    if error_code in VerificationError.ERROR_CODES:
        return VerificationError.ERROR_CODES[error_code]
    
    # Return a default title based on error code
    error_parts = error_code.split("_")
    if len(error_parts) >= 2:
        category = error_parts[0].lower()
        number = error_parts[1]
        
        if category == "lyap":
            return f"Lyapunov condition {number} not satisfied"
        elif category == "verif":
            return f"Verification error {number}"
        elif category == "param":
            return f"Parameter error {number}"
        
    return "Unknown error"


def create_error_doc(error_code: str, output_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Create a new error documentation file.
    
    Args:
        error_code: Error code to create documentation for
        output_dir: Output directory (default: docs/errors)
        
    Returns:
        Path to the created file
    """
    # Normalize error code
    error_code = normalize_error_code(error_code)
    
    # Get error title
    error_title = get_error_title(error_code)
    
    # Format the template
    doc_content = ERROR_DOC_TEMPLATE.format(
        error_code=error_code,
        error_title=error_title
    )
    
    # Get output directory
    if output_dir is None:
        output_dir = get_docs_dir()
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file
    output_file = output_dir / f"{error_code}.md"
    
    # Don't overwrite existing file
    if output_file.exists():
        logger.warning(f"Error documentation already exists: {output_file}")
        return output_file
    
    # Write the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(doc_content)
        
    logger.info(f"Created error documentation: {output_file}")
    return output_file


def verify_error_docs(error_codes: List[str], docs_dir: Optional[pathlib.Path] = None) -> bool:
    """
    Verify that error documentation exists for all error codes.
    
    Args:
        error_codes: List of error codes to verify
        docs_dir: Documentation directory (default: docs/errors)
        
    Returns:
        True if all error codes have documentation, False otherwise
    """
    if docs_dir is None:
        docs_dir = get_docs_dir()
        
    all_docs_exist = True
    
    for error_code in error_codes:
        # Normalize error code
        error_code = normalize_error_code(error_code)
        
        # Check if documentation exists
        doc_file = docs_dir / f"{error_code}.md"
        if not doc_file.exists():
            logger.warning(f"Missing error documentation: {error_code}")
            all_docs_exist = False
            
    return all_docs_exist


def build_error_docs(docs_dir: Optional[pathlib.Path] = None, output_dir: Optional[pathlib.Path] = None) -> None:
    """
    Build error documentation site using a static site generator.
    
    Args:
        docs_dir: Documentation directory (default: docs/errors)
        output_dir: Output directory (default: docs/site)
    """
    if docs_dir is None:
        docs_dir = get_docs_dir()
        
    if output_dir is None:
        output_dir = docs_dir.parent / "site"
        
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement site generation (e.g., using mkdocs or mdBook)
    # This is a placeholder for future implementation
    logger.info(f"Building error documentation site in {output_dir}")
    logger.warning("Site generation not implemented yet")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Error documentation generator for the ELFIN framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 'new' command
    new_parser = subparsers.add_parser(
        "new",
        help="Create a new error documentation file"
    )
    new_parser.add_argument(
        "error_code",
        help="Error code to create documentation for (e.g., E-LYAP-003)"
    )
    new_parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: docs/errors)"
    )
    
    # 'verify' command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify that error documentation exists for all error codes"
    )
    verify_parser.add_argument(
        "error_codes",
        nargs="+",
        help="Error codes to verify (e.g., E-LYAP-001 E-LYAP-002)"
    )
    verify_parser.add_argument(
        "--docs-dir", "-d",
        help="Documentation directory (default: docs/errors)"
    )
    
    # 'build' command
    build_parser = subparsers.add_parser(
        "build",
        help="Build error documentation site"
    )
    build_parser.add_argument(
        "--docs-dir", "-d",
        help="Documentation directory (default: docs/errors)"
    )
    build_parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: docs/site)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "new":
        output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
        create_error_doc(args.error_code, output_dir)
    elif args.command == "verify":
        docs_dir = pathlib.Path(args.docs_dir) if args.docs_dir else None
        result = verify_error_docs(args.error_codes, docs_dir)
        sys.exit(0 if result else 1)
    elif args.command == "build":
        docs_dir = pathlib.Path(args.docs_dir) if args.docs_dir else None
        output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
        build_error_docs(docs_dir, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
