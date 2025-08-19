#!/usr/bin/env python3
"""
TORI Safe File Writer Utility

PURPOSE:
    Password-protected file writing utility for secure operations.
    Prevents unauthorized file modifications in the TORI directory.

WHAT IT DOES:
    - Validates write operations are within target directory
    - Requires password authentication before writing
    - Provides secure file write operations with confirmation
    - Prevents directory traversal attacks

USAGE:
    from scripts.safe_write import safe_write_file
    safe_write_file("path/to/file.txt", b"content")

SECURITY NOTES:
    - Password is hardcoded (for development only)
    - Should use environment variables or secure config in production
    - File paths are validated against target directory
    - Uses absolute path resolution to prevent traversal

AUTHOR: TORI System Maintenance
LAST UPDATED: 2025-01-26
"""

import os
import getpass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configuration constants
TARGET_DIR = r"{PROJECT_ROOT}"
PASSWORD = "jason"  # TODO: Use environment variable in production

def check_password() -> bool:
    """Check password with user input"""
    try:
        pw = getpass.getpass("Enter password to write to kha directory: ")
        return pw == PASSWORD
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return False
    except Exception as e:
        print(f"Password check failed: {e}")
        return False

def safe_write_file(filename: str, data: bytes) -> None:
    """
    Safely write file with security checks
    
    Args:
        filename: Path to file to write
        data: Binary data to write
        
    Raises:
        PermissionError: If security checks fail
        OSError: If file operations fail
    """
    # Validate path is within target directory
    abs_path = os.path.abspath(filename)
    target_abs = os.path.abspath(TARGET_DIR)
    
    if not abs_path.startswith(target_abs):
        raise PermissionError(
            f"Write blocked: Path '{abs_path}' is not within target directory '{target_abs}'"
        )
    
    # Check password
    if not check_password():
        raise PermissionError("Write blocked: Incorrect password or cancelled")
    
    # Ensure parent directory exists
    parent_dir = Path(abs_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write file securely
    try:
        with open(abs_path, "wb") as f:
            f.write(data)
        print(f"Write successful: {filename}")
        print(f"Wrote {len(data)} bytes to {abs_path}")
    except Exception as e:
        raise OSError(f"Failed to write file: {e}") from e

def safe_write_text(filename: str, text: str, encoding: str = 'utf-8') -> None:
    """
    Safely write text file with security checks
    
    Args:
        filename: Path to file to write
        text: Text content to write
        encoding: Text encoding (default: utf-8)
    """
    safe_write_file(filename, text.encode(encoding))

# Example usage and testing
if __name__ == "__main__":
    print("TORI Safe File Writer Utility")
    print("=" * 40)
    
    # Example: try to write a test file
    target_file = os.path.join(TARGET_DIR, "test_safe_write.txt")
    test_content = b"Hello, world! This is a test of the safe write utility."
    
    try:
        print(f"Attempting to write test file: {target_file}")
        safe_write_file(target_file, test_content)
        print("Test successful!")
        
        # Clean up test file
        try:
            os.remove(target_file)
            print("Test file cleaned up")
        except:
            pass
            
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except Exception as e:
        print(f"Error: {e}")
