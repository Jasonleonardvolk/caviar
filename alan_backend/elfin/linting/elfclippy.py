"""
ELFIN code linter.

This module provides tools for linting ELFIN code files,
similar to clippy for Rust or pylint for Python.
"""

import re
import sys
import argparse
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set

# ELFIN syntax patterns
BLOCK_START = re.compile(r'^\s*(psi|barrier|lyapunov|mode|system)\s+([A-Za-z0-9_]+)\s*{')
BLOCK_END = re.compile(r'^\s*}')
STATEMENT = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)(?:;|$)')
COMMENT = re.compile(r'^\s*#(.*)$')
EMPTY_LINE = re.compile(r'^\s*$')
PSI_MODE_REFERENCE = re.compile(r'psi\(([A-Za-z0-9_]+)\)')
BARRIER_REFERENCE = re.compile(r'barrier\(([A-Za-z0-9_]+)\)')


class LintLevel(Enum):
    """Severity level for lint warnings."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


@dataclass
class LintMessage:
    """A linting message."""
    level: LintLevel
    message: str
    line: int
    column: int
    file: Path
    code: str
    
    def format(self) -> str:
        """Format the message for display."""
        level_str = self.level.value.upper()
        return f"{self.file}:{self.line}:{self.column}: {level_str} [{self.code}]: {self.message}"


class ELFINLinter:
    """
    Lints ELFIN code for common issues.
    
    Key checks:
    - Shadowed psi modes
    - Unused barriers
    - Missing semicolons
    - Unbalanced braces
    - References to undefined psi modes or barriers
    """
    
    def __init__(self):
        """Initialize linter."""
        self.messages: List[LintMessage] = []
    
    def lint_file(self, file_path: Path) -> List[LintMessage]:
        """
        Lint an ELFIN file.
        
        Args:
            file_path: Path to the file to lint
            
        Returns:
            List of lint messages
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.lint_string(content, file_path)
    
    def lint_string(self, code: str, file_path: Path) -> List[LintMessage]:
        """
        Lint ELFIN code.
        
        Args:
            code: ELFIN code to lint
            file_path: Path to the file (for reporting)
            
        Returns:
            List of lint messages
        """
        self.messages = []
        
        lines = code.split('\n')
        
        # Track blocks and identifiers
        blocks = []
        defined_psi_modes = set()
        defined_barriers = set()
        referenced_psi_modes = set()
        referenced_barriers = set()
        
        # First pass: collect definitions
        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or COMMENT.match(stripped):
                continue
            
            # Track block starts
            block_match = BLOCK_START.match(stripped)
            if block_match:
                block_type = block_match.group(1)
                block_name = block_match.group(2)
                
                blocks.append((block_type, block_name, line_num))
                
                # Track defined entities
                if block_type == 'psi' or block_type == 'mode':
                    if block_name in defined_psi_modes:
                        self.messages.append(LintMessage(
                            level=LintLevel.WARNING,
                            message=f"Shadowed psi mode '{block_name}'",
                            line=line_num,
                            column=line.find(block_name) + 1,
                            file=file_path,
                            code="E-PSI-SHADOW"
                        ))
                    defined_psi_modes.add(block_name)
                
                elif block_type == 'barrier':
                    if block_name in defined_barriers:
                        self.messages.append(LintMessage(
                            level=LintLevel.WARNING,
                            message=f"Duplicate barrier '{block_name}'",
                            line=line_num,
                            column=line.find(block_name) + 1,
                            file=file_path,
                            code="E-BARR-DUP"
                        ))
                    defined_barriers.add(block_name)
                
                continue
            
            # Track block ends
            if BLOCK_END.match(stripped):
                if not blocks:
                    self.messages.append(LintMessage(
                        level=LintLevel.ERROR,
                        message="Unbalanced closing brace",
                        line=line_num,
                        column=line.find('}') + 1,
                        file=file_path,
                        code="E-SYNTAX-BRACE"
                    ))
                else:
                    blocks.pop()
                continue
            
            # Check statements
            statement_match = STATEMENT.match(stripped)
            if statement_match:
                # Check for missing semicolon
                if not stripped.endswith(';'):
                    self.messages.append(LintMessage(
                        level=LintLevel.WARNING,
                        message="Missing semicolon at end of statement",
                        line=line_num,
                        column=len(line.rstrip()) + 1,
                        file=file_path,
                        code="E-SYNTAX-SEMI"
                    ))
                
                # Rest of statement processing can go here
                continue
            
            # Check for references to psi modes
            for match in PSI_MODE_REFERENCE.finditer(line):
                mode_name = match.group(1)
                referenced_psi_modes.add(mode_name)
            
            # Check for references to barriers
            for match in BARRIER_REFERENCE.finditer(line):
                barrier_name = match.group(1)
                referenced_barriers.add(barrier_name)
        
        # Check for unbalanced blocks
        if blocks:
            for block_type, block_name, line_num in blocks:
                self.messages.append(LintMessage(
                    level=LintLevel.ERROR,
                    message=f"Unclosed {block_type} block '{block_name}'",
                    line=line_num,
                    column=1,
                    file=file_path,
                    code="E-SYNTAX-BLOCK"
                ))
        
        # Check for undefined references
        for mode_name in referenced_psi_modes:
            if mode_name not in defined_psi_modes:
                # Find the line where the reference occurs
                for i, line in enumerate(lines):
                    if f"psi({mode_name})" in line:
                        line_num = i + 1
                        col = line.find(f"psi({mode_name})") + 1
                        self.messages.append(LintMessage(
                            level=LintLevel.ERROR,
                            message=f"Reference to undefined psi mode '{mode_name}'",
                            line=line_num,
                            column=col,
                            file=file_path,
                            code="E-PSI-UNDEF"
                        ))
                        break
        
        for barrier_name in referenced_barriers:
            if barrier_name not in defined_barriers:
                # Find the line where the reference occurs
                for i, line in enumerate(lines):
                    if f"barrier({barrier_name})" in line:
                        line_num = i + 1
                        col = line.find(f"barrier({barrier_name})") + 1
                        self.messages.append(LintMessage(
                            level=LintLevel.ERROR,
                            message=f"Reference to undefined barrier '{barrier_name}'",
                            line=line_num,
                            column=col,
                            file=file_path,
                            code="E-BARR-UNDEF"
                        ))
                        break
        
        # Check for unused barriers
        for barrier_name in defined_barriers:
            if barrier_name not in referenced_barriers:
                # Find the line where the barrier is defined
                for i, line in enumerate(lines):
                    match = BLOCK_START.match(line.strip())
                    if match and match.group(1) == 'barrier' and match.group(2) == barrier_name:
                        line_num = i + 1
                        col = line.find(barrier_name) + 1
                        self.messages.append(LintMessage(
                            level=LintLevel.WARNING,
                            message=f"Unused barrier '{barrier_name}'",
                            line=line_num,
                            column=col,
                            file=file_path,
                            code="E-BARR-UNUSED"
                        ))
                        break
        
        return self.messages


def lint_files(files: List[Path]) -> Tuple[int, int]:
    """
    Lint multiple ELFIN files.
    
    Args:
        files: List of file paths to lint
        
    Returns:
        Tuple of (number of files processed, number of messages)
    """
    linter = ELFINLinter()
    
    processed = 0
    total_messages = 0
    
    for file in files:
        processed += 1
        try:
            messages = linter.lint_file(file)
            total_messages += len(messages)
            
            if messages:
                print(f"Linting {file}:")
                for msg in messages:
                    print(f"  {msg.format()}")
                print()
            else:
                print(f"No lint issues in {file}")
        except Exception as e:
            print(f"Error linting {file}: {e}", file=sys.stderr)
    
    return processed, total_messages


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lint ELFIN code files")
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="Files to lint"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    processed, messages = lint_files(args.files)
    
    if messages > 0:
        print(f"Found {messages} lint issues in {processed} files")
        sys.exit(1)
    else:
        print(f"No lint issues found in {processed} files")
        sys.exit(0)


if __name__ == "__main__":
    main()
