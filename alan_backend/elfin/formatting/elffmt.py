"""
ELFIN code formatter.

This module provides tools for formatting ELFIN code files,
similar to rustfmt for Rust or black for Python.
"""

import re
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set

# ELFIN syntax patterns
BLOCK_START = re.compile(r'^\s*(psi|barrier|lyapunov|mode|system|hybrid_system|helpers|params|flow_dynamics|controller)\s+([A-Za-z0-9_]+)?\s*{')
BLOCK_END = re.compile(r'^\s*}')
STATEMENT = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)(?:;|$)')
PARAM_DEF = re.compile(r'^\s*([A-Za-z0-9_]+)\s*(?::\s*([A-Za-z0-9_]+))?\s*(?:\[\s*([^]]+)\s*\])?\s*=\s*(.*?)(?:;|$)')
ASSIGNMENT = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)(?:;|$)')
IMPORT_STMT = re.compile(r'^\s*import\s+([A-Za-z0-9_]+)\s+from\s+"([^"]+)"\s*;?')
COMMENT = re.compile(r'^\s*#(.*)$')
EMPTY_LINE = re.compile(r'^\s*$')
MAX_LINE_WIDTH = 80


class ELFINFormatter:
    """
    Formats ELFIN code according to the style guide.
    
    Key style rules:
    - 2 space indentation
    - Blocks have opening brace on same line as declaration
    - Consistent spacing around operators
    - One statement per line
    - Lines limited to 80 columns
    - Aligned equals signs in parameter blocks
    - Compact unit annotations
    """
    
    def __init__(self, indent_size: int = 2, max_width: int = MAX_LINE_WIDTH):
        """
        Initialize formatter.
        
        Args:
            indent_size: Number of spaces per indentation level
            max_width: Maximum line width
        """
        self.indent_size = indent_size
        self.max_width = max_width
    
    def format_file(self, file_path: Path) -> str:
        """
        Format an ELFIN file.
        
        Args:
            file_path: Path to the file to format
            
        Returns:
            Formatted code as a string
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.format_string(content)
    
    def format_string(self, code: str) -> str:
        """
        Format ELFIN code.
        
        Args:
            code: ELFIN code to format
            
        Returns:
            Formatted code
        """
        lines = code.split('\n')
        formatted_lines = []
        
        indent_level = 0
        in_params_block = False
        param_defs = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                formatted_lines.append('')
                i += 1
                continue
            
            # Handle comments
            comment_match = COMMENT.match(line)
            if comment_match:
                comment = comment_match.group(1)
                formatted_lines.append(' ' * (indent_level * self.indent_size) + f'# {comment.strip()}')
                i += 1
                continue
            
            # Handle block end
            if BLOCK_END.match(stripped):
                # If we're exiting a params block, format the collected params
                if in_params_block:
                    formatted_params = self._format_param_block(param_defs, indent_level)
                    formatted_lines.extend(formatted_params)
                    param_defs = []
                    in_params_block = False
                
                indent_level = max(0, indent_level - 1)
                formatted_lines.append(' ' * (indent_level * self.indent_size) + '}')
                i += 1
                continue
            
            # Handle block start
            block_match = BLOCK_START.match(stripped)
            if block_match:
                block_type = block_match.group(1)
                block_name = block_match.group(2) or ""
                
                # Format the block header
                if block_name:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{block_type} {block_name} {{')
                else:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{block_type} {{')
                
                indent_level += 1
                
                # Check if this is a params block
                if block_type == 'params':
                    in_params_block = True
                
                i += 1
                continue
            
            # Handle import statements
            import_match = IMPORT_STMT.match(stripped)
            if import_match:
                module = import_match.group(1)
                path = import_match.group(2)
                formatted_lines.append(' ' * (indent_level * self.indent_size) + f'import {module} from "{path}";')
                i += 1
                continue
            
            # In params block, collect parameter definitions for alignment
            if in_params_block:
                param_match = PARAM_DEF.match(stripped) or ASSIGNMENT.match(stripped)
                if param_match:
                    param_defs.append((param_match, indent_level))
                    i += 1
                    continue
            
            # Handle parameter definitions outside params block
            param_match = PARAM_DEF.match(stripped)
            if param_match:
                param = param_match.group(1)
                type_name = param_match.group(2)
                unit = param_match.group(3)
                value = param_match.group(4).strip()
                
                # Format the parameter with unit in compact form
                if type_name and unit:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{param}: {type_name}[{unit}] = {value};')
                elif type_name:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{param}: {type_name} = {value};')
                elif unit:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{param}[{unit}] = {value};')
                else:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{param} = {value};')
                
                i += 1
                continue
            
            # Handle statements
            statement_match = STATEMENT.match(stripped)
            if statement_match:
                key = statement_match.group(1)
                value = statement_match.group(2).strip()
                
                # Format the statement with proper spacing
                formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{key}: {value};')
                i += 1
                continue
            
            # Handle assignments
            assignment_match = ASSIGNMENT.match(stripped)
            if assignment_match:
                var = assignment_match.group(1)
                value = assignment_match.group(2).strip()
                
                # Format the assignment with proper spacing
                formatted_lines.append(' ' * (indent_level * self.indent_size) + f'{var} = {value};')
                i += 1
                continue
            
            # Default case: preserve the line but apply indentation
            formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
            i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_param_block(self, param_defs: List[Tuple[re.Match, int]], parent_indent: int) -> List[str]:
        """
        Format a block of parameter definitions with aligned equals signs.
        
        Args:
            param_defs: List of (match, indent_level) tuples for param definitions
            parent_indent: Indentation level of the parent block
            
        Returns:
            List of formatted parameter definition lines
        """
        if not param_defs:
            return []
        
        # Compute the maximum length of parameter names and types for alignment
        max_name_len = 0
        max_type_unit_len = 0
        
        for match, _ in param_defs:
            # Handle both PARAM_DEF and ASSIGNMENT patterns
            if len(match.groups()) >= 4:  # PARAM_DEF pattern
                name = match.group(1)
                type_name = match.group(2) or ""
                unit = match.group(3) or ""
                
                # Compute length with type and unit if present
                name_len = len(name)
                type_unit_len = len(f": {type_name}") if type_name else 0
                type_unit_len += len(f"[{unit}]") if unit else 0
                
                max_name_len = max(max_name_len, name_len)
                max_type_unit_len = max(max_type_unit_len, type_unit_len)
            else:  # ASSIGNMENT pattern
                name = match.group(1)
                max_name_len = max(max_name_len, len(name))
        
        # Format each parameter with alignment
        formatted_params = []
        for match, indent_level in param_defs:
            indent = ' ' * (indent_level * self.indent_size)
            
            if len(match.groups()) >= 4:  # PARAM_DEF pattern
                name = match.group(1)
                type_name = match.group(2) or ""
                unit = match.group(3) or ""
                value = match.group(4).strip()
                
                # Build the parameter string with proper alignment
                param_str = name
                if type_name or unit:
                    param_str = param_str.ljust(max_name_len)
                    
                    if type_name:
                        param_str += f": {type_name}"
                    
                    if unit:
                        param_str += f"[{unit}]"
                
                # If the parameter string is already too long, don't try to align the equals
                if len(indent + param_str) + 1 >= self.max_width:
                    formatted_params.append(f"{indent}{param_str} = {value};")
                else:
                    # Align the equals sign
                    if type_name or unit:
                        padding = max_type_unit_len - (len(f": {type_name}") if type_name else 0) - (len(f"[{unit}]") if unit else 0)
                        param_str += ' ' * padding
                    else:
                        param_str = param_str.ljust(max_name_len)
                    
                    formatted_params.append(f"{indent}{param_str} = {value};")
            else:  # ASSIGNMENT pattern
                name = match.group(1)
                value = match.group(2).strip()
                
                # Align the equals sign
                param_str = name.ljust(max_name_len)
                formatted_params.append(f"{indent}{param_str} = {value};")
        
        return formatted_params
    
    def write_formatted_file(self, file_path: Path) -> bool:
        """
        Format a file and write the result back to disk.
        
        Args:
            file_path: Path to the file to format
            
        Returns:
            True if the file was changed, False otherwise
        """
        original_content = None
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        formatted_content = self.format_string(original_content)
        
        # Check if content changed
        if original_content == formatted_content:
            return False
        
        # Write formatted content
        with open(file_path, 'w') as f:
            f.write(formatted_content)
        
        return True


def format_files(files: List[Path], indent_size: int = 2) -> Tuple[int, int]:
    """
    Format multiple ELFIN files.
    
    Args:
        files: List of file paths to format
        indent_size: Number of spaces per indentation level
        
    Returns:
        Tuple of (number of files processed, number of files changed)
    """
    formatter = ELFINFormatter(indent_size)
    
    processed = 0
    changed = 0
    
    for file in files:
        processed += 1
        try:
            if formatter.write_formatted_file(file):
                changed += 1
                print(f"Formatted: {file}")
            else:
                print(f"Already formatted: {file}")
        except Exception as e:
            print(f"Error formatting {file}: {e}", file=sys.stderr)
    
    return processed, changed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Format ELFIN code files")
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="Files to format"
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="Number of spaces per indentation level (default: 2)"
    )
    
    args = parser.parse_args()
    
    processed, changed = format_files(args.files, args.indent)
    
    print(f"Processed {processed} files, formatted {changed} files")


if __name__ == "__main__":
    main()
