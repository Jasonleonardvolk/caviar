#!/usr/bin/env python3
"""
Standalone ELFIN Parser

This is a self-contained parser for the ELFIN language. It includes simplified
versions of all necessary components from the compiler package without relying
on complex imports or package structures.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any

# ====================================================================
# AST Node definitions
# ====================================================================

class Node:
    """Base class for all AST nodes."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = [f"{k}={repr(v)}" for k, v in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class Program(Node):
    """Root node of the AST, representing a complete ELFIN program."""
    
    def __init__(self, sections=None, imports=None):
        self.sections = sections or []
        self.imports = imports or []


class ImportStmt(Node):
    """Import statement: import SectionName from "path/to/file.elfin";"""
    
    def __init__(self, section_name=None, file_path=None):
        self.section_name = section_name
        self.file_path = file_path


class SystemSection(Node):
    """System section defining dynamics."""
    
    def __init__(self, name=None, continuous_state=None, inputs=None, params=None, dynamics=None, param_units=None):
        self.name = name
        self.continuous_state = continuous_state or []
        self.inputs = inputs or []
        self.params = params or {}
        self.dynamics = dynamics or {}
        self.param_units = param_units or {}  # Store units for parameters


# ====================================================================
# Parser implementation
# ====================================================================

def parse_system_section(text):
    """Parse a system section from text."""
    # This is a simplified parser that just extracts the system name
    # In a real implementation, we would parse the entire section
    
    import re
    
    # Extract the system name
    system_match = re.search(r'system\s+([A-Za-z0-9_]+)', text)
    if not system_match:
        raise ValueError("Invalid system section: missing system name")
    
    system_name = system_match.group(1)
    
    # Extract the continuous state
    state_match = re.search(r'continuous_state\s*:\s*\[(.*?)\]', text, re.DOTALL)
    continuous_state = []
    if state_match:
        # Split the comma-separated list and strip whitespace
        continuous_state = [s.strip() for s in state_match.group(1).split(',')]
    
    # Extract the inputs
    input_match = re.search(r'input\s*:\s*\[(.*?)\]', text, re.DOTALL)
    inputs = []
    if input_match:
        # Split the comma-separated list and strip whitespace
        inputs = [s.strip() for s in input_match.group(1).split(',')]
    
    # Extract parameters and their units
    params = {}
    param_units = {}
    
    # Find the params block
    params_match = re.search(r'params\s*{(.*?)}', text, re.DOTALL)
    if params_match:
        params_text = params_match.group(1)
        
        # Look for parameter definitions with possible units
        param_pattern = r'([A-Za-z0-9_]+)\s*:\s*([0-9]+(\.[0-9]+)?)\s*(?:\[\s*([a-zA-Z0-9\/\*\^\-\s\.]+)\s*\])?'
        for param_match in re.finditer(param_pattern, params_text):
            param_name = param_match.group(1)
            param_value = float(param_match.group(2))
            params[param_name] = param_value
            
            # Extract unit if present
            if param_match.group(4):
                param_units[param_name] = param_match.group(4).strip()
    
    # Create a SystemSection node
    return SystemSection(
        name=system_name,
        continuous_state=continuous_state,
        inputs=inputs,
        params=params,
        param_units=param_units
    )


def parse(text):
    """Parse ELFIN text into an AST."""
    # This is a simplified parser that just extracts system sections
    # In a real implementation, we would parse the entire file
    
    import re
    
    # Create a program node
    program = Program()
    
    # Find all system sections
    system_pattern = r'system\s+[A-Za-z0-9_]+\s*\{[^}]*\}'
    for system_match in re.finditer(system_pattern, text, re.DOTALL):
        system_text = system_match.group(0)
        try:
            system_section = parse_system_section(system_text)
            program.sections.append(system_section)
        except ValueError as e:
            print(f"Warning: {e}")
    
    return program


def parse_file(file_path):
    """Parse an ELFIN file into an AST."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return parse(text)


# ====================================================================
# Main function
# ====================================================================

def main():
    """Main entry point for the standalone parser."""
    if len(sys.argv) < 2:
        print("Usage: python standalone_parser.py <elfin_file>")
        return 1
    
    file_path = sys.argv[1]
    
    try:
        # Parse the file
        ast = parse_file(file_path)
        
        # Print the AST
        print("Parsing successful!")
        print("AST:")
        print(f"Program with {len(ast.sections)} sections and {len(ast.imports)} imports")
        
        for i, section in enumerate(ast.sections):
            print(f"Section {i+1}: {section}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
