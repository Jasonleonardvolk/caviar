"""
ELFIN Parser Implementation

This module implements the parser for the ELFIN language using the Lark parser
generator. It parses ELFIN files into an Abstract Syntax Tree (AST).
"""

import os
import sys
from pathlib import Path
from lark import Lark, Transformer as LarkTransformer, v_args
from lark.exceptions import LarkError

# Import our Megatron AST converter
# Using direct imports to avoid relative import issues when imported directly
import sys
import os
from pathlib import Path

# Get the path to the ast directory
ast_dir = Path(__file__).parent.parent / "ast"
megatron_path = ast_dir / "megatron.py"

# Import using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("megatron", megatron_path)
megatron_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(megatron_module)
ELFINMegatron = megatron_module.ELFINMegatron

# Custom exception for syntax errors
class ELFINSyntaxError(Exception):
    """Exception raised for syntax errors in ELFIN code."""
    
    def __init__(self, message, line=None, column=None, filename=None):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        
        location = ""
        if filename:
            location += f"in {filename} "
        if line is not None:
            location += f"at line {line}"
            if column is not None:
                location += f", column {column}"
        
        super().__init__(f"Syntax error {location}: {message}")

# Get the grammar file path
_grammar_file = Path(__file__).parent / "elfin.lark"

# Load the grammar from the file
with open(_grammar_file, "r") as f:
    _grammar = f.read()

# Create the parser with the grammar
_parser = Lark(_grammar, start="start", parser="earley")

def parse(text, filename=None):
    """
    Parse ELFIN text into an AST.
    
    Args:
        text: The ELFIN code to parse
        filename: Optional filename for error reporting
        
    Returns:
        The parse tree
        
    Raises:
        ELFINSyntaxError: If the text contains syntax errors
    """
    try:
        # Parse the text with Lark
        parse_tree = _parser.parse(text)
        
        # Transform the parse tree into our AST using the Megatron
        megatron = ELFINMegatron(filename)
        ast = megatron.transform(parse_tree)
        
        return ast
    except LarkError as e:
        # Convert Lark errors to our own error type
        line = getattr(e, 'line', None)
        column = getattr(e, 'column', None)
        
        raise ELFINSyntaxError(str(e), line, column, filename) from e

def parse_file(file_path):
    """
    Parse an ELFIN file into an AST.
    
    Args:
        file_path: Path to the ELFIN file
        
    Returns:
        The parse tree
        
    Raises:
        ELFINSyntaxError: If the file contains syntax errors
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        return parse(text, str(file_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"ELFIN file not found: {file_path}")

if __name__ == "__main__":
    # Simple command-line interface for testing
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        tree = parse_file(file_path)
        print("Parsing successful!")
        print(tree.pretty())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
