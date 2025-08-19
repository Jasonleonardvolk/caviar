"""
ELFIN Compiler Package - Kia Magic Compiler

This package contains all modules for the ELFIN language compiler,
which transforms ELFIN specification files into executable code.

The compiler is organized into several sub-packages:
- grammar: Grammar definition and parser implementation
- ast: Abstract Syntax Tree nodes and visitors
- codegen: Code generation for target languages
- tests: Test cases and fixtures

The main compilation pipeline is:
1. Parse ELFIN text into an AST
2. Transform and validate the AST
3. Generate code for the target language (Python/C++)
"""

__version__ = "0.1.0"
