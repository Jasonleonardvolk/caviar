"""
Grammar module for the ELFIN language compiler.

This module defines the grammar for the ELFIN language and implements
the parser that converts ELFIN text into an Abstract Syntax Tree (AST).

We use the Lark parser generator to implement the parser, which provides:
- A clean PEG grammar definition
- Good error reporting
- Efficient parsing
"""

__all__ = ['parse', 'ELFINSyntaxError']
