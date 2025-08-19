"""
Code Generation module for the ELFIN language compiler.

This module is responsible for generating executable code in target languages
(Python, C++) from the AST representation of an ELFIN program.

Key components:
- CodeGenerator: Base class for all code generators
- PythonGenerator: Generates Python code for simulation
- TemplateManager: Manages code templates for different targets
"""

__all__ = ['CodeGenerator', 'PythonGenerator', 'generate_code']
