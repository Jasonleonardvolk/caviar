"""
ELFIN Parser Module.

This module provides parsing capabilities for the ELFIN DSL.
"""

from alan_backend.elfin.parser.lexer import Token, TokenType, tokenize
from alan_backend.elfin.parser.parser import Parser, ParseError, parse_elfin
from alan_backend.elfin.parser.module_ast import (
    ImportDecl, TemplateParamDecl, TemplateDecl, TemplateArgument,
    TemplateInstantiation, ModuleNode
)
from alan_backend.elfin.parser.module_parser import ModuleAwareParser, parse_elfin_module

__all__ = [
    'Token', 'TokenType', 'tokenize',
    'Parser', 'ParseError', 'parse_elfin',
    'ImportDecl', 'TemplateParamDecl', 'TemplateDecl', 
    'TemplateArgument', 'TemplateInstantiation', 'ModuleNode',
    'ModuleAwareParser', 'parse_elfin_module'
]
