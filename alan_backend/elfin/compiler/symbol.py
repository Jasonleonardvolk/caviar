"""
Symbol representation for ELFIN language.

This module defines the Symbol class that represents named entities in ELFIN
programs, including their dimensions for dimensional analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from alan_backend.elfin.units.unit_expr import UnitExpr


@dataclass
class Symbol:
    """
    Represents a named symbol in ELFIN code with semantic information.
    
    Symbols include variables, parameters, function names, and other named
    entities in the ELFIN language. Each symbol can have an associated
    dimension for dimensional analysis.
    """
    name: str
    
    # Dimension information for dimensional analysis
    dim: Optional[UnitExpr] = None
    
    # Symbol type information
    type_name: Optional[str] = None
    
    # Scope information
    scope: Optional[str] = None
    
    # Source location information
    line: Optional[int] = None
    column: Optional[int] = None
    
    # Reference information
    references: List[Dict[str, Any]] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation of the symbol."""
        dim_str = f" [{self.dim}]" if self.dim else ""
        type_str = f": {self.type_name}" if self.type_name else ""
        return f"{self.name}{type_str}{dim_str}"
    
    def add_reference(self, line: int, column: int) -> None:
        """
        Add a reference to this symbol.
        
        Args:
            line: Line number of the reference
            column: Column number of the reference
        """
        self.references.append({"line": line, "column": column})


class SymbolTable:
    """
    Table of symbols in an ELFIN program.
    
    The symbol table keeps track of all named entities in an ELFIN program
    and their properties, including dimensions for dimensional analysis.
    """
    
    def __init__(self):
        """Initialize an empty symbol table."""
        self.symbols: Dict[str, Symbol] = {}
        self.scopes: Dict[str, List[str]] = {"global": []}
        self.current_scope = "global"
    
    def add_symbol(self, symbol: Symbol) -> None:
        """
        Add a symbol to the table.
        
        Args:
            symbol: Symbol to add
        """
        if not symbol.scope:
            symbol.scope = self.current_scope
        
        scope = symbol.scope
        if scope not in self.scopes:
            self.scopes[scope] = []
        
        self.symbols[symbol.name] = symbol
        self.scopes[scope].append(symbol.name)
    
    def get_symbol(self, name: str) -> Optional[Symbol]:
        """
        Get a symbol by name.
        
        Args:
            name: Name of the symbol
            
        Returns:
            Symbol with the given name, or None if not found
        """
        return self.symbols.get(name)
    
    def set_dimension(self, name: str, dim: UnitExpr) -> None:
        """
        Set the dimension of a symbol.
        
        Args:
            name: Name of the symbol
            dim: Dimension to set
        """
        symbol = self.get_symbol(name)
        if symbol:
            symbol.dim = dim
    
    def get_symbols_in_scope(self, scope: str) -> List[Symbol]:
        """
        Get all symbols in a given scope.
        
        Args:
            scope: Name of the scope
            
        Returns:
            List of symbols in the scope
        """
        if scope not in self.scopes:
            return []
        
        return [self.symbols[name] for name in self.scopes[scope]]
    
    def enter_scope(self, scope: str) -> None:
        """
        Enter a new scope.
        
        Args:
            scope: Name of the scope to enter
        """
        if scope not in self.scopes:
            self.scopes[scope] = []
        
        self.current_scope = scope
    
    def exit_scope(self) -> None:
        """Exit the current scope and return to the global scope."""
        self.current_scope = "global"
