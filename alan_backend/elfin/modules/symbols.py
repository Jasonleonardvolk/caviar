"""
ELFIN Symbol Table

This module provides the symbol table implementation for ELFIN, which is responsible
for tracking symbols and their scopes. It's a key component for name resolution
in the module system.
"""

from typing import Dict, List, Optional, Any, Set


class Symbol:
    """
    Represents a symbol in the ELFIN language.
    
    A symbol can be a variable, function, system, template, etc. It has a name,
    a type, a value (optional), and a source (where it was defined).
    """
    
    def __init__(self, name: str, symbol_type: str, value: Optional[Any] = None,
                 source: Optional[str] = None):
        """
        Initialize a symbol.
        
        Args:
            name: The name of the symbol
            symbol_type: The type of the symbol (e.g., 'variable', 'function', 'template')
            value: The value of the symbol (optional)
            source: The source of the symbol (e.g., module path) (optional)
        """
        self.name = name
        self.symbol_type = symbol_type
        self.value = value
        self.source = source
    
    def __str__(self) -> str:
        """String representation of the symbol."""
        if self.source:
            return f"{self.name} ({self.symbol_type} from {self.source})"
        else:
            return f"{self.name} ({self.symbol_type})"


class Scope:
    """
    Represents a scope in the ELFIN language.
    
    A scope contains symbols defined in that scope and can have a parent scope.
    Symbols are looked up first in the current scope, then in the parent scope.
    """
    
    def __init__(self, name: str, parent: Optional['Scope'] = None):
        """
        Initialize a scope.
        
        Args:
            name: The name of the scope
            parent: The parent scope (optional)
        """
        self.name = name
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.children: List['Scope'] = []
        
        # If we have a parent, add this scope as a child
        if parent:
            parent.children.append(self)
    
    def define(self, symbol: Symbol) -> Symbol:
        """
        Define a symbol in this scope.
        
        Args:
            symbol: The symbol to define
            
        Returns:
            The defined symbol
            
        Raises:
            ValueError: If the symbol is already defined in this scope
        """
        if symbol.name in self.symbols:
            raise ValueError(f"Symbol '{symbol.name}' already defined in scope '{self.name}'")
        
        self.symbols[symbol.name] = symbol
        return symbol
    
    def lookup(self, name: str, local_only: bool = False) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        
        Args:
            name: The name of the symbol to look up
            local_only: Whether to only look in the current scope
            
        Returns:
            The symbol if found, None otherwise
        """
        # Check in the current scope
        if name in self.symbols:
            return self.symbols[name]
        
        # If local_only, don't look in parent scopes
        if local_only or not self.parent:
            return None
        
        # Check in parent scopes
        return self.parent.lookup(name)
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """
        Get all symbols defined in this scope and parent scopes.
        
        Returns:
            A dictionary of symbol names to symbols
        """
        symbols = {}
        
        # Add symbols from parent scopes first (so we can override them)
        if self.parent:
            symbols.update(self.parent.get_all_symbols())
        
        # Add symbols from this scope
        symbols.update(self.symbols)
        
        return symbols
    
    def get_local_symbols(self) -> Dict[str, Symbol]:
        """
        Get symbols defined in this scope only.
        
        Returns:
            A dictionary of symbol names to symbols
        """
        return self.symbols.copy()
    
    def create_child(self, name: str) -> 'Scope':
        """
        Create a child scope.
        
        Args:
            name: The name of the child scope
            
        Returns:
            The child scope
        """
        return Scope(name, self)


class SymbolTable:
    """
    A symbol table for ELFIN.
    
    The symbol table keeps track of all symbols defined in a module or system
    and provides methods for defining and looking up symbols.
    """
    
    def __init__(self):
        """Initialize a symbol table with a global scope."""
        self.global_scope = Scope("global")
        self.current_scope = self.global_scope
        
        # Track imported modules and their symbols
        self.imported_modules: Dict[str, Dict[str, Symbol]] = {}
    
    def enter_scope(self, name: str) -> Scope:
        """
        Enter a new scope.
        
        Args:
            name: The name of the new scope
            
        Returns:
            The new scope
        """
        self.current_scope = self.current_scope.create_child(name)
        return self.current_scope
    
    def exit_scope(self) -> Scope:
        """
        Exit the current scope and return to the parent scope.
        
        Returns:
            The new current scope (parent of the previous scope)
            
        Raises:
            ValueError: If there is no parent scope
        """
        if not self.current_scope.parent:
            raise ValueError("Cannot exit global scope")
        
        self.current_scope = self.current_scope.parent
        return self.current_scope
    
    def define(self, name: str, symbol_type: str, value: Optional[Any] = None,
               source: Optional[str] = None) -> Symbol:
        """
        Define a symbol in the current scope.
        
        Args:
            name: The name of the symbol
            symbol_type: The type of the symbol
            value: The value of the symbol (optional)
            source: The source of the symbol (optional)
            
        Returns:
            The defined symbol
        """
        symbol = Symbol(name, symbol_type, value, source)
        return self.current_scope.define(symbol)
    
    def lookup(self, name: str, local_only: bool = False) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        
        Args:
            name: The name of the symbol
            local_only: Whether to only look in the current scope
            
        Returns:
            The symbol if found, None otherwise
        """
        return self.current_scope.lookup(name, local_only)
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """
        Get all symbols in the current scope and parent scopes.
        
        Returns:
            A dictionary of symbol names to symbols
        """
        return self.current_scope.get_all_symbols()
    
    def get_local_symbols(self) -> Dict[str, Symbol]:
        """
        Get symbols in the current scope only.
        
        Returns:
            A dictionary of symbol names to symbols
        """
        return self.current_scope.get_local_symbols()
    
    def import_module(self, module_path: str, module_symbols: Dict[str, Any],
                      imports: List[Dict[str, str]]) -> None:
        """
        Import symbols from a module.
        
        Args:
            module_path: The path to the module
            module_symbols: The symbols defined in the module
            imports: A list of dictionaries with 'name' and 'alias' keys
        """
        # Track the module and its symbols
        self.imported_modules[module_path] = {}
        
        # Import the specified symbols
        for import_spec in imports:
            name = import_spec['name']
            alias = import_spec['alias'] or name
            
            if name in module_symbols:
                # Create a symbol in the current scope that refers to the imported symbol
                symbol = module_symbols[name]
                imported_symbol = Symbol(
                    name=alias,
                    symbol_type=symbol.symbol_type,
                    value=symbol.value,
                    source=module_path
                )
                self.current_scope.define(imported_symbol)
                
                # Track the imported symbol
                self.imported_modules[module_path][name] = imported_symbol
            else:
                raise ValueError(f"Symbol '{name}' not found in module '{module_path}'")
    
    def get_imported_modules(self) -> Dict[str, Dict[str, Symbol]]:
        """
        Get all imported modules and their symbols.
        
        Returns:
            A dictionary of module paths to dictionaries of symbol names to symbols
        """
        return self.imported_modules.copy()
