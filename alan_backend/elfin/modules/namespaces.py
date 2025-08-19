"""
ELFIN Namespaces.

This module provides namespace support for ELFIN modules, allowing for
better organization and avoiding name conflicts.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from alan_backend.elfin.modules.symbols import Symbol, SymbolTable


class Namespace:
    """
    An ELFIN namespace.
    
    A namespace is a container for symbols that provides isolation
    and organization of related components.
    """
    
    def __init__(self, name: str, parent: Optional['Namespace'] = None):
        """
        Initialize a namespace.
        
        Args:
            name: The name of the namespace
            parent: The parent namespace (optional)
        """
        self.name = name
        self.parent = parent
        self.symbol_table = SymbolTable()
        self.children: Dict[str, 'Namespace'] = {}
        
        # Full path includes the parent's path
        if parent is None:
            self.path = name
        else:
            if parent.path:
                self.path = f"{parent.path}::{name}"
            else:
                self.path = name
    
    def create_child(self, name: str) -> 'Namespace':
        """
        Create a child namespace.
        
        Args:
            name: The name of the child namespace
            
        Returns:
            The child namespace
            
        Raises:
            ValueError: If a child with the same name already exists
        """
        if name in self.children:
            raise ValueError(f"Child namespace '{name}' already exists")
        
        child = Namespace(name, self)
        self.children[name] = child
        return child
    
    def get_child(self, name: str) -> Optional['Namespace']:
        """
        Get a child namespace by name.
        
        Args:
            name: The name of the child namespace
            
        Returns:
            The child namespace if found, None otherwise
        """
        return self.children.get(name)
    
    def define(self, name: str, symbol_type: str, value: Any = None, source: Optional[str] = None) -> Symbol:
        """
        Define a symbol in this namespace.
        
        Args:
            name: The name of the symbol
            symbol_type: The type of the symbol
            value: The value of the symbol (optional)
            source: The source of the symbol (optional)
            
        Returns:
            The defined symbol
            
        Raises:
            ValueError: If the symbol is already defined in this namespace
        """
        return self.symbol_table.define(name, symbol_type, value, source)
    
    def lookup(self, name: str, recursive: bool = True) -> Optional[Symbol]:
        """
        Look up a symbol by name.
        
        Args:
            name: The name of the symbol
            recursive: Whether to recursively look up in parent namespaces
            
        Returns:
            The symbol if found, None otherwise
        """
        # Try to find in this namespace
        symbol = self.symbol_table.lookup(name)
        if symbol is not None:
            return symbol
        
        # If not found and recursive, try parent
        if recursive and self.parent is not None:
            return self.parent.lookup(name, recursive)
        
        return None
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """
        Get all symbols in this namespace.
        
        Returns:
            A dictionary of symbol names to symbols
        """
        return self.symbol_table.get_all_symbols()
    
    def __str__(self) -> str:
        """Get a string representation of the namespace."""
        return f"Namespace({self.path})"
    
    def __repr__(self) -> str:
        """Get a detailed string representation of the namespace."""
        return f"Namespace({self.path}, symbols={len(self.get_all_symbols())}, children={len(self.children)})"


class NamespaceRegistry:
    """
    A registry of namespaces.
    
    This registry manages the hierarchy of namespaces in an ELFIN program.
    """
    
    def __init__(self):
        """Initialize an empty namespace registry."""
        # Global namespace has an empty name
        self.global_namespace = Namespace("")
        
        # Current namespace for symbol definitions
        self.current_namespace = self.global_namespace
    
    def create_namespace(self, path: str) -> Namespace:
        """
        Create a namespace at the specified path.
        
        Args:
            path: The path to the namespace (e.g., "std::math")
            
        Returns:
            The created namespace
            
        Raises:
            ValueError: If a namespace component already exists
        """
        if not path:
            return self.global_namespace
        
        # Split the path into components
        components = path.split("::")
        
        # Start at the global namespace
        current = self.global_namespace
        
        # Create or get each component
        for component in components:
            child = current.get_child(component)
            if child is None:
                child = current.create_child(component)
            current = child
        
        return current
    
    def get_namespace(self, path: str) -> Optional[Namespace]:
        """
        Get a namespace by path.
        
        Args:
            path: The path to the namespace (e.g., "std::math")
            
        Returns:
            The namespace if found, None otherwise
        """
        if not path:
            return self.global_namespace
        
        # Split the path into components
        components = path.split("::")
        
        # Start at the global namespace
        current = self.global_namespace
        
        # Navigate the namespace hierarchy
        for component in components:
            child = current.get_child(component)
            if child is None:
                return None
            current = child
        
        return current
    
    def set_current_namespace(self, path: str) -> Namespace:
        """
        Set the current namespace.
        
        Args:
            path: The path to the namespace (e.g., "std::math")
            
        Returns:
            The current namespace
            
        Raises:
            ValueError: If the namespace does not exist
        """
        namespace = self.get_namespace(path)
        if namespace is None:
            raise ValueError(f"Namespace '{path}' not found")
        
        self.current_namespace = namespace
        return namespace
    
    def resolve_symbol(self, name: str) -> Optional[Tuple[Symbol, Namespace]]:
        """
        Resolve a symbol by name.
        
        This method searches for a symbol starting from the current namespace
        and moving up the namespace hierarchy.
        
        Args:
            name: The name of the symbol
            
        Returns:
            A tuple of (symbol, namespace) if found, None otherwise
        """
        # Check if it's a qualified name
        if "::" in name:
            # Split into namespace path and symbol name
            path, symbol_name = name.rsplit("::", 1)
            
            # Get the namespace
            namespace = self.get_namespace(path)
            if namespace is None:
                return None
            
            # Look up the symbol in that namespace
            symbol = namespace.lookup(symbol_name, recursive=False)
            if symbol is not None:
                return (symbol, namespace)
            
            return None
        
        # Unqualified name, start from current namespace
        namespace = self.current_namespace
        while namespace is not None:
            symbol = namespace.lookup(name, recursive=False)
            if symbol is not None:
                return (symbol, namespace)
            namespace = namespace.parent
        
        return None
    
    def define_symbol(self, name: str, symbol_type: str, value: Any = None, source: Optional[str] = None) -> Symbol:
        """
        Define a symbol in the current namespace.
        
        Args:
            name: The name of the symbol
            symbol_type: The type of the symbol
            value: The value of the symbol (optional)
            source: The source of the symbol (optional)
            
        Returns:
            The defined symbol
        """
        return self.current_namespace.define(name, symbol_type, value, source)
    
    def import_symbols(self, namespace_path: str, symbols: List[Dict[str, str]]) -> None:
        """
        Import symbols from a namespace.
        
        Args:
            namespace_path: The path to the source namespace
            symbols: List of dictionaries with 'name' and 'alias' keys
        """
        # Get the source namespace
        source_namespace = self.get_namespace(namespace_path)
        if source_namespace is None:
            raise ValueError(f"Namespace '{namespace_path}' not found")
        
        # Import each symbol
        for symbol_info in symbols:
            name = symbol_info["name"]
            alias = symbol_info["alias"] or name
            
            # Look up the symbol in the source namespace
            symbol = source_namespace.lookup(name, recursive=False)
            if symbol is None:
                raise ValueError(f"Symbol '{name}' not found in namespace '{namespace_path}'")
            
            # Define the symbol in the current namespace with the alias
            self.current_namespace.define(
                name=alias,
                symbol_type=symbol.symbol_type,
                value=symbol.value,
                source=symbol.source
            )
