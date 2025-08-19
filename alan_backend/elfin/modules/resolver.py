"""
ELFIN Module Resolver

This module provides the import resolution system for ELFIN, which is responsible for
finding, loading, and caching modules imported by ELFIN files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from alan_backend.elfin.modules.errors import (
    ModuleNotFoundError,
    CircularDependencyError,
    ModuleParseError
)


class ModuleSearchPath:
    """
    Represents a search path for ELFIN modules.
    
    Search paths can be absolute or relative to a base directory.
    """
    
    def __init__(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize a module search path.
        
        Args:
            path: The search path (absolute or relative to base_dir)
            base_dir: The base directory for relative paths
        """
        self.path = Path(path)
        self.base_dir = Path(base_dir) if base_dir else None
        
        # Resolve path if it's relative and we have a base directory
        if not self.path.is_absolute() and self.base_dir:
            self.resolved_path = (self.base_dir / self.path).resolve()
        else:
            self.resolved_path = self.path.resolve()
    
    def resolve(self, module_path: Union[str, Path]) -> Optional[Path]:
        """
        Resolve a module path within this search path.
        
        Args:
            module_path: The module path to resolve
            
        Returns:
            The absolute path to the module if found, None otherwise
        """
        # Convert to Path object if it's a string
        module_path = Path(module_path)
        
        # Check if the module exists directly
        full_path = (self.resolved_path / module_path).resolve()
        if full_path.exists():
            return full_path
        
        # Check if the module exists with .elfin extension
        if not module_path.suffix:
            full_path_with_ext = (self.resolved_path / f"{module_path}.elfin").resolve()
            if full_path_with_ext.exists():
                return full_path_with_ext
        
        return None
    
    def __str__(self) -> str:
        return str(self.resolved_path)


class ModuleCache:
    """
    Cache for parsed ELFIN modules.
    
    This class caches parsed modules to avoid redundant parsing.
    """
    
    def __init__(self):
        """Initialize an empty module cache."""
        self._cache: Dict[Path, Any] = {}
    
    def get(self, path: Path) -> Optional[Any]:
        """
        Get a module from the cache.
        
        Args:
            path: The absolute path to the module
            
        Returns:
            The cached module if found, None otherwise
        """
        return self._cache.get(path.resolve())
    
    def put(self, path: Path, module: Any) -> None:
        """
        Put a module in the cache.
        
        Args:
            path: The absolute path to the module
            module: The parsed module
        """
        self._cache[path.resolve()] = module
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class ImportResolver:
    """
    Resolves ELFIN module imports.
    
    This class is responsible for finding, loading, and caching modules imported
    by ELFIN files. It handles search paths, module resolution, and circular
    dependency detection.
    """
    
    def __init__(self, search_paths: Optional[List[Union[str, Path]]] = None,
                 base_dir: Optional[Union[str, Path]] = None,
                 parser: Optional[Any] = None):
        """
        Initialize an import resolver.
        
        Args:
            search_paths: List of search paths for modules
            base_dir: Base directory for relative search paths
            parser: Parser to use for parsing modules
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.search_paths = [
            ModuleSearchPath(path, self.base_dir)
            for path in (search_paths or ['.'])
        ]
        self.cache = ModuleCache()
        self.parser = parser
        
        # Add standard library paths if they exist
        lib_path = self.base_dir / 'lib'
        if lib_path.exists() and lib_path.is_dir():
            self.search_paths.append(ModuleSearchPath(lib_path))
    
    def resolve(self, module_path: Union[str, Path],
                from_module: Optional[Path] = None) -> Tuple[Path, Any]:
        """
        Resolve a module import.
        
        Args:
            module_path: The module path to resolve
            from_module: The module that is importing this module
            
        Returns:
            A tuple of (resolved_path, parsed_module)
            
        Raises:
            ModuleNotFoundError: If the module cannot be found
            CircularDependencyError: If a circular dependency is detected
            ModuleParseError: If the module cannot be parsed
        """
        # Convert to Path object if it's a string
        module_path = Path(module_path)
        
        # Handle relative imports
        if from_module and not self._is_absolute_import(module_path):
            from_dir = from_module.parent
            absolute_path = self._resolve_relative_import(module_path, from_dir)
            if absolute_path:
                return self._load_module(absolute_path, from_module)
        
        # Search for the module in all search paths
        for search_path in self.search_paths:
            resolved_path = search_path.resolve(module_path)
            if resolved_path:
                return self._load_module(resolved_path, from_module)
        
        # Module not found
        raise ModuleNotFoundError(str(module_path), [str(p) for p in self.search_paths])
    
    def _is_absolute_import(self, module_path: Path) -> bool:
        """
        Check if an import is absolute.
        
        Args:
            module_path: The module path to check
            
        Returns:
            True if the import is absolute, False otherwise
        """
        # TODO: Define what makes an import absolute in ELFIN
        # For now, we'll just check if it starts with a slash or has a colon (Windows drive letter)
        path_str = str(module_path)
        return path_str.startswith('/') or ':' in path_str
    
    def _resolve_relative_import(self, module_path: Path, from_dir: Path) -> Optional[Path]:
        """
        Resolve a relative import.
        
        Args:
            module_path: The relative module path
            from_dir: The directory to resolve from
            
        Returns:
            The absolute path to the module if found, None otherwise
        """
        # Try direct path
        full_path = (from_dir / module_path).resolve()
        if full_path.exists():
            return full_path
        
        # Try with .elfin extension
        if not module_path.suffix:
            full_path_with_ext = (from_dir / f"{module_path}.elfin").resolve()
            if full_path_with_ext.exists():
                return full_path_with_ext
        
        return None
    
    def _load_module(self, path: Path, from_module: Optional[Path] = None) -> Tuple[Path, Any]:
        """
        Load a module from a path.
        
        Args:
            path: The path to the module
            from_module: The module that is importing this module
            
        Returns:
            A tuple of (path, parsed_module)
            
        Raises:
            CircularDependencyError: If a circular dependency is detected
            ModuleParseError: If the module cannot be parsed
        """
        # Check for circular dependencies
        import_chain = self._get_import_chain(path, from_module)
        if path in import_chain:
            raise CircularDependencyError(
                [str(p) for p in import_chain] + [str(path)]
            )
        
        # Check cache first
        cached_module = self.cache.get(path)
        if cached_module:
            return path, cached_module
        
        # Load and parse the module
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # TODO: Replace this with the actual parser when available
            if self.parser:
                parsed_module = self.parser.parse(content, path=path, resolver=self)
            else:
                # Placeholder for testing
                parsed_module = {
                    "path": str(path),
                    "content": content,
                    "imports": [],  # Placeholder for detected imports
                }
            
            # Cache the module
            self.cache.put(path, parsed_module)
            
            return path, parsed_module
        
        except Exception as e:
            raise ModuleParseError(path, e)
    
    def _get_import_chain(self, target_path: Path, from_module: Optional[Path]) -> List[Path]:
        """
        Get the import chain from a module to the target module.
        
        Args:
            target_path: The target module path
            from_module: The module that is importing the target module
            
        Returns:
            A list of module paths in the import chain
        """
        if not from_module:
            return []
        
        # TODO: Implement a more sophisticated import chain detection
        # For now, we'll just return the from_module as the only element in the chain
        return [from_module]
    
    def add_search_path(self, path: Union[str, Path]) -> None:
        """
        Add a search path to the resolver.
        
        Args:
            path: The search path to add
        """
        self.search_paths.append(ModuleSearchPath(path, self.base_dir))
    
    def clear_cache(self) -> None:
        """Clear the module cache."""
        self.cache.clear()
