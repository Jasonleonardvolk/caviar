"""
ELFIN Module Compiler.

This module provides integration between the ELFIN module system and the compiler,
enabling compilation of modules with imports and templates.
"""

import os
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

from alan_backend.elfin.parser.module_parser import parse_elfin_module
from alan_backend.elfin.parser.module_ast import (
    ImportDecl, TemplateDecl, TemplateInstantiation, ModuleNode
)
from alan_backend.elfin.modules.resolver import ImportResolver
from alan_backend.elfin.modules.symbols import SymbolTable
from alan_backend.elfin.modules.templates import TemplateRegistry
from alan_backend.elfin.compiler.compiler import compile_elfin
from alan_backend.elfin.compiler.cache import ModuleCache


class ModuleCompiler:
    """
    A compiler for ELFIN modules.
    
    This compiler integrates the module system with the main ELFIN compiler,
    providing support for imports and templates.
    """
    
    def __init__(
        self, 
        search_paths: Optional[List[str]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize a module compiler.
        
        Args:
            search_paths: Paths to search for imported modules
            cache_dir: Directory for caching compiled modules
        """
        self.resolver = ImportResolver(search_paths)
        self.symbol_table = SymbolTable()
        self.template_registry = TemplateRegistry()
        
        # Initialize the module cache
        self.cache = ModuleCache(cache_dir)
        
        # Set of modules currently being compiled (for circular dependency detection)
        self.compiling: Set[str] = set()
        
        # Dependency graph of modules
        self.dependencies: Dict[str, Set[str]] = {}
    
    def compile_file(self, file_path: str, force_recompile: bool = False) -> Any:
        """
        Compile an ELFIN module from a file.
        
        Args:
            file_path: Path to the file to compile
            force_recompile: Whether to force recompilation of cached modules
            
        Returns:
            The compiled module
        """
        # Convert to an absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the module is already in the cache and not forced to recompile
        if not force_recompile:
            compiled_module, metadata = self.cache.get_module(abs_path)
            if compiled_module is not None:
                # Update the dependency graph with cached information
                if "dependencies" in metadata:
                    self.dependencies[abs_path] = metadata["dependencies"]
                return compiled_module
        
        # Check for circular dependencies
        if abs_path in self.compiling:
            raise ValueError(f"Circular dependency detected: {abs_path}")
        
        # Mark as compiling
        self.compiling.add(abs_path)
        
        try:
            # Parse the module
            with open(abs_path, "r") as f:
                source = f.read()
            
            # Parse the module with the module-aware parser
            module_ast = parse_elfin_module(
                source,
                file_path=abs_path,
                resolver=self.resolver,
                symbol_table=self.symbol_table,
                template_registry=self.template_registry
            )
            
            # Track dependencies
            self.dependencies[abs_path] = set()
            
            # Compile imported modules first
            for import_decl in module_ast.imports:
                import_path = self._resolve_import_path(import_decl.source, abs_path)
                if import_path:
                    # Compile the imported module
                    self.compile_file(import_path, force_recompile)
                    
                    # Add to dependencies
                    self.dependencies[abs_path].add(import_path)
            
            # Compile the module AST with the main compiler
            compiled_module = self._compile_module_ast(module_ast)
            
            # Cache the compiled module with its metadata
            metadata = {
                "dependencies": self.dependencies[abs_path],
                "templates": [tmpl.name for tmpl in module_ast.templates],
                "imports": [imp.source for imp in module_ast.imports]
            }
            self.cache.put_module(abs_path, compiled_module, metadata)
            
            return compiled_module
            
        finally:
            # Mark as no longer compiling
            self.compiling.remove(abs_path)
    
    def _resolve_import_path(self, import_source: str, importing_file: str) -> Optional[str]:
        """
        Resolve an import source to an absolute file path.
        
        Args:
            import_source: The import source string
            importing_file: The file that contains the import
            
        Returns:
            The absolute path to the imported file, or None if not found
        """
        try:
            # Use the resolver to find the file
            resolved_path, _ = self.resolver.resolve(
                import_source, 
                from_module=Path(importing_file)
            )
            return str(resolved_path)
        except Exception as e:
            print(f"Error resolving import: {e}")
            return None
    
    def _compile_module_ast(self, module_ast: ModuleNode) -> Any:
        """
        Compile a module AST.
        
        Args:
            module_ast: The module AST to compile
            
        Returns:
            The compiled module
        """
        # Process imports (already done during parsing)
        
        # Process templates
        self._process_templates(module_ast)
        
        # Pass the processed AST to the main compiler
        # TODO: Integrate with the main compiler
        # For now, we'll just return a placeholder
        return {
            "path": module_ast.path,
            "declarations": [decl for decl in module_ast.declarations],
            "imports": [imp.source for imp in module_ast.imports],
            "templates": [tmpl.name for tmpl in module_ast.templates]
        }
    
    def _process_templates(self, module_ast: ModuleNode) -> None:
        """
        Process template declarations and instantiations in a module.
        
        Args:
            module_ast: The module AST to process
        """
        # Templates have already been registered during parsing
        # However, we may need to do additional processing here
        # For example, template instantiation may require code generation
        
        # For now, we'll just perform a placeholder operation
        for template in module_ast.templates:
            if hasattr(template, 'name') and hasattr(template, 'parameters'):
                print(f"Processing template: {template.name} with {len(template.parameters)} parameters")
    
    def get_dependencies(self, file_path: str) -> Set[str]:
        """
        Get the dependencies of a module.
        
        Args:
            file_path: Path to the module
            
        Returns:
            A set of paths to modules that this module depends on
        """
        # Convert to an absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check cache first
        cached_deps = self.cache.get_dependencies(abs_path)
        if cached_deps:
            return cached_deps
        
        # Return from in-memory dependencies if we have them
        if abs_path in self.dependencies:
            return self.dependencies[abs_path]
        
        # If the module hasn't been compiled yet, compile it to get dependencies
        if abs_path not in self.dependencies:
            self.compile_file(abs_path)
        
        # Return dependencies
        return self.dependencies.get(abs_path, set())
    
    def get_dependents(self, file_path: str) -> Set[str]:
        """
        Get the modules that depend on a module.
        
        Args:
            file_path: Path to the module
            
        Returns:
            A set of paths to modules that depend on this module
        """
        # Convert to an absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check cache first
        cached_dependents = self.cache.get_dependents(abs_path)
        if cached_dependents:
            return cached_dependents
        
        # Find modules that depend on this module
        dependents = set()
        for module, deps in self.dependencies.items():
            if abs_path in deps:
                dependents.add(module)
        
        return dependents
    
    def invalidate_cache(self, file_path: str, recursive: bool = True) -> None:
        """
        Invalidate the cache for a module and optionally its dependents.
        
        Args:
            file_path: Path to the module
            recursive: Whether to recursively invalidate dependents
        """
        # Convert to an absolute path
        abs_path = os.path.abspath(file_path)
        
        # Use the cache's invalidation method
        self.cache.invalidate_dependents(abs_path, recursive)
        
        # Also update our in-memory dependency graph
        if abs_path in self.dependencies:
            # Store dependents first if we need to do recursive invalidation
            dependents = self.get_dependents(abs_path) if recursive else set()
            
            # Remove from the dependency graph
            del self.dependencies[abs_path]
            
            # Recursively invalidate dependents
            if recursive:
                for dependent in dependents:
                    self.invalidate_cache(dependent, recursive=True)
    
    def clear_cache(self) -> None:
        """Clear the entire compilation cache."""
        self.cache.clear()
        self.dependencies.clear()
        
    def compile_incremental(self, file_path: str) -> Any:
        """
        Compile a module and only recompile its dependencies if they have changed.
        
        This method provides incremental compilation by checking if modules
        have changed since they were last compiled.
        
        Args:
            file_path: Path to the file to compile
            
        Returns:
            The compiled module
        """
        # Convert to an absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the file is in the cache
        if self.cache.has_module(abs_path):
            # Check if any dependencies have changed
            deps_changed = False
            for dep in self.get_dependencies(abs_path):
                if not self.cache.has_module(dep):
                    deps_changed = True
                    break
            
            # If no dependencies have changed, use the cached module
            if not deps_changed:
                compiled_module, _ = self.cache.get_module(abs_path)
                if compiled_module is not None:
                    return compiled_module
        
        # Either the module or its dependencies have changed, recompile
        return self.compile_file(abs_path)


def compile_elfin_module(file_path: str, search_paths: Optional[List[str]] = None) -> Any:
    """
    Compile an ELFIN module from a file.
    
    This is a convenience function that creates a ModuleCompiler and compiles a file.
    
    Args:
        file_path: Path to the file to compile
        search_paths: Paths to search for imported modules
        
    Returns:
        The compiled module
    """
    compiler = ModuleCompiler(search_paths)
    return compiler.compile_file(file_path)


def compile_elfin_module_incremental(file_path: str, search_paths: Optional[List[str]] = None) -> Any:
    """
    Compile an ELFIN module incrementally, reusing cached modules when possible.
    
    This is a convenience function that creates a ModuleCompiler and compiles a file
    with incremental compilation.
    
    Args:
        file_path: Path to the file to compile
        search_paths: Paths to search for imported modules
        
    Returns:
        The compiled module
    """
    compiler = ModuleCompiler(search_paths)
    return compiler.compile_incremental(file_path)
