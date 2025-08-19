"""
Tests for the ELFIN module resolver system.

This module tests the functionality of the ImportResolver class and related
components that handle module imports in ELFIN.
"""

import os
import pytest
from pathlib import Path

from alan_backend.elfin.modules.resolver import ImportResolver, ModuleSearchPath
from alan_backend.elfin.modules.errors import (
    ModuleNotFoundError,
    CircularDependencyError,
    ModuleParseError
)


# Get path to the test fixtures
FIXTURES_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "fixtures"


class TestModuleSearchPath:
    """Tests for the ModuleSearchPath class."""
    
    def test_create_with_absolute_path(self):
        """Test creating a search path with an absolute path."""
        path = FIXTURES_DIR
        search_path = ModuleSearchPath(path)
        
        assert search_path.path == Path(path)
        assert search_path.resolved_path == Path(path).resolve()
    
    def test_create_with_relative_path(self):
        """Test creating a search path with a relative path."""
        base_dir = FIXTURES_DIR
        path = "math"
        search_path = ModuleSearchPath(path, base_dir)
        
        assert search_path.path == Path(path)
        assert search_path.base_dir == Path(base_dir)
        assert search_path.resolved_path == (Path(base_dir) / path).resolve()
    
    def test_resolve_existing_file(self):
        """Test resolving an existing file."""
        search_path = ModuleSearchPath(FIXTURES_DIR)
        
        # Test with exact path
        resolved = search_path.resolve("controller.elfin")
        assert resolved is not None
        assert resolved.name == "controller.elfin"
        
        # Test with directory/file path
        resolved = search_path.resolve("math/linear.elfin")
        assert resolved is not None
        assert resolved.name == "linear.elfin"
    
    def test_resolve_nonexistent_file(self):
        """Test resolving a non-existent file."""
        search_path = ModuleSearchPath(FIXTURES_DIR)
        
        resolved = search_path.resolve("nonexistent.elfin")
        assert resolved is None
    
    def test_resolve_with_implicit_extension(self):
        """Test resolving a file without specifying the extension."""
        search_path = ModuleSearchPath(FIXTURES_DIR)
        
        # Should automatically append .elfin extension
        resolved = search_path.resolve("controller")
        assert resolved is not None
        assert resolved.name == "controller.elfin"


class TestImportResolver:
    """Tests for the ImportResolver class."""
    
    def test_init_with_defaults(self):
        """Test initializing with default values."""
        resolver = ImportResolver()
        
        assert len(resolver.search_paths) >= 1  # At least the current directory
    
    def test_init_with_search_paths(self):
        """Test initializing with custom search paths."""
        search_paths = [FIXTURES_DIR, FIXTURES_DIR / "math"]
        resolver = ImportResolver(search_paths)
        
        assert len(resolver.search_paths) >= 2
        # Check that the resolved paths match what we expect
        resolved_paths = [p.resolved_path for p in resolver.search_paths]
        assert Path(FIXTURES_DIR).resolve() in resolved_paths
    
    def test_resolve_module(self):
        """Test resolving a module."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # Resolve a top-level module
        path, module = resolver.resolve("controller.elfin")
        assert path.name == "controller.elfin"
        assert "content" in module  # Our test parser returns a dict with content
        
        # Resolve a module in a subdirectory
        path, module = resolver.resolve("math/linear.elfin")
        assert path.name == "linear.elfin"
        assert "content" in module
    
    def test_resolve_with_implicit_extension(self):
        """Test resolving a module without specifying the extension."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # Should automatically append .elfin extension
        path, module = resolver.resolve("controller")
        assert path.name == "controller.elfin"
    
    def test_resolve_from_relative_import(self):
        """Test resolving a module from a relative import."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # First resolve the main module
        main_path, _ = resolver.resolve("main.elfin")
        
        # Then resolve controller from main
        path, module = resolver.resolve("controller.elfin", from_module=main_path)
        assert path.name == "controller.elfin"
    
    def test_module_not_found(self):
        """Test handling a module that cannot be found."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        with pytest.raises(ModuleNotFoundError):
            resolver.resolve("nonexistent.elfin")
    
    def test_caching(self):
        """Test that modules are cached."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # First resolution should parse the module
        path1, module1 = resolver.resolve("controller.elfin")
        
        # Second resolution should use the cache
        path2, module2 = resolver.resolve("controller.elfin")
        
        # Check that we got the same module object (not just equal modules)
        assert module1 is module2
    
    def test_clear_cache(self):
        """Test clearing the module cache."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # First resolution
        path1, module1 = resolver.resolve("controller.elfin")
        
        # Clear the cache
        resolver.clear_cache()
        
        # Second resolution should parse again
        path2, module2 = resolver.resolve("controller.elfin")
        
        # Check that we got a different module object
        assert module1 is not module2
    
    def test_add_search_path(self):
        """Test adding a search path."""
        resolver = ImportResolver()
        
        # Try to resolve a module that doesn't exist in the default paths
        with pytest.raises(ModuleNotFoundError):
            resolver.resolve("controller.elfin")
        
        # Add the fixtures directory as a search path
        resolver.add_search_path(FIXTURES_DIR)
        
        # Now the resolver should find the module
        path, module = resolver.resolve("controller.elfin")
        assert path.name == "controller.elfin"


class TestCircularImports:
    """Tests for handling circular imports."""
    
    def test_detect_circular_import(self, monkeypatch):
        """Test detecting a circular import."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # Mock _get_import_chain to simulate a circular dependency
        def mock_get_import_chain(self, target_path, from_module):
            if from_module and from_module.name == "main.elfin" and target_path.name == "controller.elfin":
                return [from_module]
            return []
        
        # Apply the monkey patch
        monkeypatch.setattr(ImportResolver, "_get_import_chain", mock_get_import_chain)
        
        # First resolve the main module
        main_path, _ = resolver.resolve("main.elfin")
        
        # Set up the circular dependency
        monkeypatch.setattr(ImportResolver, "_get_import_chain", 
                            lambda self, target, source: [source] if source and target.name == source.name else [])
        
        # Try to resolve a circular import
        with pytest.raises(CircularDependencyError):
            resolver.resolve(main_path, from_module=main_path)


class TestParseErrors:
    """Tests for handling parse errors."""
    
    def test_module_parse_error(self, monkeypatch):
        """Test handling a module that cannot be parsed."""
        resolver = ImportResolver([FIXTURES_DIR])
        
        # Mock open to simulate a parse error
        original_open = open
        
        def mock_open(path, mode):
            if str(path).endswith("parser_error.elfin"):
                raise Exception("Simulated parse error")
            return original_open(path, mode)
        
        # Apply the monkey patch
        monkeypatch.setattr("builtins.open", mock_open)
        
        # Try to resolve a module that will have a parse error
        with pytest.raises(ModuleParseError):
            resolver.resolve("parser_error.elfin")
