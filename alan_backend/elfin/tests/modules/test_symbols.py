"""
Tests for the ELFIN symbol table system.

This module tests the functionality of the SymbolTable class and related
components that handle symbol management and scope tracking in ELFIN.
"""

import pytest

from alan_backend.elfin.modules.symbols import Symbol, Scope, SymbolTable


class TestSymbol:
    """Tests for the Symbol class."""
    
    def test_symbol_creation(self):
        """Test creating a symbol."""
        symbol = Symbol("x", "variable", 42, "test.elfin")
        
        assert symbol.name == "x"
        assert symbol.symbol_type == "variable"
        assert symbol.value == 42
        assert symbol.source == "test.elfin"
    
    def test_symbol_string_representation(self):
        """Test the string representation of a symbol."""
        # Symbol with source
        symbol1 = Symbol("x", "variable", 42, "test.elfin")
        assert str(symbol1) == "x (variable from test.elfin)"
        
        # Symbol without source
        symbol2 = Symbol("y", "function")
        assert str(symbol2) == "y (function)"


class TestScope:
    """Tests for the Scope class."""
    
    def test_scope_creation(self):
        """Test creating a scope."""
        # Create a scope without a parent
        scope1 = Scope("global")
        assert scope1.name == "global"
        assert scope1.parent is None
        assert len(scope1.symbols) == 0
        assert len(scope1.children) == 0
        
        # Create a scope with a parent
        scope2 = Scope("local", scope1)
        assert scope2.name == "local"
        assert scope2.parent is scope1
        assert len(scope2.symbols) == 0
        assert len(scope2.children) == 0
        
        # Check that the parent has the child
        assert len(scope1.children) == 1
        assert scope1.children[0] is scope2
    
    def test_symbol_definition(self):
        """Test defining a symbol in a scope."""
        scope = Scope("test")
        
        # Define a symbol
        symbol = Symbol("x", "variable", 42)
        result = scope.define(symbol)
        
        # Check that the symbol was added to the scope
        assert len(scope.symbols) == 1
        assert "x" in scope.symbols
        assert scope.symbols["x"] is symbol
        assert result is symbol
        
        # Check that defining a symbol with the same name raises an error
        with pytest.raises(ValueError):
            scope.define(Symbol("x", "variable", 43))
    
    def test_symbol_lookup(self):
        """Test looking up a symbol in a scope."""
        # Create a scope hierarchy
        global_scope = Scope("global")
        function_scope = Scope("function", global_scope)
        block_scope = Scope("block", function_scope)
        
        # Define symbols in different scopes
        global_x = Symbol("x", "variable", 1)
        global_scope.define(global_x)
        
        function_y = Symbol("y", "variable", 2)
        function_scope.define(function_y)
        
        function_x = Symbol("x", "variable", 3)  # Shadows global_x
        function_scope.define(function_x)
        
        block_z = Symbol("z", "variable", 4)
        block_scope.define(block_z)
        
        # Test lookup from block scope
        assert block_scope.lookup("z") is block_z  # Defined in block scope
        assert block_scope.lookup("y") is function_y  # Defined in function scope
        assert block_scope.lookup("x") is function_x  # Shadows global_x
        assert block_scope.lookup("w") is None  # Not defined anywhere
        
        # Test local_only
        assert block_scope.lookup("y", local_only=True) is None  # Not in block scope
        
        # Test lookup from function scope
        assert function_scope.lookup("z") is None  # Defined in child scope, not visible
        assert function_scope.lookup("y") is function_y  # Defined in function scope
        assert function_scope.lookup("x") is function_x  # Shadows global_x
        
        # Test lookup from global scope
        assert global_scope.lookup("z") is None  # Defined in grandchild scope, not visible
        assert global_scope.lookup("y") is None  # Defined in child scope, not visible
        assert global_scope.lookup("x") is global_x  # Defined in global scope
    
    def test_get_all_symbols(self):
        """Test getting all symbols from a scope and its parents."""
        # Create a scope hierarchy
        global_scope = Scope("global")
        function_scope = Scope("function", global_scope)
        block_scope = Scope("block", function_scope)
        
        # Define symbols in different scopes
        global_scope.define(Symbol("x", "variable", 1))
        global_scope.define(Symbol("y", "variable", 2))
        
        function_scope.define(Symbol("z", "variable", 3))
        function_scope.define(Symbol("x", "variable", 4))  # Shadows global x
        
        block_scope.define(Symbol("w", "variable", 5))
        
        # Get all symbols from block scope
        all_symbols = block_scope.get_all_symbols()
        
        # Check that we got the expected symbols
        assert len(all_symbols) == 4  # x, y, z, w
        assert all_symbols["w"].value == 5  # From block scope
        assert all_symbols["z"].value == 3  # From function scope
        assert all_symbols["x"].value == 4  # From function scope (shadowing global)
        assert all_symbols["y"].value == 2  # From global scope
    
    def test_get_local_symbols(self):
        """Test getting symbols from a scope only."""
        # Create a scope hierarchy
        global_scope = Scope("global")
        function_scope = Scope("function", global_scope)
        
        # Define symbols in different scopes
        global_scope.define(Symbol("x", "variable", 1))
        global_scope.define(Symbol("y", "variable", 2))
        
        function_scope.define(Symbol("z", "variable", 3))
        function_scope.define(Symbol("x", "variable", 4))  # Shadows global x
        
        # Get local symbols from function scope
        local_symbols = function_scope.get_local_symbols()
        
        # Check that we got the expected symbols
        assert len(local_symbols) == 2  # z, x
        assert local_symbols["z"].value == 3  # From function scope
        assert local_symbols["x"].value == 4  # From function scope
        assert "y" not in local_symbols  # From global scope, not included
    
    def test_create_child(self):
        """Test creating a child scope."""
        parent = Scope("parent")
        
        # Create a child scope
        child = parent.create_child("child")
        
        # Check that the child was created correctly
        assert child.name == "child"
        assert child.parent is parent
        assert len(parent.children) == 1
        assert parent.children[0] is child


class TestSymbolTable:
    """Tests for the SymbolTable class."""
    
    def test_init(self):
        """Test initializing a symbol table."""
        table = SymbolTable()
        
        assert table.current_scope is table.global_scope
        assert table.global_scope.name == "global"
        assert len(table.imported_modules) == 0
    
    def test_enter_exit_scope(self):
        """Test entering and exiting scopes."""
        table = SymbolTable()
        
        # Enter a function scope
        function_scope = table.enter_scope("function")
        assert table.current_scope is function_scope
        assert function_scope.name == "function"
        assert function_scope.parent is table.global_scope
        
        # Enter a block scope
        block_scope = table.enter_scope("block")
        assert table.current_scope is block_scope
        assert block_scope.name == "block"
        assert block_scope.parent is function_scope
        
        # Exit the block scope
        result = table.exit_scope()
        assert table.current_scope is function_scope
        assert result is function_scope
        
        # Exit the function scope
        result = table.exit_scope()
        assert table.current_scope is table.global_scope
        assert result is table.global_scope
        
        # Can't exit the global scope
        with pytest.raises(ValueError):
            table.exit_scope()
    
    def test_define_lookup(self):
        """Test defining and looking up symbols."""
        table = SymbolTable()
        
        # Define a symbol in the global scope
        global_x = table.define("x", "variable", 1)
        assert global_x.name == "x"
        assert global_x.symbol_type == "variable"
        assert global_x.value == 1
        
        # Enter a function scope
        table.enter_scope("function")
        
        # Define a symbol in the function scope
        function_y = table.define("y", "variable", 2)
        assert function_y.name == "y"
        
        # Define a symbol that shadows a global symbol
        function_x = table.define("x", "variable", 3)
        assert function_x.name == "x"
        assert function_x.value == 3
        
        # Lookup symbols
        assert table.lookup("y") is function_y  # Defined in function scope
        assert table.lookup("x") is function_x  # Shadows global_x
        assert table.lookup("z") is None  # Not defined anywhere
        
        # Lookup with local_only
        assert table.lookup("y", local_only=True) is function_y  # Defined in function scope
        assert table.lookup("x", local_only=True) is function_x  # Defined in function scope
        
        # Exit function scope and lookup again
        table.exit_scope()
        assert table.lookup("y") is None  # Not defined in global scope
        assert table.lookup("x") is global_x  # Defined in global scope
    
    def test_get_all_symbols(self):
        """Test getting all symbols from the current scope and its parents."""
        table = SymbolTable()
        
        # Define symbols in global scope
        table.define("x", "variable", 1)
        table.define("y", "variable", 2)
        
        # Enter a function scope
        table.enter_scope("function")
        
        # Define symbols in function scope
        table.define("z", "variable", 3)
        table.define("x", "variable", 4)  # Shadows global x
        
        # Get all symbols
        all_symbols = table.get_all_symbols()
        
        # Check that we got the expected symbols
        assert len(all_symbols) == 3  # x, y, z
        assert all_symbols["z"].value == 3  # From function scope
        assert all_symbols["x"].value == 4  # From function scope (shadowing global)
        assert all_symbols["y"].value == 2  # From global scope
    
    def test_get_local_symbols(self):
        """Test getting symbols from the current scope only."""
        table = SymbolTable()
        
        # Define symbols in global scope
        table.define("x", "variable", 1)
        table.define("y", "variable", 2)
        
        # Enter a function scope
        table.enter_scope("function")
        
        # Define symbols in function scope
        table.define("z", "variable", 3)
        table.define("x", "variable", 4)  # Shadows global x
        
        # Get local symbols
        local_symbols = table.get_local_symbols()
        
        # Check that we got the expected symbols
        assert len(local_symbols) == 2  # z, x
        assert local_symbols["z"].value == 3  # From function scope
        assert local_symbols["x"].value == 4  # From function scope
        assert "y" not in local_symbols  # From global scope, not included
    
    def test_import_module(self):
        """Test importing symbols from a module."""
        table = SymbolTable()
        
        # Create a mock module with symbols
        module_symbols = {
            "Vector3": Symbol("Vector3", "template", None, "math/linear.elfin"),
            "Matrix3": Symbol("Matrix3", "template", None, "math/linear.elfin"),
        }
        
        # Import symbols from the module
        imports = [
            {"name": "Vector3", "alias": None},
            {"name": "Matrix3", "alias": "Mat3"},
        ]
        table.import_module("math/linear.elfin", module_symbols, imports)
        
        # Check that the symbols were imported
        assert table.lookup("Vector3") is not None
        assert table.lookup("Vector3").symbol_type == "template"
        assert table.lookup("Vector3").source == "math/linear.elfin"
        
        assert table.lookup("Mat3") is not None
        assert table.lookup("Mat3").symbol_type == "template"
        assert table.lookup("Mat3").source == "math/linear.elfin"
        
        # Check that the imported modules were tracked
        imported_modules = table.get_imported_modules()
        assert len(imported_modules) == 1
        assert "math/linear.elfin" in imported_modules
        assert "Vector3" in imported_modules["math/linear.elfin"]
        assert "Matrix3" in imported_modules["math/linear.elfin"]
        
        # Check that importing a non-existent symbol raises an error
        with pytest.raises(ValueError):
            table.import_module("math/linear.elfin", module_symbols, [
                {"name": "NonExistent", "alias": None},
            ])
