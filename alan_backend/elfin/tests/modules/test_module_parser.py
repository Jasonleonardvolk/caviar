"""
Tests for the ELFIN module-aware parser.

This module tests the functionality of the ModuleAwareParser class and related
components that handle module parsing in ELFIN.
"""

import os
import pytest
from pathlib import Path

from alan_backend.elfin.parser.module_ast import (
    ImportDecl, TemplateParamDecl, TemplateDecl, TemplateArgument,
    TemplateInstantiation, ModuleNode
)
from alan_backend.elfin.parser.module_parser import ModuleAwareParser, parse_elfin_module
from alan_backend.elfin.modules.resolver import ImportResolver
from alan_backend.elfin.modules.symbols import SymbolTable
from alan_backend.elfin.modules.templates import TemplateRegistry


# Get path to the test fixtures
FIXTURES_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "fixtures"


class TestModuleParser:
    """Tests for the ModuleAwareParser class."""
    
    def test_parse_import_statement(self):
        """Test parsing a simple import statement."""
        # Source code with an import statement
        source = """
        import Controller from "controller.elfin";
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the import was parsed correctly
        assert len(module.imports) == 1
        import_decl = module.imports[0]
        assert isinstance(import_decl, ImportDecl)
        assert len(import_decl.imports) == 1
        assert import_decl.imports[0]["name"] == "Controller"
        assert import_decl.imports[0]["alias"] is None
        assert import_decl.source == "controller.elfin"
    
    def test_parse_import_with_alias(self):
        """Test parsing an import statement with an alias."""
        # Source code with an import statement with alias
        source = """
        import Sensor as DistanceSensor from "sensors/distance.elfin";
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the import was parsed correctly
        assert len(module.imports) == 1
        import_decl = module.imports[0]
        assert isinstance(import_decl, ImportDecl)
        assert len(import_decl.imports) == 1
        assert import_decl.imports[0]["name"] == "Sensor"
        assert import_decl.imports[0]["alias"] == "DistanceSensor"
        assert import_decl.source == "sensors/distance.elfin"
    
    def test_parse_multiple_imports(self):
        """Test parsing an import statement with multiple imports."""
        # Source code with multiple imports
        source = """
        import { Vector3, Matrix3 } from "math/linear.elfin";
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the import was parsed correctly
        assert len(module.imports) == 1
        import_decl = module.imports[0]
        assert isinstance(import_decl, ImportDecl)
        assert len(import_decl.imports) == 2
        assert import_decl.imports[0]["name"] == "Vector3"
        assert import_decl.imports[0]["alias"] is None
        assert import_decl.imports[1]["name"] == "Matrix3"
        assert import_decl.imports[1]["alias"] is None
        assert import_decl.source == "math/linear.elfin"
    
    def test_parse_multiple_imports_with_aliases(self):
        """Test parsing an import statement with multiple imports and aliases."""
        # Source code with multiple imports and aliases
        source = """
        import { Vector3 as Vec3, Matrix3 as Mat3 } from "math/linear.elfin";
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the import was parsed correctly
        assert len(module.imports) == 1
        import_decl = module.imports[0]
        assert isinstance(import_decl, ImportDecl)
        assert len(import_decl.imports) == 2
        assert import_decl.imports[0]["name"] == "Vector3"
        assert import_decl.imports[0]["alias"] == "Vec3"
        assert import_decl.imports[1]["name"] == "Matrix3"
        assert import_decl.imports[1]["alias"] == "Mat3"
        assert import_decl.source == "math/linear.elfin"
    
    def test_parse_template_declaration(self):
        """Test parsing a simple template declaration."""
        # Source code with a template declaration
        source = """
        template Vector3(x=0.0, y=0.0, z=0.0) {
            // Template body
        }
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the template was parsed correctly
        assert len(module.templates) == 1
        template_decl = module.templates[0]
        assert isinstance(template_decl, TemplateDecl)
        assert template_decl.name == "Vector3"
        assert len(template_decl.parameters) == 3
        assert template_decl.parameters[0].name == "x"
        assert template_decl.parameters[0].default_value == 0.0
        assert template_decl.parameters[1].name == "y"
        assert template_decl.parameters[1].default_value == 0.0
        assert template_decl.parameters[2].name == "z"
        assert template_decl.parameters[2].default_value == 0.0
    
    def test_parse_template_with_typed_parameters(self):
        """Test parsing a template declaration with typed parameters."""
        # Source code with a template that has typed parameters
        source = """
        template Point(x: float, y: float) {
            // Template body
        }
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the template was parsed correctly
        assert len(module.templates) == 1
        template_decl = module.templates[0]
        assert isinstance(template_decl, TemplateDecl)
        assert template_decl.name == "Point"
        assert len(template_decl.parameters) == 2
        assert template_decl.parameters[0].name == "x"
        assert template_decl.parameters[0].param_type == "float"
        assert template_decl.parameters[0].default_value is None
        assert template_decl.parameters[1].name == "y"
        assert template_decl.parameters[1].param_type == "float"
        assert template_decl.parameters[1].default_value is None
    
    def test_parse_template_with_mixed_parameters(self):
        """Test parsing a template with mixed parameter types."""
        # Source code with a template that has mixed parameter types
        source = """
        template Mixed(a, b: int, c=3, d: float=4.0) {
            // Template body
        }
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that the template was parsed correctly
        assert len(module.templates) == 1
        template_decl = module.templates[0]
        assert isinstance(template_decl, TemplateDecl)
        assert template_decl.name == "Mixed"
        assert len(template_decl.parameters) == 4
        
        assert template_decl.parameters[0].name == "a"
        assert template_decl.parameters[0].param_type is None
        assert template_decl.parameters[0].default_value is None
        
        assert template_decl.parameters[1].name == "b"
        assert template_decl.parameters[1].param_type == "int"
        assert template_decl.parameters[1].default_value is None
        
        assert template_decl.parameters[2].name == "c"
        assert template_decl.parameters[2].param_type is None
        assert template_decl.parameters[2].default_value == 3
        
        assert template_decl.parameters[3].name == "d"
        assert template_decl.parameters[3].param_type == "float"
        assert template_decl.parameters[3].default_value == 4.0
    
    def test_parse_mixed_content(self):
        """Test parsing a file with imports, templates, and other declarations."""
        # Source code with imports, templates, and a concept
        source = """
        import Controller from "controller.elfin";
        import { Vector3, Matrix3 } from "math/linear.elfin";
        
        template PID(kp, ki=0.0, kd=0.0) {
            // Template body
        }
        
        concept Robot {
            // Concept body
        }
        """
        
        # Parse the source
        module = parse_elfin_module(source)
        
        # Check that everything was parsed correctly
        assert len(module.imports) == 2
        assert len(module.templates) == 1
        assert len(module.declarations) == 1  # The concept
    
    def test_resolver_integration(self):
        """Test integration with the import resolver."""
        # Create a resolver that can resolve imports from the test fixtures
        resolver = ImportResolver([FIXTURES_DIR])
        
        # Source code with an import
        source = """
        import Controller from "controller.elfin";
        """
        
        # Parse the source with the resolver
        symbol_table = SymbolTable()
        module = parse_elfin_module(
            source,
            resolver=resolver,
            symbol_table=symbol_table
        )
        
        # Check that the import was processed
        assert symbol_table.lookup("Controller") is not None
    
    def test_template_registry_integration(self):
        """Test integration with the template registry."""
        # Source code with a template
        source = """
        template Vector3(x=0.0, y=0.0, z=0.0) {
            // Template body
        }
        """
        
        # Parse the source with a template registry
        template_registry = TemplateRegistry()
        symbol_table = SymbolTable()
        module = parse_elfin_module(
            source,
            template_registry=template_registry,
            symbol_table=symbol_table
        )
        
        # Check that the template was registered
        assert template_registry.get_template("Vector3") is not None
        assert symbol_table.lookup("Vector3") is not None
    
    def test_parse_actual_files(self):
        """Test parsing actual ELFIN files from the fixtures."""
        # Test parsing the main.elfin file
        main_path = FIXTURES_DIR / "main.elfin"
        with open(main_path, "r") as f:
            source = f.read()
        
        # Parse the source
        resolver = ImportResolver([FIXTURES_DIR])
        symbol_table = SymbolTable()
        template_registry = TemplateRegistry()
        
        module = parse_elfin_module(
            source,
            file_path=str(main_path),
            resolver=resolver,
            symbol_table=symbol_table,
            template_registry=template_registry
        )
        
        # Check that the imports were processed
        assert len(module.imports) == 3
        
        # Check that the symbols were imported
        assert symbol_table.lookup("Controller") is not None
        assert symbol_table.lookup("Vector3") is not None
        assert symbol_table.lookup("Matrix3") is not None
        assert symbol_table.lookup("DistanceSensor") is not None
