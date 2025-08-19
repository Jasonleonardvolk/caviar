#!/usr/bin/env python3
"""
ELFIN Module System Demo

This script demonstrates the ELFIN module system by parsing a file with imports
and templates, and showing how the imports are resolved and templates are processed.
"""

import os
import sys
from pathlib import Path

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from alan_backend.elfin.modules.resolver import ImportResolver
from alan_backend.elfin.modules.symbols import SymbolTable
from alan_backend.elfin.modules.templates import TemplateRegistry
from alan_backend.elfin.parser.module_parser import parse_elfin_module


def print_separator(title):
    """Print a section separator."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def main():
    """Run the module system demo."""
    # Get the path to the test fixtures
    fixtures_dir = Path(repo_root) / "alan_backend" / "elfin" / "tests" / "modules" / "fixtures"
    
    # Print the header
    print_separator("ELFIN Module System Demo")
    print(f"Using fixtures from: {fixtures_dir}\n")
    
    # Create the module system components
    resolver = ImportResolver([fixtures_dir])
    symbol_table = SymbolTable()
    template_registry = TemplateRegistry()
    
    # Parse the main.elfin file
    main_path = fixtures_dir / "main.elfin"
    with open(main_path, "r") as f:
        source = f.read()
    
    print("Parsing main.elfin...")
    module = parse_elfin_module(
        source,
        file_path=str(main_path),
        resolver=resolver,
        symbol_table=symbol_table,
        template_registry=template_registry
    )
    
    # Print information about the parsed module
    print(f"\nFile: {module.path}")
    print(f"Imports: {len(module.imports)}")
    print(f"Templates: {len(module.templates)}")
    print(f"Declarations: {len(module.declarations)}")
    
    # Print information about the imports
    print_separator("Imports")
    for i, import_decl in enumerate(module.imports):
        print(f"Import #{i+1}:")
        print(f"  Source: {import_decl.source}")
        for item in import_decl.imports:
            name = item["name"]
            alias = item["alias"] or name
            print(f"  - {name}" + (f" as {alias}" if alias != name else ""))
    
    # Print information about the resolved symbols
    print_separator("Resolved Symbols")
    symbols = symbol_table.get_all_symbols()
    for name, symbol in symbols.items():
        print(f"Symbol: {name}")
        print(f"  Type: {symbol.symbol_type}")
        print(f"  Source: {symbol.source or 'local'}")
    
    # Parse the controller.elfin file to see a template definition
    controller_path = fixtures_dir / "controller.elfin"
    with open(controller_path, "r") as f:
        controller_source = f.read()
    
    print_separator("Template Definition")
    print("Parsing controller.elfin...")
    controller_module = parse_elfin_module(
        controller_source,
        file_path=str(controller_path),
        resolver=resolver,
        symbol_table=symbol_table,
        template_registry=template_registry
    )
    
    # Print information about the templates
    for i, template_decl in enumerate(controller_module.templates):
        print(f"Template #{i+1}: {template_decl.name}")
        print("  Parameters:")
        for param in template_decl.parameters:
            param_str = param.name
            if param.param_type:
                param_str += f": {param.param_type}"
            if param.default_value is not None:
                param_str += f" = {param.default_value}"
            print(f"    - {param_str}")
    
    # Show that the template is now in the registry
    print("\nTemplates in registry:")
    for name in template_registry.templates:
        template = template_registry.get_template(name)
        print(f"  - {name} ({len(template.parameters)} parameters)")
    
    print_separator("Demo Complete")


if __name__ == "__main__":
    main()
