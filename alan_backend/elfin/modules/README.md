# ELFIN Module System

This directory contains the implementation of the ELFIN module system, which enables code reuse and composition through imports and templates.

## Components

The module system consists of the following components:

### 1. Module Resolver (`resolver.py`)

The module resolver is responsible for finding, loading, and caching modules imported by ELFIN files. It handles:

- Resolving import paths (absolute and relative)
- Managing module search paths
- Caching parsed modules
- Detecting circular dependencies

### 2. Symbol Table (`symbols.py`)

The symbol table keeps track of symbols defined in a module and provides methods for:

- Managing symbol scopes
- Defining and looking up symbols
- Importing symbols from other modules
- Handling symbol shadowing

### 3. Template System (`templates.py`)

The template system enables the definition and instantiation of parametrized components:

- Template definitions with parameters
- Template instantiation with arguments
- Parameter validation
- Template registry for looking up templates

### 4. Grammar Extensions (`grammar.py`)

This module defines the grammar extensions for the module system:

- Import statement syntax
- Template instantiation syntax
- Parser for import statements
- Parser for template declarations

## Usage Examples

### Importing Modules

```elfin
// Import a single component
import Controller from "controller.elfin";

// Import multiple components
import { Vector3, Matrix3 } from "math/linear.elfin";

// Import with alias
import Sensor as DistanceSensor from "sensors/distance.elfin";
```

### Defining Templates

```elfin
template Vector3(x=0.0, y=0.0, z=0.0) {
  parameters {
    x: dimensionless = x;
    y: dimensionless = y;
    z: dimensionless = z;
  }
  
  // ... template body ...
}
```

### Instantiating Templates

```elfin
// Using positional arguments
myVector: Vector3(1.0, 2.0, 3.0);

// Using named arguments
anotherVector: Vector3(x=10.0, z=30.0);

// Using default values for omitted parameters
partialVector: Vector3(x=5.0);
```

## Integration with Main Parser

To fully integrate the module system with the main ELFIN parser, the following steps remain:

1. **AST Extensions**: Extend the main AST nodes to include import and template information
2. **Import Processing**: Add import statement processing to the parser
3. **Symbol Resolution**: Integrate the symbol table with the main parser for name resolution
4. **Template Processing**: Integrate template instantiation into the parser

## Testing

The module system is tested using a comprehensive suite of unit tests:

- `test_resolver.py` - Tests for the module resolver
- `test_symbols.py` - Tests for the symbol table
- `test_templates.py` - Tests for the template system

Test fixtures are provided in the `tests/modules/fixtures` directory, which contains example ELFIN files with imports and templates.

## Next Steps

1. Integrate with main parser:
   - Add import statement processing to the parser
   - Extend the AST nodes to include import and template information
   - Integrate the symbol table for name resolution
   - Modify the parser to use the module resolver

2. Expand template system:
   - Add support for generic templates
   - Implement template specialization
   - Add template inheritance

3. Add advanced features:
   - Conditional compilation
   - Reflection capabilities
   - Interface definitions
