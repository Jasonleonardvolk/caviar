# Welcome to ELFIN 1.0

Thank you for installing the ELFIN Language Support extension! This guide will help you get started with the ELFIN language and take advantage of all the features available.

## Getting Started

ELFIN is a domain-specific language designed for control system specification. It focuses on dimensional consistency and mathematical rigor.

### Creating Your First ELFIN System

You can quickly create a new ELFIN system by pressing `Ctrl+Shift+P` and selecting `ELFIN: New Controller`. This will create a new file with a basic system template.

Alternatively, you can use the status bar button (`$(play) ELFIN`) to build and run your ELFIN system.

## Key Features

### Dimensional Analysis

ELFIN performs dimensional analysis at compile time to ensure your equations are physically consistent. If you write:

```elfin
v = 5.0 * [m/s];
x = 2.0 * [m];
position = x + v;  # Warning: Dimensional mismatch [m] + [m/s]
```

The system will warn you about the mismatch and offer a quick-fix to convert between units.

### Standard Library

ELFIN comes with a standard library of common helper functions:

```elfin
import Helpers from "std/helpers.elfin";

angle = Helpers.wrapAngle(theta);  # Normalize to [-π, π]
absValue = Helpers.hAbs(x);        # Absolute value
```

### Visual Aids

- **Inlay Hints**: See the dimensions of expressions directly in your code
- **CodeLens**: Quick access to documentation and tests for each system
- **Hover Information**: View type information and documentation by hovering
- **Syntax Highlighting**: Units are highlighted differently from variables

## Keyboard Shortcuts

- `Shift+Alt+F`: Format document
- `F2`: Rename symbol
- `Ctrl+Space`: Code completion
- `Ctrl+Shift+P`: Command palette (try typing "ELFIN:" to see available commands)

## Documentation

For more information, check out the [ELFIN documentation](https://elfin-lang.github.io/docs/).

## Feedback

If you encounter any issues or have suggestions, please submit them to the [ELFIN GitHub repository](https://github.com/elfin-lang/elfin/issues).

Happy coding!
