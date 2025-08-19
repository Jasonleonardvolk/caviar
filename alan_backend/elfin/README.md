# ELFIN: Embedded Language For Integrated Networks

[![ELFIN CI Workflow](https://github.com/your-org/elfin/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/elfin/actions/workflows/ci.yml)

ELFIN is a domain-specific language for expressing and analyzing control systems,
especially those with formal guarantees like barrier and Lyapunov functions.

## Features

- **Expressive syntax** for control system specifications
- **Dimensional analysis** with automatic unit checking
- **Standard library** of common helper functions
- **Format enforcement** for clean, consistent code
- **Code generation** for target languages (Rust, C++, Python)
- **Verification** of control system properties

## Getting Started

### Installation

```bash
# Install from source
git clone https://github.com/your-org/elfin.git
cd elfin
pip install -e .
```

### Creating a Simple System

```elfin
# Import standard helpers
import StdHelpers from "std/helpers.elfin";

# Define a simple pendulum system
system Pendulum {
  # State variables: angle and angular velocity
  continuous_state: [theta, omega];
  
  # Input: applied torque
  input: [tau];
  
  # System parameters (with units)
  params {
    m: mass[kg] = 1.0;           # Mass
    l: length[m] = 1.0;          # Length
    g: acceleration[m/s^2] = 9.8; # Gravity
    b: damping[kg*m^2/s] = 0.1;   # Damping coefficient
  }
  
  # Continuous dynamics
  flow_dynamics {
    theta_dot = omega;
    omega_dot = -(g/l)*sin(theta) - (b/m/l^2)*omega + tau/(m*l^2);
  }
}
```

## Command Line Tools

### Format ELFIN Files

Format your ELFIN files according to style guidelines:

```bash
# Format all ELFIN files in a directory
elfin fmt templates/

# Check formatting without making changes (useful for CI)
elfin fmt --check templates/
```

### Check Unit Consistency

Check dimensional consistency across ELFIN files:

```bash
# Check unit consistency in a directory (warning mode)
elfin check-units templates/
```

### Generate Code

Generate code from ELFIN specifications:

```bash
# Generate Rust code
elfin generate --language rust pendulum.elfin
```

## Development Tools

### VS Code Integration

This repository includes VS Code integration:

- **Format on Command**: Use `Ctrl+Shift+B` to format the current file
- **Template Verification**: Use the "elfin verify template" task to check formatting and units
- **Code Snippets**: IntelliSense integration for standard helpers

### Pre-commit Hook

Set up the pre-commit hook for automatic verification:

```bash
npx husky install
chmod +x .husky/pre-commit
```

## API Documentation

For detailed API documentation, see the [API Reference](docs/api_reference.md).

## License

Apache License 2.0
