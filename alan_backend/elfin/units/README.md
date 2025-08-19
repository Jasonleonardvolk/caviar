# ELFIN Unit Annotation System

The ELFIN Unit Annotation System adds physical dimension tracking to the ELFIN domain-specific language. This enables catch dimensional errors at compile time, improving safety and reliability of robotic control systems.

## Overview

This system provides:

1. **Physical Dimension Tracking**: Annotate state variables, parameters, and inputs with their physical units
2. **Dimensional Consistency Checking**: Verify that operations (addition, multiplication, etc.) respect dimensional rules
3. **Rust Code Generation**: Generate dimensionally-safe Rust code using the `uom` crate

## Features

### Unit Annotations

Add unit annotations to any variable in your ELFIN specification:

```elfin
continuous_state {
  theta: angle[rad];       # Angle in radians
  omega: angular_velocity[rad/s];  # Angular velocity
}

parameters {
  m: mass[kg] = 1.0;       # Mass in kilograms
  l: length[m] = 1.0;      # Length in meters
  g: acceleration[m/s^2] = 9.81;  # Acceleration
}
```

### Dimension Checking

The system will automatically check dimensional consistency in expressions:

```elfin
# Valid: Same dimensions on both sides
theta_dot = omega;  # [rad/s] = [rad/s] ✓

# Valid: Consistent dimensions after calculation
omega_dot = -g/l * sin(theta);  # [m/s²]/[m] * sin([rad]) = [rad/s²] ✓

# Invalid: Mismatched dimensions
error = theta + g;  # [rad] + [m/s²] ❌
```

### Code Generation

Generate dimensionally-safe Rust code using the `uom` crate:

```rust
pub struct Pendulum {
    pub theta: Angle,
    pub omega: AngularVelocity,
    m: Mass,
    l: Length,
    g: Acceleration,
    b: Torque,
}

impl Pendulum {
    pub fn step(&mut self, u: Torque, dt: f32) {
        // Dimensionally-safe computation
        self.theta += self.omega * dt;
        // ...
    }
}
```

## Supported Units

The system supports all SI base units and many derived units:

### Base Units

- `kg`: Mass (kilogram)
- `m`: Length (meter)
- `s`: Time (second)
- `rad`: Angle (radian)
- `K`: Temperature (kelvin)
- `A`: Current (ampere)
- `cd`: Luminosity (candela)

### Common Derived Units

- `m/s`: Velocity
- `m/s²`: Acceleration
- `N`: Force (newton)
- `N·m`: Torque
- `rad/s`: Angular velocity
- `rad/s²`: Angular acceleration
- `N·m·s/rad`: Rotational damping (angular damping)
- `J`: Energy (joule)
- `W`: Power (watt)

## Example Usage

### Example ELFIN File

```elfin
# Simple pendulum system with dimensional units
system Pendulum {
  continuous_state {
    theta: angle[rad];
    omega: angular_velocity[rad/s];
  }

  parameters {
    m: mass[kg] = 1.0;
    l: length[m] = 1.0;
    g: acceleration[m/s^2] = 9.81;
    b: angular_damping[N·m·s/rad] = 0.1;
  }

  input {
    u: torque[N·m];
  }

  flow_dynamics {
    theta_dot = omega;
    omega_dot = -g/l * sin(theta) - b/(m*l^2) * omega + u/(m*l^2);
  }
}
```

### Command-Line Usage

Check dimensional consistency:

```bash
python -m alan_backend.elfin.cli check examples/pendulum_units.elfin
```

Generate Rust code:

```bash
python -m alan_backend.elfin.cli generate --language rust examples/pendulum_units.elfin
```

Generate Rust code without unit safety (for embedded targets):

```bash
python -m alan_backend.elfin.cli generate --language rust --no-units examples/pendulum_units.elfin
```

## Implementation Details

The system consists of several components:

1. **Unit Registry**: Base and derived units with dimensional information
2. **Dimension Checker**: Expression analyzer that verifies dimensional consistency
3. **Code Generator**: Translates ELFIN to target languages (Rust, etc.)

### Extending the System

To add new units:

1. Add the unit to `UnitTable._add_derived_units()` in `units.py`
2. Map the unit to a Rust type in `RustCodeGenerator._generate_uom_code()` in `rust.py`

### Limitations

- Expression parsing is currently based on Python's `ast` module, which has limitations
- Full ELFIN language support requires a more sophisticated parser
- Expression translation to target languages is incomplete

## Future Work

- Full ELFIN parser integration
- Support for additional target languages (C++, Python)
- Higher-order numerical methods for simulation
- More sophisticated expression translation

## License

[MIT License](LICENSE)
