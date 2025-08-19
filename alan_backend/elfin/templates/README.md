# ELFIN Package Templates

This directory contains template files for creating new ELFIN packages using the `elf new` command.

## Available Templates

- **basic**: Minimal package with core dependencies
- **application**: Full-featured application with UI components  
- **library**: Reusable library package
- **quadrotor**: Complete quadrotor controller example
- **manipulator**: Robotic manipulator control with joint limits and human safety
- **mobile**: Mobile robot navigation and obstacle avoidance
- **aerial**: Aerial vehicle control (multirotor, fixed-wing, VTOL)
- **hybrid**: Hybrid system control with discrete mode transitions
- **barrier**: Specialized package for barrier function verification
- **koopman**: Package for Koopman-based analysis
- **learning**: Template for machine learning components

## Creating a New Package

To create a new package using a template:

```bash
run_elfpkg.bat new my_package --template quadrotor
```

### Domain-Specific Templates

For domain-specific applications, consider these specialized templates:

- **manipulator**: For robot arm control with collision avoidance and force control
- **mobile**: For ground vehicles with path planning and obstacle avoidance
- **aerial**: For drones and aircraft with geofencing and attitude safety
- **hybrid**: For systems with mixed continuous/discrete dynamics

## Template Structure

Each template directory must contain:

1. A basic structure that will be copied to the new package
2. Template placeholder values that will be replaced during creation:
   - `{{name}}`: Package name
   - `{{edition}}`: ELFIN edition
   - `{{author}}`: Author name (if provided)

## Creating Custom Templates

To create a custom template:

1. Create a new directory under `templates/`
2. Add template files with appropriate placeholders
3. The template will automatically become available for the `elf new` command
