#!/usr/bin/env python3
"""
ELFIN Project Initializer

This script creates a new ELFIN project with a template that includes:
- Standard library imports
- Basic system structure
- Parameter definitions with units
- Flow dynamics

Usage:
  python elfin_init.py [--name NAME] [--type TYPE] [output_dir]

Arguments:
  output_dir       Output directory for the project (default: current directory)

Options:
  --name NAME      Name of the system [default: MySystem]
  --type TYPE      Type of system template to use: 
                   manipulator, mobile, pendulum [default: pendulum]
  --quick          Use quick mode (minimal prompting)
  -h, --help       Show this help message and exit
"""

import os
import sys
import argparse
import textwrap
from pathlib import Path

# Templates for different system types
TEMPLATES = {
    "pendulum": {
        "description": "Single pendulum system",
        "states": ["theta", "omega"],
        "inputs": ["tau"],
        "params": [
            {"name": "m", "type": "mass", "unit": "kg", "value": "1.0"},
            {"name": "l", "type": "length", "unit": "m", "value": "1.0"},
            {"name": "g", "type": "acceleration", "unit": "m/s^2", "value": "9.81"},
            {"name": "b", "type": "damping", "unit": "kg*m^2/s", "value": "0.1"},
        ],
        "dynamics": [
            "# Angular position derivative",
            "theta_dot = omega;",
            "",
            "# Angular velocity derivative",
            "omega_dot = (tau - b * omega - m * g * l * Helpers.wrapAngle(theta)) / (m * l^2);",
        ]
    },
    "mobile": {
        "description": "Mobile robot with unicycle dynamics",
        "states": ["x", "y", "theta"],
        "inputs": ["v", "omega"],
        "params": [
            {"name": "wheelbase", "type": "length", "unit": "m", "value": "0.5"},
        ],
        "dynamics": [
            "# Position derivatives",
            "x_dot = v * cos(theta);",
            "y_dot = v * sin(theta);",
            "",
            "# Orientation derivative",
            "theta_dot = omega;",
        ]
    },
    "manipulator": {
        "description": "Robotic manipulator arm",
        "states": ["q1", "q2", "dq1", "dq2"],
        "inputs": ["tau1", "tau2"],
        "params": [
            {"name": "m1", "type": "mass", "unit": "kg", "value": "1.0"},
            {"name": "m2", "type": "mass", "unit": "kg", "value": "0.5"},
            {"name": "l1", "type": "length", "unit": "m", "value": "1.0"},
            {"name": "l2", "type": "length", "unit": "m", "value": "0.8"},
            {"name": "g", "type": "acceleration", "unit": "m/s^2", "value": "9.81"},
        ],
        "dynamics": [
            "# Joint position derivatives",
            "q1_dot = dq1;",
            "q2_dot = dq2;",
            "",
            "# Simplified joint acceleration model",
            "dq1_dot = (tau1 - m2 * l1 * l2 * sin(q2) * dq2^2 - (m1 + m2) * g * l1 * sin(q1)) / (m1 + m2 * l2^2);",
            "dq2_dot = (tau2 + m2 * l1 * l2 * sin(q2) * dq1^2 - m2 * g * l2 * sin(q1 + q2)) / (m2 * l2^2);",
        ]
    }
}

def generate_system_file(name, system_type, output_dir):
    """Generate an ELFIN system file based on the template."""
    template = TEMPLATES.get(system_type)
    if not template:
        print(f"Unknown system type: {system_type}")
        print(f"Available types: {', '.join(TEMPLATES.keys())}")
        sys.exit(1)
    
    # Create the states and inputs sections
    states = ", ".join(template["states"])
    inputs = ", ".join(template["inputs"])
    
    # Create the parameters section
    params_lines = []
    for param in template["params"]:
        params_lines.append(f"  {param['name']}: {param['type']}[{param['unit']}] = {param['value']};")
    params = "\n".join(params_lines)
    
    # Create the dynamics section
    dynamics_lines = []
    for line in template["dynamics"]:
        if line:
            dynamics_lines.append(f"  {line}")
        else:
            dynamics_lines.append("")
    dynamics = "\n".join(dynamics_lines)
    
    # Create the full file content
    content = f"""import Helpers from "std/helpers.elfin";

system {name} {{
  continuous_state: [{states}];
  inputs: [{inputs}];
  
  params {{
{params}
  }}
  
  flow_dynamics {{
{dynamics}
  }}
}}
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to a file
    file_path = os.path.join(output_dir, f"{name.lower()}.elfin")
    with open(file_path, "w") as f:
        f.write(content)
    
    return file_path

def create_std_dir(output_dir):
    """Create the standard library directory if it doesn't exist."""
    std_dir = os.path.join(output_dir, "std")
    os.makedirs(std_dir, exist_ok=True)
    
    # Check if helpers.elfin already exists
    helpers_path = os.path.join(std_dir, "helpers.elfin")
    if not os.path.exists(helpers_path):
        # Create the helpers file
        helpers_content = """helpers {
  hAbs(x) = if x >= 0 then x else -x;
  hMin(a, b) = if a <= b then a else b;
  hMax(a, b) = if a >= b then a else b;
  wrapAngle(t) = mod(t + pi, 2*pi) - pi;
  clamp(x, min, max) = if x < min then min else if x > max then max else x;
  lerp(a, b, t) = a + (b - a) * t;
}
"""
        with open(helpers_path, "w") as f:
            f.write(helpers_content)
    
    return std_dir

def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new ELFIN project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__)
    )
    parser.add_argument("output_dir", nargs="?", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--name", default="MySystem", help="Name of the system")
    parser.add_argument("--type", default="pendulum", choices=TEMPLATES.keys(), help="Type of system template")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (minimal prompting)")
    
    args = parser.parse_args()
    
    # Resolve the output directory to an absolute path
    output_dir = os.path.abspath(args.output_dir)
    
    # If not in quick mode, prompt for confirmation
    if not args.quick:
        print(f"Creating a new ELFIN project with the following settings:")
        print(f"  System name: {args.name}")
        print(f"  System type: {args.type} ({TEMPLATES[args.type]['description']})")
        print(f"  Output directory: {output_dir}")
        
        confirm = input("Proceed? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("Aborted.")
            return
    
    # Create the std directory if needed
    std_dir = create_std_dir(output_dir)
    
    # Generate the system file
    file_path = generate_system_file(args.name, args.type, output_dir)
    
    print(f"Created ELFIN project:")
    print(f"  - System file: {file_path}")
    print(f"  - Standard library: {std_dir}")
    print(f"  - Template type: {args.type}")
    print("\nTo use the system, run:")
    print(f"  elfin check-units {file_path}")
    print(f"  elfin fmt {file_path}")

if __name__ == "__main__":
    main()
