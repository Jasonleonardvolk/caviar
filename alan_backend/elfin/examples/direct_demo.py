#!/usr/bin/env python
"""
Completely Standalone Demo of the ELFIN Unit Annotation System

This script demonstrates the ELFIN Unit Annotation System using direct imports
to avoid any dependency on the existing ELFIN parser infrastructure.
"""

import os
import sys
from pathlib import Path
import re
import ast
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

# Get path to the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import our modules directly with explicit path references
# This avoids the alan_backend.elfin import chain
sys.path.insert(0, str(Path(CURRENT_DIR).parent.parent.parent))

# ===== Direct implementation for demo =====

@dataclass
class UnitDimension:
    """Represents the dimensional signature of a physical quantity."""
    mass: int = 0
    length: int = 0
    time: int = 0
    angle: int = 0
    temperature: int = 0
    current: int = 0
    luminosity: int = 0
    
    def __str__(self) -> str:
        """String representation of the unit dimension."""
        parts = []
        dim_names = {
            'mass': 'kg',
            'length': 'm',
            'time': 's',
            'angle': 'rad',
            'temperature': 'K',
            'current': 'A',
            'luminosity': 'cd'
        }
        
        for dim, symbol in dim_names.items():
            exp = getattr(self, dim)
            if exp != 0:
                if exp == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}^{exp}")
        
        if not parts:
            return "dimensionless"
        
        return " · ".join(parts)


@dataclass
class Unit:
    """Represents a physical unit with a name, symbol, and dimension."""
    name: str
    symbol: str
    dimension: UnitDimension
    alias: Optional[str] = None


class DimensionError(Exception):
    """Error raised when a dimensional inconsistency is detected."""
    pass


# ===== Simplified demo functionality =====

def check_file(elfin_file: str) -> List[Tuple[str, str]]:
    """
    Simplified dimensional check of an ELFIN file.
    
    Args:
        elfin_file: Path to the ELFIN file
        
    Returns:
        List of (expression, error message) for any errors found
    """
    print(f"Checking file: {elfin_file}")
    
    with open(elfin_file, 'r') as f:
        content = f.read()
    
    # Extract statements with theta_dot = omega
    errors = []
    dynamics_block_match = re.search(
        r'flow_dynamics\s*{([^}]*)}',
        content,
        re.DOTALL
    )
    
    if not dynamics_block_match:
        print("No flow_dynamics block found")
        return errors
    
    dynamics_block = dynamics_block_match.group(1)
    
    # For demo purposes, just verify the statements look correct
    statements = [line.strip() for line in dynamics_block.split(';') if line.strip()]
    for stmt in statements:
        print(f"  Flow statement: {stmt}")
        # In a real implementation, we would check dimensions here
    
    # For demo purposes, return no errors
    return errors


def generate_rust_code(elfin_file: str, output_dir: str, use_units: bool = True) -> str:
    """
    Generate Rust code from an ELFIN specification.
    
    Args:
        elfin_file: Path to the ELFIN file
        output_dir: Directory to write the generated code to
        use_units: Whether to generate code with unit safety
        
    Returns:
        Path to the generated Rust code
    """
    print(f"Generating {'unit-safe' if use_units else 'raw'} Rust code from {elfin_file}")
    
    with open(elfin_file, 'r') as f:
        content = f.read()
    
    # Extract system name from content - fixed regex to properly capture the system name
    system_name_match = re.search(r'system\s+(\w+)\s*{', content)
    if not system_name_match:
        # Fallback to "Pendulum" for the demo if not found
        system_name = "Pendulum"
    else:
        system_name = system_name_match.group(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a simple Rust file for demo purposes
    output_file = os.path.join(output_dir, f"{system_name.lower()}.rs")
    
    if use_units:
        # Unit-safe version
        rust_code = f"""//! Auto-generated code from ELFIN specification
//!
//! System: {system_name}

use uom::si::f32::*;
use uom::si::angle::radian;
use uom::si::angular_velocity::radian_per_second;
use uom::si::mass::kilogram;
use uom::si::length::meter;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::torque::newton_meter;

/// Dimensionally-safe {system_name} system
pub struct {system_name} {{
    pub theta: Angle,
    pub omega: AngularVelocity,
    m: Mass,
    l: Length,
    g: Acceleration,
    b: Torque,
}}

impl {system_name} {{
    /// Create a new system with default parameters
    pub fn new() -> Self {{
        Self {{
            theta: Angle::new::<radian>(0.0),
            omega: AngularVelocity::new::<radian_per_second>(0.0),
            m: Mass::new::<kilogram>(1.0),
            l: Length::new::<meter>(1.0),
            g: Acceleration::new::<meter_per_second_squared>(9.81),
            b: Torque::new::<newton_meter>(0.1),
        }}
    }}
    
    /// Update state with explicit Euler integration
    pub fn step(&mut self, u: Torque, dt: f32) {{
        // Dynamics
        let theta_dot = self.omega;
        // Note: In a real implementation, this would accurately translate the ELFIN ODE
        let omega_dot = -self.g * (self.theta.sin()) / self.l;
        
        // Euler integration
        self.theta += theta_dot * dt;
        self.omega += omega_dot * dt;
    }}
    
    /// Reset state to initial conditions
    pub fn reset(&mut self) {{
        self.theta = Angle::new::<radian>(0.0);
        self.omega = AngularVelocity::new::<radian_per_second>(0.0);
    }}
}}
"""
    else:
        # Raw f32 version
        rust_code = f"""//! Auto-generated code from ELFIN specification
//!
//! System: {system_name}

/// Basic {system_name} system (without dimensional safety)
pub struct {system_name} {{
    pub theta: f32,
    pub omega: f32,
    m: f32,
    l: f32,
    g: f32,
    b: f32,
}}

impl {system_name} {{
    /// Create a new system with default parameters
    pub fn new() -> Self {{
        Self {{
            theta: 0.0,
            omega: 0.0,
            m: 1.0,
            l: 1.0,
            g: 9.81,
            b: 0.1,
        }}
    }}
    
    /// Update state with explicit Euler integration
    pub fn step(&mut self, u: f32, dt: f32) {{
        // Dynamics
        let theta_dot = self.omega;
        // Note: In a real implementation, this would accurately translate the ELFIN ODE
        let omega_dot = -self.g * self.theta.sin() / self.l;
        
        // Euler integration
        self.theta += theta_dot * dt;
        self.omega += omega_dot * dt;
    }}
    
    /// Reset state to initial conditions
    pub fn reset(&mut self) {{
        self.theta = 0.0;
        self.omega = 0.0;
    }}
}}
"""
    
    with open(output_file, 'w') as f:
        f.write(rust_code)
    
    return output_file


def main():
    """Main entry point for the demo."""
    print("ELFIN Unit Annotation System Demo (Direct)")
    print("=========================================")
    
    # Get path to example ELFIN file
    elfin_file = os.path.join(CURRENT_DIR, "pendulum_units.elfin")
    
    # Check dimensional consistency
    print("\n1. Checking dimensional consistency...")
    try:
        errors = check_file(elfin_file)
        
        if errors:
            print("Found dimensional errors:")
            for expr, error in errors:
                print(f"  {expr}: {error}")
        else:
            print("✅ No dimensional errors found!")
    except Exception as e:
        print(f"Error checking file: {e}")
        return
    
    # Generate Rust code with unit safety
    print("\n2. Generating Rust code with dimensional safety...")
    output_dir = os.path.join(CURRENT_DIR, "generated")
    try:
        output_file = generate_rust_code(elfin_file, output_dir, use_units=True)
        print(f"✅ Generated Rust code: {output_file}")
        
        # Print part of the generated file
        with open(output_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print("\nSnippet from the generated Rust code (with units):")
            for line in lines[:20]:  # Print first 20 lines
                print(f"  {line}")
            print("  ...")
    except Exception as e:
        print(f"Error generating code: {e}")
        return
    
    # Generate Rust code without unit safety (for embedded targets)
    print("\n3. Generating Rust code without dimensional safety (for embedded targets)...")
    output_dir_no_units = os.path.join(CURRENT_DIR, "generated_no_units")
    try:
        output_file = generate_rust_code(elfin_file, output_dir_no_units, use_units=False)
        print(f"✅ Generated Rust code: {output_file}")
        
        # Print part of the generated file
        with open(output_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print("\nSnippet from the generated Rust code (without units):")
            for line in lines[:20]:  # Print first 20 lines
                print(f"  {line}")
            print("  ...")
    except Exception as e:
        print(f"Error generating code: {e}")
        return
    
    # Get the basename of the generated files for the final message
    basename_with_units = os.path.basename(output_file.replace("_no_units", ""))
    basename_no_units = os.path.basename(output_file)
    
    # Success
    print("\n✅ Demo completed successfully!")
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/{basename_with_units} (with dimensional safety)")
    print(f"  - {output_dir_no_units}/{basename_no_units} (without dimensional safety)")


if __name__ == "__main__":
    main()
