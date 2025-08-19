"""
Rust code generation for ELFIN specifications.

This module provides utilities for generating Rust code from ELFIN specifications,
with a focus on dimensional safety using the uom crate for unit-of-measure types.
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from ..units import Unit, UnitDimension, UnitTable


class RustCodeGenerator:
    """
    Generator for Rust code from ELFIN specifications.
    
    This class provides methods for generating Rust code from ELFIN specifications,
    including support for unit-safe type checking using the uom crate.
    """
    
    def __init__(self, unit_table: Optional[UnitTable] = None, use_units: bool = True):
        """
        Initialize the Rust code generator.
        
        Args:
            unit_table: Unit table to use for unit lookups (optional)
            use_units: Whether to generate code with unit safety (optional)
        """
        self.unit_table = unit_table or UnitTable()
        self.use_units = use_units
    
    def generate_code(self, elfin_file: str, output_dir: str) -> str:
        """
        Generate Rust code from an ELFIN specification.
        
        Args:
            elfin_file: Path to the ELFIN file
            output_dir: Directory to write the generated code to
            
        Returns:
            Path to the generated Rust code
        """
        # Read the ELFIN file
        with open(elfin_file, 'r') as f:
            elfin_content = f.read()
        
        # Extract system name
        system_name_match = re.search(r'system\s+(\w+)', elfin_content)
        if not system_name_match:
            raise ValueError("Could not find system name in ELFIN file")
        
        system_name = system_name_match.group(1)
        
        # Extract state variables with units
        state_vars = self._extract_state_vars(elfin_content)
        
        # Extract parameters with units
        parameters = self._extract_parameters(elfin_content)
        
        # Extract inputs with units
        inputs = self._extract_inputs(elfin_content)
        
        # Extract dynamics
        dynamics = self._extract_dynamics(elfin_content)
        
        # Generate Rust code
        rust_code = self._generate_rust_code(
            system_name, state_vars, parameters, inputs, dynamics
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write Rust code to file
        output_file = os.path.join(output_dir, f"{system_name.lower()}.rs")
        with open(output_file, 'w') as f:
            f.write(rust_code)
        
        return output_file
    
    def _extract_state_vars(self, elfin_content: str) -> List[Dict[str, Any]]:
        """
        Extract state variables with units from ELFIN content.
        
        Args:
            elfin_content: ELFIN file content
            
        Returns:
            List of state variable dictionaries
        """
        # Match continuous_state block
        state_block_match = re.search(
            r'continuous_state\s*{([^}]*)}',
            elfin_content,
            re.DOTALL
        )
        
        if not state_block_match:
            return []
        
        state_block = state_block_match.group(1)
        
        # Extract variables with units
        var_pattern = r'(\w+)\s*:\s*(\w+)(?:\[([^\]]+)\])?(?:\s*=\s*([^;]+))?;'
        vars_matches = re.finditer(var_pattern, state_block)
        
        state_vars = []
        for match in vars_matches:
            var_name, var_type, unit_str, default_value = match.groups()
            
            # Try to get the unit from the unit table
            unit = None
            if unit_str:
                try:
                    unit = Unit.parse(unit_str)
                except ValueError:
                    print(f"Warning: Unknown unit {unit_str} for {var_name}")
            
            state_vars.append({
                'name': var_name,
                'type': var_type,
                'unit': unit,
                'default': default_value.strip() if default_value else None
            })
        
        return state_vars
    
    def _extract_parameters(self, elfin_content: str) -> List[Dict[str, Any]]:
        """
        Extract parameters with units from ELFIN content.
        
        Args:
            elfin_content: ELFIN file content
            
        Returns:
            List of parameter dictionaries
        """
        # Match parameters block
        param_block_match = re.search(
            r'parameters\s*{([^}]*)}',
            elfin_content,
            re.DOTALL
        )
        
        if not param_block_match:
            return []
        
        param_block = param_block_match.group(1)
        
        # Extract parameters with units
        param_pattern = r'(\w+)\s*:\s*(\w+)(?:\[([^\]]+)\])?(?:\s*=\s*([^;]+))?;'
        param_matches = re.finditer(param_pattern, param_block)
        
        parameters = []
        for match in param_matches:
            param_name, param_type, unit_str, default_value = match.groups()
            
            # Try to get the unit from the unit table
            unit = None
            if unit_str:
                try:
                    unit = Unit.parse(unit_str)
                except ValueError:
                    print(f"Warning: Unknown unit {unit_str} for {param_name}")
            
            parameters.append({
                'name': param_name,
                'type': param_type,
                'unit': unit,
                'default': default_value.strip() if default_value else None
            })
        
        return parameters
    
    def _extract_inputs(self, elfin_content: str) -> List[Dict[str, Any]]:
        """
        Extract inputs with units from ELFIN content.
        
        Args:
            elfin_content: ELFIN file content
            
        Returns:
            List of input dictionaries
        """
        # Match input block
        input_block_match = re.search(
            r'input\s*{([^}]*)}',
            elfin_content,
            re.DOTALL
        )
        
        if not input_block_match:
            return []
        
        input_block = input_block_match.group(1)
        
        # Extract inputs with units
        input_pattern = r'(\w+)\s*:\s*(\w+)(?:\[([^\]]+)\])?(?:\s*=\s*([^;]+))?;'
        input_matches = re.finditer(input_pattern, input_block)
        
        inputs = []
        for match in input_matches:
            input_name, input_type, unit_str, default_value = match.groups()
            
            # Try to get the unit from the unit table
            unit = None
            if unit_str:
                try:
                    unit = Unit.parse(unit_str)
                except ValueError:
                    print(f"Warning: Unknown unit {unit_str} for {input_name}")
            
            inputs.append({
                'name': input_name,
                'type': input_type,
                'unit': unit,
                'default': default_value.strip() if default_value else None
            })
        
        return inputs
    
    def _extract_dynamics(self, elfin_content: str) -> Dict[str, str]:
        """
        Extract dynamics from ELFIN content.
        
        Args:
            elfin_content: ELFIN file content
            
        Returns:
            Dictionary of variable name to expression
        """
        # Match flow_dynamics block
        dynamics_block_match = re.search(
            r'flow_dynamics\s*{([^}]*)}',
            elfin_content,
            re.DOTALL
        )
        
        if not dynamics_block_match:
            return {}
        
        dynamics_block = dynamics_block_match.group(1)
        
        # Extract dynamics expressions
        dynamics_pattern = r'(\w+)(?:_dot)?\s*=\s*([^;]+);'
        dynamics_matches = re.finditer(dynamics_pattern, dynamics_block)
        
        dynamics = {}
        for match in dynamics_matches:
            var_name, expr = match.groups()
            dynamics[var_name] = expr.strip()
        
        return dynamics
    
    def _generate_rust_code(
        self,
        system_name: str,
        state_vars: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
        dynamics: Dict[str, str]
    ) -> str:
        """
        Generate Rust code for the given system.
        
        Args:
            system_name: Name of the system
            state_vars: List of state variable dictionaries
            parameters: List of parameter dictionaries
            inputs: List of input dictionaries
            dynamics: Dictionary of variable name to expression
            
        Returns:
            Generated Rust code
        """
        # Generate code with or without unit safety
        if self.use_units:
            return self._generate_uom_code(system_name, state_vars, parameters, inputs, dynamics)
        else:
            return self._generate_f32_code(system_name, state_vars, parameters, inputs, dynamics)
    
    def _generate_uom_code(
        self,
        system_name: str,
        state_vars: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
        dynamics: Dict[str, str]
    ) -> str:
        """
        Generate Rust code with uom unit safety.
        
        Args:
            system_name: Name of the system
            state_vars: List of state variable dictionaries
            parameters: List of parameter dictionaries
            inputs: List of input dictionaries
            dynamics: Dictionary of variable name to expression
            
        Returns:
            Generated Rust code
        """
        # Map from dimension to Rust uom type
        dim_to_uom = {
            'mass': ('Mass', 'kilogram'),
            'length': ('Length', 'meter'),
            'time': ('Time', 'second'),
            'angle': ('Angle', 'radian'),
            'angular_velocity': ('AngularVelocity', 'radian_per_second'),
            'angular_acceleration': ('AngularAcceleration', 'radian_per_second_squared'),
            'velocity': ('Velocity', 'meter_per_second'),
            'acceleration': ('Acceleration', 'meter_per_second_squared'),
            'force': ('Force', 'newton'),
            'torque': ('Torque', 'newton_meter'),
            'energy': ('Energy', 'joule'),
            'power': ('Power', 'watt'),
            'angular_damping': ('Torque', 'newton_meter'),  # Approximation for now
        }
        
        # Collect all dimensions used in the system
        all_dimensions = set()
        for var in state_vars + parameters + inputs:
            if var['unit'] and var['type'] in dim_to_uom:
                all_dimensions.add(var['type'])
        
        # Build import list
        uom_imports = ['use uom::si::f32::*;']
        for dim, (uom_type, _) in dim_to_uom.items():
            if dim in all_dimensions:
                uom_imports.append(f"use uom::si::{dim}::{dim_to_uom[dim][1]};")
        
        # Generate struct definition
        struct_lines = [f"/// Dimensionally-safe {system_name} system", f"pub struct {system_name} {{"]
        
        # Add state variables
        for var in state_vars:
            if var['unit'] and var['type'] in dim_to_uom:
                uom_type, _ = dim_to_uom[var['type']]
                struct_lines.append(f"    pub {var['name']}: {uom_type},")
            else:
                struct_lines.append(f"    pub {var['name']}: f32,")
        
        # Add parameters
        for param in parameters:
            if param['unit'] and param['type'] in dim_to_uom:
                uom_type, _ = dim_to_uom[param['type']]
                struct_lines.append(f"    {param['name']}: {uom_type},")
            else:
                struct_lines.append(f"    {param['name']}: f32,")
        
        struct_lines.append("}")
        
        # Generate impl block
        impl_lines = [f"impl {system_name} {{"]
        
        # Add constructor
        impl_lines.append("    /// Create a new system with default parameters")
        impl_lines.append("    pub fn new() -> Self {")
        impl_lines.append("        Self {")
        
        # Initialize state variables
        for var in state_vars:
            default = var['default'] or "0.0"
            if var['unit'] and var['type'] in dim_to_uom:
                uom_type, unit = dim_to_uom[var['type']]
                impl_lines.append(f"            {var['name']}: {uom_type}::new::<{unit}>({default}),")
            else:
                impl_lines.append(f"            {var['name']}: {default},")
        
        # Initialize parameters
        for param in parameters:
            default = param['default'] or "0.0"
            if param['unit'] and param['type'] in dim_to_uom:
                uom_type, unit = dim_to_uom[param['type']]
                impl_lines.append(f"            {param['name']}: {uom_type}::new::<{unit}>({default}),")
            else:
                impl_lines.append(f"            {param['name']}: {default},")
        
        impl_lines.append("        }")
        impl_lines.append("    }")
        
        # Add step method
        impl_lines.append("")
        impl_lines.append("    /// Update state with explicit Euler integration")
        input_params = []
        for input_var in inputs:
            if input_var['unit'] and input_var['type'] in dim_to_uom:
                uom_type, _ = dim_to_uom[input_var['type']]
                input_params.append(f"{input_var['name']}: {uom_type}")
            else:
                input_params.append(f"{input_var['name']}: f32")
        
        impl_lines.append(f"    pub fn step(&mut self, {', '.join(input_params)}, dt: f32) {{")
        
        # Add dynamics computation
        # This is a simplified approach; for a real implementation, we'd need
        # to properly translate ELFIN expressions to Rust expressions with units
        for state_var in state_vars:
            var_name = state_var['name']
            dot_var = f"{var_name}_dot"
            if dot_var in dynamics:
                # Very simple expression translation for demo purposes
                expr = dynamics[dot_var]
                impl_lines.append(f"        // {dot_var} = {expr}")
                impl_lines.append(f"        // TODO: Properly translate expression with units")
                impl_lines.append(f"        let {dot_var} = {expr};  // Placeholder")
            else:
                impl_lines.append(f"        // No dynamics for {var_name}")
        
        # Add Euler integration
        for state_var in state_vars:
            var_name = state_var['name']
            dot_var = f"{var_name}_dot"
            if dot_var in dynamics:
                impl_lines.append(f"        self.{var_name} += {dot_var} * dt;")
        
        impl_lines.append("    }")
        
        # Add reset method
        impl_lines.append("")
        impl_lines.append("    /// Reset state to initial conditions")
        impl_lines.append("    pub fn reset(&mut self) {")
        for var in state_vars:
            if var['default']:
                if var['unit'] and var['type'] in dim_to_uom:
                    _, unit = dim_to_uom[var['type']]
                    impl_lines.append(f"        self.{var['name']} = {var['type'].capitalize()}::new::<{unit}>({var['default']});")
                else:
                    impl_lines.append(f"        self.{var['name']} = {var['default']};")
            else:
                if var['unit'] and var['type'] in dim_to_uom:
                    _, unit = dim_to_uom[var['type']]
                    impl_lines.append(f"        self.{var['name']} = {var['type'].capitalize()}::new::<{unit}>(0.0);")
                else:
                    impl_lines.append(f"        self.{var['name']} = 0.0;")
        
        impl_lines.append("    }")
        
        impl_lines.append("}")
        
        # Combine everything
        code = [
            "//! Auto-generated code from ELFIN specification",
            "//!",
            f"//! System: {system_name}",
            "",
            *uom_imports,
            "",
            *struct_lines,
            "",
            *impl_lines,
            "",
            "// TODO: Add proper WASM exports if needed",
            "#[cfg(feature = \"wasm\")]",
            "mod wasm {",
            "    use super::*;",
            "    use wasm_bindgen::prelude::*;",
            "",
            "    #[wasm_bindgen]",
            f"    pub struct {system_name}Sim {{",
            f"        system: {system_name},",
            "    }}",
            "",
            "    #[wasm_bindgen]",
            f"    impl {system_name}Sim {{",
            "        #[wasm_bindgen(constructor)]",
            "        pub fn new() -> Self {",
            f"            Self {{ system: {system_name}::new() }}",
            "        }}",
            "    }}",
            "}"
        ]
        
        return "\n".join(code)
    
    def _generate_f32_code(
        self,
        system_name: str,
        state_vars: List[Dict[str, Any]],
        parameters: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
        dynamics: Dict[str, str]
    ) -> str:
        """
        Generate Rust code without unit safety (using f32).
        
        Args:
            system_name: Name of the system
            state_vars: List of state variable dictionaries
            parameters: List of parameter dictionaries
            inputs: List of input dictionaries
            dynamics: Dictionary of variable name to expression
            
        Returns:
            Generated Rust code
        """
        # Generate struct definition
        struct_lines = [f"/// Basic {system_name} system (without dimensional safety)", f"pub struct {system_name} {{"]
        
        # Add state variables
        for var in state_vars:
            struct_lines.append(f"    pub {var['name']}: f32,")
        
        # Add parameters
        for param in parameters:
            struct_lines.append(f"    {param['name']}: f32,")
        
        struct_lines.append("}")
        
        # Generate impl block
        impl_lines = [f"impl {system_name} {{"]
        
        # Add constructor
        impl_lines.append("    /// Create a new system with default parameters")
        impl_lines.append("    pub fn new() -> Self {")
        impl_lines.append("        Self {")
        
        # Initialize state variables
        for var in state_vars:
            default = var['default'] or "0.0"
            impl_lines.append(f"            {var['name']}: {default},")
        
        # Initialize parameters
        for param in parameters:
            default = param['default'] or "0.0"
            impl_lines.append(f"            {param['name']}: {default},")
        
        impl_lines.append("        }")
        impl_lines.append("    }")
        
        # Add step method
        impl_lines.append("")
        impl_lines.append("    /// Update state with explicit Euler integration")
        input_params = [f"{input_var['name']}: f32" for input_var in inputs]
        impl_lines.append(f"    pub fn step(&mut self, {', '.join(input_params)}, dt: f32) {{")
        
        # Add dynamics computation
        for state_var in state_vars:
            var_name = state_var['name']
            dot_var = f"{var_name}_dot"
            if dot_var in dynamics:
                expr = dynamics[dot_var]
                impl_lines.append(f"        let {dot_var} = {expr};")
            else:
                impl_lines.append(f"        // No dynamics for {var_name}")
        
        # Add Euler integration
        for state_var in state_vars:
            var_name = state_var['name']
            dot_var = f"{var_name}_dot"
            if dot_var in dynamics:
                impl_lines.append(f"        self.{var_name} += {dot_var} * dt;")
        
        impl_lines.append("    }")
        
        # Add reset method
        impl_lines.append("")
        impl_lines.append("    /// Reset state to initial conditions")
        impl_lines.append("    pub fn reset(&mut self) {")
        for var in state_vars:
            default = var['default'] or "0.0"
            impl_lines.append(f"        self.{var['name']} = {default};")
        
        impl_lines.append("    }")
        
        impl_lines.append("}")
        
        # Combine everything
        code = [
            "//! Auto-generated code from ELFIN specification",
            "//!",
            f"//! System: {system_name}",
            "",
            *struct_lines,
            "",
            *impl_lines
        ]
        
        return "\n".join(code)


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
    generator = RustCodeGenerator(use_units=use_units)
    return generator.generate_code(elfin_file, output_dir)
