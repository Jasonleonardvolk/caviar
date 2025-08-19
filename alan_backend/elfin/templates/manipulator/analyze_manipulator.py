#!/usr/bin/env python3
"""
ELFIN Manipulator Controller Syntax Analyzer and Validator

This script performs a static analysis of the manipulator_controller.elfin file
to validate syntax, check for circular references, and verify mathematical correctness.
"""

import os
import re
import json
import sys
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

class ELFINAnalyzer:
    """Analyzer for ELFIN specification files."""
    
    def __init__(self, file_path: str):
        """Initialize with path to ELFIN file."""
        self.file_path = file_path
        self.content = ""
        self.sections = {}
        self.symbols = {}
        self.references = {}
        self.issues = []
        
    def load_file(self) -> None:
        """Load ELFIN file content."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"Loaded file: {self.file_path} ({len(self.content)} bytes)")
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
    
    def parse_sections(self) -> None:
        """Parse the file into main sections."""
        # Find all top-level blocks: helpers, system, lyapunov, barrier, mode
        section_pattern = r'(?:helpers|system|lyapunov|barrier|mode)\s+([A-Za-z0-9_]+)\s*{([^}]*)}'
        for match in re.finditer(section_pattern, self.content, re.DOTALL):
            section_type = match.group(0).split()[0]
            section_name = match.group(1)
            section_content = match.group(2)
            
            key = f"{section_type}:{section_name}"
            self.sections[key] = section_content.strip()
            print(f"Found section: {key}")
    
    def extract_symbols(self) -> None:
        """Extract symbol definitions from all sections."""
        for section_key, content in self.sections.items():
            section_type, section_name = section_key.split(':')
            
            # Extract different symbol types based on section
            if section_type == 'system':
                # Extract state and input variables
                state_match = re.search(r'continuous_state:\s*\[(.*?)\]', content, re.DOTALL)
                if state_match:
                    states = [s.strip() for s in state_match.group(1).split(',')]
                    for state in states:
                        if state:  # Skip empty entries
                            self.symbols[state] = {
                                'type': 'state',
                                'section': section_key
                            }
                
                input_match = re.search(r'input:\s*\[(.*?)\]', content, re.DOTALL)
                if input_match:
                    inputs = [i.strip() for i in input_match.group(1).split(',')]
                    for input_var in inputs:
                        if input_var:  # Skip empty entries
                            self.symbols[input_var] = {
                                'type': 'input',
                                'section': section_key
                            }
                
                # Extract parameters
                params_match = re.search(r'params\s*{([^}]*)}', content, re.DOTALL)
                if params_match:
                    params_content = params_match.group(1)
                    param_entries = re.findall(r'([A-Za-z0-9_]+):(.*?);', params_content + ';')
                    for param, value in param_entries:
                        self.symbols[param.strip()] = {
                            'type': 'param',
                            'section': section_key,
                            'value': value.strip()
                        }
                
                # Extract dynamics variables
                dynamics_match = re.search(r'flow_dynamics\s*{([^}]*)}', content, re.DOTALL)
                if dynamics_match:
                    dynamics_content = dynamics_match.group(1)
                    dynamics_entries = re.findall(r'([A-Za-z0-9_]+)\s*=\s*(.*?);', dynamics_content + ';')
                    for var, expr in dynamics_entries:
                        self.symbols[var.strip()] = {
                            'type': 'dynamic',
                            'section': section_key,
                            'expression': expr.strip()
                        }
            
            elif section_type == 'helpers':
                # Extract helper functions
                helper_entries = re.findall(r'([A-Za-z0-9_]+)\((.*?)\)\s*=\s*(.*?);', content + ';')
                for helper, args, expr in helper_entries:
                    self.symbols[helper.strip()] = {
                        'type': 'helper',
                        'section': section_key,
                        'args': [a.strip() for a in args.split(',')],
                        'expression': expr.strip()
                    }
            
            elif section_type in ['lyapunov', 'barrier']:
                # Extract V or B expressions and parameters
                params_match = re.search(r'params\s*{([^}]*)}', content, re.DOTALL)
                if params_match:
                    params_content = params_match.group(1)
                    param_entries = re.findall(r'([A-Za-z0-9_]+):(.*?);', params_content + ';')
                    for param, value in param_entries:
                        symbol_key = f"{section_name}.{param.strip()}"
                        self.symbols[symbol_key] = {
                            'type': 'param',
                            'section': section_key,
                            'value': value.strip()
                        }
                
                # Extract main function (V for lyapunov, B for barrier)
                main_var = 'V' if section_type == 'lyapunov' else 'B'
                main_match = re.search(rf'{main_var}\s*=\s*(.*?);', content, re.DOTALL)
                if main_match:
                    self.symbols[f"{section_name}.{main_var}"] = {
                        'type': 'expression',
                        'section': section_key,
                        'expression': main_match.group(1).strip()
                    }
                
                # Extract alpha function for barriers
                if section_type == 'barrier':
                    alpha_match = re.search(r'alphaFun\s*=\s*(.*?);', content, re.DOTALL)
                    if alpha_match:
                        self.symbols[f"{section_name}.alphaFun"] = {
                            'type': 'expression',
                            'section': section_key,
                            'expression': alpha_match.group(1).strip()
                        }
            
            elif section_type == 'mode':
                # Extract controller expressions and parameters
                params_match = re.search(r'params\s*{([^}]*)}', content, re.DOTALL)
                if params_match:
                    params_content = params_match.group(1)
                    param_entries = re.findall(r'([A-Za-z0-9_]+):(.*?);', params_content + ';')
                    for param, value in param_entries:
                        symbol_key = f"{section_name}.{param.strip()}"
                        self.symbols[symbol_key] = {
                            'type': 'param',
                            'section': section_key,
                            'value': value.strip()
                        }
                
                # Extract controller expressions
                controller_match = re.search(r'controller\s*{([^}]*)}', content, re.DOTALL)
                if controller_match:
                    controller_content = controller_match.group(1)
                    controller_entries = re.findall(r'([A-Za-z0-9_]+)\s*=\s*(.*?);', controller_content + ';')
                    for var, expr in controller_entries:
                        symbol_key = f"{section_name}.{var.strip()}"
                        self.symbols[symbol_key] = {
                            'type': 'controller',
                            'section': section_key,
                            'expression': expr.strip()
                        }
    
    def analyze_references(self) -> None:
        """Analyze symbol references in expressions."""
        # Extract all potential identifiers from expressions
        for symbol, info in self.symbols.items():
            if 'expression' in info:
                # Find potential variable references in the expression
                # This simple regex matches potential variable names
                identifiers = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', info['expression'])
                
                # Store references but filter out numeric functions and constants
                common_functions = {'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'if', 'then', 'else'}
                common_constants = {'pi', 'e'}
                
                references = set()
                for identifier in identifiers:
                    if (
                        identifier not in common_functions and 
                        identifier not in common_constants and
                        identifier not in info.get('args', [])  # Skip function arguments
                    ):
                        references.add(identifier)
                
                self.references[symbol] = references
    
    def check_for_circular_references(self) -> None:
        """Detect circular references in dynamics definitions."""
        # Focus on dynamics expressions
        dynamics_vars = {s: info for s, info in self.symbols.items() if info['type'] == 'dynamic'}
        
        for var, info in dynamics_vars.items():
            expression = info['expression']
            # Exact name match (not as part of another name)
            var_name = var.replace('_dot', '')
            pattern = r'\b' + re.escape(var_name) + r'\b'
            
            # Check if variable references itself
            if re.search(pattern, expression):
                # Special case: q1_dot = dq1 is fine because dq1 is a different variable
                if var.endswith('_dot') and var_name != expression:
                    continue
                
                self.issues.append({
                    'type': 'circular_reference',
                    'severity': 'error',
                    'message': f"Circular reference detected: {var} depends on itself",
                    'variable': var,
                    'expression': expression
                })
    
    def validate_references(self) -> None:
        """Validate that all referenced symbols are defined."""
        for symbol, references in self.references.items():
            section_type = self.symbols[symbol]['section'].split(':')[0]
            
            for ref in references:
                # Handle section-specific symbols (e.g., barrier.alpha)
                if '.' in ref:
                    if ref not in self.symbols:
                        self.issues.append({
                            'type': 'undefined_reference',
                            'severity': 'error',
                            'message': f"Undefined reference: {ref} in {symbol}",
                            'variable': symbol,
                            'reference': ref
                        })
                # Regular symbols
                elif ref not in self.symbols:
                    self.issues.append({
                        'type': 'undefined_reference',
                        'severity': 'warning',
                        'message': f"Potentially undefined reference: {ref} in {symbol}",
                        'variable': symbol,
                        'reference': ref
                    })
    
    def check_dynamics_completeness(self) -> None:
        """Verify that all state variables have dynamics definitions."""
        state_vars = {s for s, info in self.symbols.items() if info['type'] == 'state'}
        dynamics_vars = {s.replace('_dot', '') for s, info in self.symbols.items() if info['type'] == 'dynamic'}
        
        missing_dynamics = state_vars - dynamics_vars
        if missing_dynamics:
            for var in missing_dynamics:
                self.issues.append({
                    'type': 'missing_dynamics',
                    'severity': 'error',
                    'message': f"Missing dynamics definition for state variable: {var}",
                    'variable': var
                })
    
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        self.load_file()
        self.parse_sections()
        self.extract_symbols()
        self.analyze_references()
        self.check_for_circular_references()
        self.validate_references()
        self.check_dynamics_completeness()
    
    def export_results(self, output_file: str) -> None:
        """Export analysis results to a JSON file."""
        results = {
            'file': self.file_path,
            'sections': len(self.sections),
            'symbols': len(self.symbols),
            'references': {k: list(v) for k, v in self.references.items()},
            'issues': self.issues
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results written to: {output_file}")
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        print("\n=== ELFIN Analysis Summary ===")
        print(f"File: {self.file_path}")
        print(f"Sections: {len(self.sections)}")
        print(f"Symbols: {len(self.symbols)}")
        
        # Count issues by severity
        error_count = sum(1 for issue in self.issues if issue['severity'] == 'error')
        warning_count = sum(1 for issue in self.issues if issue['severity'] == 'warning')
        
        print(f"Issues: {len(self.issues)} ({error_count} errors, {warning_count} warnings)")
        
        if self.issues:
            print("\n=== Issues ===")
            for issue in self.issues:
                severity_marker = "ERROR" if issue['severity'] == 'error' else "WARNING"
                print(f"[{severity_marker}] {issue['message']}")


def prepare_simulation_scaffold(analyzer: ELFINAnalyzer, output_dir: str) -> None:
    """
    Generate a simulation scaffold based on the analyzed ELFIN file.
    
    Args:
        analyzer: The ELFINAnalyzer with completed analysis
        output_dir: Directory to write simulation files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract system parameters and state variables
    system_keys = [k for k in analyzer.sections.keys() if k.startswith('system:')]
    if not system_keys:
        print("Error: No system section found in ELFIN file")
        return
    
    system_key = system_keys[0]
    system_name = system_key.split(':')[1]
    
    # Get state variables
    state_vars = [s for s, info in analyzer.symbols.items() if info['type'] == 'state']
    
    # Get dynamics expressions
    dynamics = {s: info['expression'] for s, info in analyzer.symbols.items() if info['type'] == 'dynamic'}
    
    # Get parameters with values
    params = {s: info['value'] for s, info in analyzer.symbols.items() 
              if info['type'] == 'param' and info['section'] == system_key}
    
    # Generate a simulation stub
    sim_path = os.path.join(output_dir, f"simulate_{system_name.lower()}.py")
    
    with open(sim_path, 'w', encoding='utf-8') as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
{system_name} Simulation Scaffold

Auto-generated from ELFIN specification. Provides a basic simulation
framework for the {system_name} system with JAX-based integration.
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial

class {system_name}Simulator:
    \"\"\"Simulator for the {system_name} system.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the simulator with default parameters.\"\"\"
        # System parameters
        self.params = {{
{chr(10).join(f'            "{param}": {value},' for param, value in params.items())}
        }}
        
        # State indices for easier access
        self.state_indices = {{
{chr(10).join(f'            "{var}": {i},' for i, var in enumerate(state_vars))}
        }}
        
        # Current controller mode
        self.current_mode = "JointPD"
        
        # Control input
        self.control_input = np.zeros({len(dynamics) // 2})  # Assuming half the dynamics are for control inputs
    
    def dynamics(self, state, t, control):
        \"\"\"
        System dynamics function for integration.
        
        Args:
            state: System state vector
            t: Current time
            control: Control input vector
            
        Returns:
            State derivative vector
        \"\"\"
        # Create a state dictionary for easy reference
        state_dict = {{}}
        for var, idx in self.state_indices.items():
            state_dict[var] = state[idx]
        
        # Create a control dictionary
{chr(10).join(f'        {input_var} = control[{i}]' for i, input_var in enumerate(['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']))}
        
        # Compute derivatives using ELFIN dynamics equations
        derivatives = np.zeros_like(state)
        
        # Position derivatives (dq = ω)
        derivatives[self.state_indices["q1"]] = state_dict["dq1"]
        derivatives[self.state_indices["q2"]] = state_dict["dq2"]
        derivatives[self.state_indices["q3"]] = state_dict["dq3"]
        derivatives[self.state_indices["q4"]] = state_dict["dq4"]
        derivatives[self.state_indices["q5"]] = state_dict["dq5"]
        derivatives[self.state_indices["q6"]] = state_dict["dq6"]
        
        # Velocity derivatives (simplified implementation)
        derivatives[self.state_indices["dq1"]] = (tau1 - self.params["d1"]*state_dict["dq1"] - 
                                            self.params["g"]*self.params["m1"]*self.params["l1"]*np.sin(state_dict["q1"])) / self.params["I1"]
        derivatives[self.state_indices["dq2"]] = (tau2 - self.params["d2"]*state_dict["dq2"] - 
                                            self.params["g"]*self.params["m2"]*self.params["l2"]*np.sin(state_dict["q2"])) / self.params["I2"]
        derivatives[self.state_indices["dq3"]] = (tau3 - self.params["d3"]*state_dict["dq3"] - 
                                            self.params["g"]*self.params["m3"]*self.params["l3"]*np.sin(state_dict["q3"])) / self.params["I3"]
        derivatives[self.state_indices["dq4"]] = (tau4 - self.params["d4"]*state_dict["dq4"]) / self.params["I4"]
        derivatives[self.state_indices["dq5"]] = (tau5 - self.params["d5"]*state_dict["dq5"]) / self.params["I5"]
        derivatives[self.state_indices["dq6"]] = (tau6 - self.params["d6"]*state_dict["dq6"]) / self.params["I6"]
        
        return derivatives
    
    def joint_pd_controller(self, state, target):
        \"\"\"
        JointPD controller implementation.
        
        Args:
            state: Current system state
            target: Target joint positions
            
        Returns:
            Control input vector
        \"\"\"
        # Controller parameters
        k_p = [10, 10, 10, 8, 5, 3]
        k_d = [2, 2, 2, 1.5, 1, 0.5]
        
        # Extract state
        q = [state[self.state_indices[f"q{i+1}"]] for i in range(6)]
        dq = [state[self.state_indices[f"dq{i+1}"]] for i in range(6)]
        
        # Compute control inputs
        control = np.zeros(6)
        
        # Joint 1-3 (with gravity compensation)
        for i in range(3):
            control[i] = k_p[i]*(target[i]-q[i]) + k_d[i]*(-dq[i])
            
        # Gravity compensation for joints 1-3
        control[0] += self.params["g"]*self.params["m1"]*self.params["l1"]*np.sin(q[0])
        control[1] += self.params["g"]*self.params["m2"]*self.params["l2"]*np.sin(q[1])
        control[2] += self.params["g"]*self.params["m3"]*self.params["l3"]*np.sin(q[2])
        
        # Joint 4-6 (no gravity compensation needed)
        for i in range(3, 6):
            control[i] = k_p[i]*(target[i]-q[i]) + k_d[i]*(-dq[i])
        
        return control
    
    def human_collab_controller(self, state, target, human_factor=0.5):
        \"\"\"
        Human collaboration controller implementation.
        
        Args:
            state: Current system state
            target: Target joint positions
            human_factor: Collaboration factor (0-1)
            
        Returns:
            Control input vector
        \"\"\"
        # Base controller gains
        k_p = [10, 10, 10, 8, 5, 3]
        k_d = [2, 2, 2, 1.5, 1, 0.5]
        
        # Adjust gains based on human factor
        k_p_adjusted = [kp * human_factor for kp in k_p]
        k_d_adjusted = [kd * (2-human_factor) for kd in k_d]
        
        # Extract state
        q = [state[self.state_indices[f"q{i+1}"]] for i in range(6)]
        dq = [state[self.state_indices[f"dq{i+1}"]] for i in range(6)]
        
        # Compute control inputs
        control = np.zeros(6)
        
        # Apply adjusted gains
        for i in range(6):
            control[i] = k_p_adjusted[i]*(target[i]-q[i]) + k_d_adjusted[i]*(-dq[i])
        
        # Gravity compensation for joints 1-3
        control[0] += self.params["g"]*self.params["m1"]*self.params["l1"]*np.sin(q[0])
        control[1] += self.params["g"]*self.params["m2"]*self.params["l2"]*np.sin(q[1])
        control[2] += self.params["g"]*self.params["m3"]*self.params["l3"]*np.sin(q[2])
        
        return control
    
    def force_hybrid_controller(self, state, target, F_d=5.0, F_meas=0.0):
        \"\"\"
        Force hybrid controller implementation.
        
        Args:
            state: Current system state
            target: Target joint positions
            F_d: Desired force
            F_meas: Measured force
            
        Returns:
            Control input vector
        \"\"\"
        # Controller parameters
        k_p = [10, 10, 0, 8, 5, 3]  # No position control for joint 3
        k_d = [2, 2, 0, 1.5, 1, 0.5]  # No velocity damping for joint 3
        k_f = 0.1  # Force control gain
        
        # Extract state
        q = [state[self.state_indices[f"q{i+1}"]] for i in range(6)]
        dq = [state[self.state_indices[f"dq{i+1}"]] for i in range(6)]
        
        # Compute control inputs
        control = np.zeros(6)
        
        # Position control for joints 1-2, 4-6
        for i in [0, 1, 3, 4, 5]:
            control[i] = k_p[i]*(target[i]-q[i]) + k_d[i]*(-dq[i])
        
        # Force control for joint 3
        control[2] = k_f * (F_d - F_meas)
        
        # Gravity compensation
        control[0] += self.params["g"]*self.params["m1"]*self.params["l1"]*np.sin(q[0])
        control[1] += self.params["g"]*self.params["m2"]*self.params["l2"]*np.sin(q[1])
        control[2] += self.params["g"]*self.params["m3"]*self.params["l3"]*np.sin(q[2])
        
        return control
    
    def compute_barrier_value(self, state, barrier_name):
        \"\"\"
        Compute the value of a barrier function.
        
        Args:
            state: Current system state
            barrier_name: Name of the barrier function
            
        Returns:
            Barrier function value
        \"\"\"
        # Extract state
        state_dict = {}
        for var, idx in self.state_indices.items():
            state_dict[var] = state[idx]
        
        # Compute barrier value based on name
        if barrier_name == "JointLimits":
            # Joint limits barrier
            q1_min, q1_max = -2.0, 2.0
            q2_min, q2_max = -1.5, 1.5
            q3_min, q3_max = -2.5, 2.5
            q4_min, q4_max = -1.8, 1.8
            q5_min, q5_max = -1.5, 1.5
            q6_min, q6_max = -3.0, 3.0
            
            return ((q1_max-state_dict["q1"])*(state_dict["q1"]-q1_min) * 
                   (q2_max-state_dict["q2"])*(state_dict["q2"]-q2_min) *
                   (q3_max-state_dict["q3"])*(state_dict["q3"]-q3_min) *
                   (q4_max-state_dict["q4"])*(state_dict["q4"]-q4_min) *
                   (q5_max-state_dict["q5"])*(state_dict["q5"]-q5_min) *
                   (q6_max-state_dict["q6"])*(state_dict["q6"]-q6_min))
        
        elif barrier_name == "SelfCollision":
            # Self collision barrier
            d_min = 0.1
            l2 = self.params["l2"]
            l4 = self.params["l4"]
            
            return d_min**2 - (l2*np.sin(state_dict["q2"])-l4*np.sin(state_dict["q4"]))**2 - (l2*np.cos(state_dict["q2"])-l4*np.cos(state_dict["q4"]))**2
        
        elif barrier_name == "Workspace":
            # Workspace limits barrier
            r_max = 1.0
            l1 = self.params["l1"]
            l2 = self.params["l2"]
            l3 = self.params["l3"]
            
            return r_max**2 - (l1*np.cos(state_dict["q1"])+l2*np.cos(state_dict["q1"]+state_dict["q2"])+l3*np.cos(state_dict["q1"]+state_dict["q2"]+state_dict["q3"]))**2 - (l1*np.sin(state_dict["q1"])+l2*np.sin(state_dict["q1"]+state_dict["q2"])+l3*np.sin(state_dict["q1"]+state_dict["q2"]+state_dict["q3"]))**2
        
        elif barrier_name == "HumanSafety":
            # Human safety barrier
            x_h = 0.5
            y_h = 0.5
            d_safe = 0.3
            l1 = self.params["l1"]
            l2 = self.params["l2"]
            
            return (x_h-(l1*np.cos(state_dict["q1"])+l2*np.cos(state_dict["q1"]+state_dict["q2"])))**2 + (y_h-(l1*np.sin(state_dict["q1"])+l2*np.sin(state_dict["q1"]+state_dict["q2"])))**2 - d_safe**2
        
        else:
            return 0.0
    
    def simulate(self, initial_state, control_mode, target_positions, duration, dt=0.01):
        \"\"\"
        Run a simulation with the specified parameters.
        
        Args:
            initial_state: Initial state vector
            control_mode: Controller to use ('JointPD', 'HumanCollab', 'ForceHybrid')
            target_positions: Target joint positions
            duration: Simulation duration in seconds
            dt: Time step
            
        Returns:
            Dictionary of simulation results
        \"\"\"
        # Number of simulation steps
        steps = int(duration / dt)
        
        # Initialize arrays for storing results
        t_values = np.linspace(0, duration, steps+1)
        state_history = np.zeros((steps+1, len(initial_state)))
        state_history[0] = initial_state
        
        control_history = np.zeros((steps+1, 6))
        barrier_history = {
            "JointLimits": np.zeros(steps+1),
            "Workspace": np.zeros(steps+1),
            "SelfCollision": np.zeros(steps+1),
            "HumanSafety": np.zeros(steps+1)
        }
        
        # Compute initial barrier values
        for barrier_name in barrier_history:
            barrier_history[barrier_name][0] = self.compute_barrier_value(initial_state, barrier_name)
            
        # Select controller based on mode
        if control_mode == 'JointPD':
            controller_fn = self.joint_pd_controller
        elif control_mode == 'HumanCollab':
            controller_fn = self.human_collab_controller
        elif control_mode == 'ForceHybrid':
            controller_fn = self.force_hybrid_controller
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")
        
        # Run simulation
        for i in range(1, steps+1):
            # Compute control input
            control = controller_fn(state_history[i-1], target_positions)
            control_history[i] = control
            
            # Update state using simple Euler integration
            state_derivatives = self.dynamics(state_history[i-1], t_values[i], control)
            state_history[i] = state_history[i-1] + state_derivatives * dt
            
            # Compute barrier values
            for barrier_name in barrier_history:
                barrier_history[barrier_name][i] = self.compute_barrier_value(state_history[i], barrier_name)
        
        # Return results
        return {
            't': t_values,
            'state': state_history,
            'control': control_history,
            'barriers': barrier_history
        }
