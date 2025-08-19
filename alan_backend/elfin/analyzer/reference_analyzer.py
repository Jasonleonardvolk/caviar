#!/usr/bin/env python3
"""
ELFIN Reference Analyzer

Enhanced static analyzer for ELFIN specification files, with focus on:
- Circular reference detection
- Derivative and state tracking
- Alias detection
"""

import os
import re
import json
import sys
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union


class ELFINAnalyzer:
    """Enhanced analyzer for ELFIN specification files."""
    
    def __init__(self, file_path: str):
        """Initialize with path to ELFIN file."""
        self.file_path = file_path
        self.content = ""
        self.sections = {}
        self.symbols = {}
        self.references = {}
        self.issues = []
        
        # Track derivatives using multiple patterns
        self.derivative_patterns = [
            # Standard patterns for derivatives
            (r'(\w+)_dot', lambda x: x),         # x_dot -> x
            (r'd(\w+)', lambda x: x),            # dx -> x, dq1 -> q1
            (r'(\w+)_prime', lambda x: x),       # x_prime -> x
            (r'(\w+)dt', lambda x: x),           # xdt -> x
            (r'dot_(\w+)', lambda x: x),         # dot_x -> x
            (r'derivative_(\w+)', lambda x: x),  # derivative_x -> x
        ]
        
        # Keep track of potential aliases 
        self.potential_aliases = {}
        
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
        # Support both colon and non-colon syntax styles
        section_pattern = r'(?:helpers|system|lyapunov|barrier|mode)\s*:?\s*([A-Za-z0-9_]+)\s*{([^}]*)}'
        for match in re.finditer(section_pattern, self.content, re.DOTALL):
            section_type = match.group(0).split()[0].rstrip(':')  # Remove colon if present
            section_name = match.group(1)
            section_content = match.group(2)
            
            key = f"{section_type}:{section_name}"
            self.sections[key] = section_content.strip()
            print(f"Found section: {key}")
    
    def extract_symbols(self) -> None:
        """
        Extract symbol definitions from all sections using direct string parsing.
        
        NOTE: This implementation has been deprecated in favor of the more focused 
        circular_analyzer.py, which uses a more robust approach for circular reference detection.
        This remains as a template for future enhancements beyond circular references.
        """
        # For the time being, delegate to the simpler implementation
        print("\nNOTE: Using simplified parsing approach from circular_analyzer.py")
        
        # For each line in the file, look for assignments
        lines = self.content.split('\n')
        for line in lines:
            # Remove comments
            if '//' in line:
                line = line[:line.find('//')]
            
            line = line.strip()
            if not line:
                continue
            
            # Look for assignment patterns: variable = expression;
            if '=' in line and not line.startswith('if') and not line.startswith('for'):
                # Skip comparison operators
                if ' == ' in line or ' != ' in line or ' <= ' in line or ' >= ' in line:
                    continue
                
                # Extract variable and expression
                parts = line.split('=', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()
                
                # Remove trailing semicolon if present
                if expr.endswith(';'):
                    expr = expr[:-1].strip()
                
                # Skip function definitions
                if '(' in var_name:
                    continue
                
                # Store the variable and its expression
                if var_name:
                    section_key = next(iter(self.sections.keys())) if self.sections else "unknown:unknown"
                    self.symbols[var_name] = {
                        'type': 'expression',
                        'section': section_key,
                        'expression': expr
                    }
                    print(f"  Found assignment: {var_name} = {expr}")
        
        return
        
        # The rest of this function is kept for reference but not used
        for section_key, section_content in self.sections.items():
            section_type, section_name = section_key.split(':')
            print(f"Processing section: {section_type} {section_name}")
            
            # Extract different symbol types based on section
            if section_type == 'system':
                # Extract state variables - direct string approach
                cs_start = section_content.find("continuous_state")
                if cs_start >= 0:
                    print(f"  Found continuous_state block")
                    cs_open = section_content.find("{", cs_start)
                    if cs_open >= 0:
                        cs_close = section_content.find("}", cs_open)
                        if cs_close >= 0:
                            cs_content = section_content[cs_open+1:cs_close]
                            # Split by semicolons
                            for var_entry in cs_content.split(';'):
                                # Remove comments
                                if '//' in var_entry:
                                    var_entry = var_entry[:var_entry.find('//')]
                                
                                var_name = var_entry.strip()
                                if var_name:
                                    self.symbols[var_name] = {
                                        'type': 'state',
                                        'section': section_key
                                    }
                                    print(f"    Found state var: {var_name}")
                
                # Extract input variables - direct string approach
                input_start = section_content.find("input")
                if input_start >= 0:
                    print(f"  Found input block")
                    input_open = section_content.find("{", input_start)
                    if input_open >= 0:
                        input_close = section_content.find("}", input_open)
                        if input_close >= 0:
                            input_content = section_content[input_open+1:input_close]
                            # Split by semicolons
                            for var_entry in input_content.split(';'):
                                # Remove comments
                                if '//' in var_entry:
                                    var_entry = var_entry[:var_entry.find('//')]
                                
                                var_name = var_entry.strip()
                                if var_name:
                                    self.symbols[var_name] = {
                                        'type': 'input',
                                        'section': section_key
                                    }
                                    print(f"    Found input var: {var_name}")
                
                # Extract parameters - direct string approach
                params_start = section_content.find("params")
                if params_start >= 0:
                    print(f"  Found params block")
                    params_open = section_content.find("{", params_start)
                    if params_open >= 0:
                        params_close = section_content.find("}", params_open)
                        if params_close >= 0:
                            params_content = section_content[params_open+1:params_close]
                            # Process each line
                            for line in params_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if ':' in line and ';' in line:
                                    param_name = line[:line.find(':')].strip()
                                    param_value = line[line.find(':')+1:line.find(';')].strip()
                                    if param_name:
                                        self.symbols[param_name] = {
                                            'type': 'param',
                                            'section': section_key,
                                            'value': param_value
                                        }
                                        print(f"    Found param: {param_name} = {param_value}")
                
                # Extract dynamics variables - direct string approach
                fd_start = section_content.find("flow_dynamics")
                if fd_start >= 0:
                    print(f"  Found flow_dynamics block")
                    fd_open = section_content.find("{", fd_start)
                    if fd_open >= 0:
                        fd_close = section_content.find("}", fd_open)
                        if fd_close >= 0:
                            fd_content = section_content[fd_open+1:fd_close]
                            # Process each line
                            for line in fd_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if '=' in line:
                                    parts = line.split('=', 1)
                                    var_name = parts[0].strip()
                                    expr = parts[1].strip()
                                    if expr.endswith(';'):
                                        expr = expr[:-1].strip()
                                    
                                    self.symbols[var_name] = {
                                        'type': 'dynamic',
                                        'section': section_key,
                                        'expression': expr
                                    }
                                    print(f"    Found dynamic var: {var_name} = {expr}")
                                    
                                    # Track state-derivative relationships
                                    for pattern, transform in self.derivative_patterns:
                                        match = re.match(pattern, var_name)
                                        if match:
                                            base_var = transform(match.group(1))
                                            if base_var in self.symbols and self.symbols[base_var]['type'] == 'state':
                                                self.symbols[var_name]['derivative_of'] = base_var
                                                print(f"    Derived from state: {base_var}")
                                                break
            
            elif section_type == 'helpers':
                # Extract helper functions - direct string approach
                print(f"  Processing helpers section")
                # Parse each line directly
                for line in section_content.strip().split('\n'):
                    # Skip comments or empty lines
                    if not line.strip() or line.strip().startswith('//'):
                        continue
                    
                    # Look for helper function definitions like: name(args) = expr;
                    if '(' in line and ')' in line and '=' in line and ';' in line:
                        # Extract function name
                        fn_name = line[:line.find('(')].strip()
                        if fn_name:
                            # Extract arguments
                            args_start = line.find('(') + 1
                            args_end = line.find(')', args_start)
                            if args_end > args_start:
                                args_str = line[args_start:args_end].strip()
                                args_list = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                                
                                # Extract expression
                                expr_start = line.find('=', args_end) + 1
                                expr_end = line.find(';', expr_start)
                                if expr_end > expr_start:
                                    expr = line[expr_start:expr_end].strip()
                                    
                                    # Add to symbols
                                    self.symbols[fn_name] = {
                                        'type': 'helper',
                                        'section': section_key,
                                        'args': args_list,
                                        'expression': expr
                                    }
                                    print(f"    Found helper function: {fn_name}({', '.join(args_list)}) = {expr}")
            
            elif section_type in ['lyapunov', 'barrier']:
                # Extract V or B expressions and parameters - direct string approach
                params_start = section_content.find("params")
                if params_start >= 0:
                    print(f"  Found {section_type} params block")
                    params_open = section_content.find("{", params_start)
                    if params_open >= 0:
                        params_close = section_content.find("}", params_open)
                        if params_close >= 0:
                            params_content = section_content[params_open+1:params_close]
                            # Process each line
                            for line in params_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if ':' in line and ';' in line:
                                    param_name = line[:line.find(':')].strip()
                                    param_value = line[line.find(':')+1:line.find(';')].strip()
                                    if param_name:
                                        symbol_key = f"{section_name}.{param_name}"
                                        self.symbols[symbol_key] = {
                                            'type': 'param',
                                            'section': section_key,
                                            'value': param_value
                                        }
                                        print(f"    Found {section_type} param: {symbol_key} = {param_value}")
                
                # Extract main function (V for lyapunov, B for barrier) - direct string approach
                main_var = 'V' if section_type == 'lyapunov' else 'B'
                var_assign = f"{main_var} ="
                var_assign_pos = section_content.find(var_assign)
                if var_assign_pos >= 0:
                    # Find the end of the expression (semicolon)
                    expr_start = var_assign_pos + len(var_assign)
                    expr_end = section_content.find(';', expr_start)
                    if expr_end >= 0:
                        expr = section_content[expr_start:expr_end].strip()
                        self.symbols[f"{section_name}.{main_var}"] = {
                            'type': 'expression',
                            'section': section_key,
                            'expression': expr
                        }
                        print(f"    Found {section_type} expression: {main_var} = {expr}")
                
                # Extract alpha function for barriers - direct string approach
                if section_type == 'barrier':
                    alpha_vars = ['alphaFun', 'alpha_fun']
                    for alpha_var in alpha_vars:
                        var_assign = f"{alpha_var} ="
                        var_assign_pos = section_content.find(var_assign)
                        if var_assign_pos >= 0:
                            # Find the end of the expression (semicolon)
                            expr_start = var_assign_pos + len(var_assign)
                            expr_end = section_content.find(';', expr_start)
                            if expr_end >= 0:
                                expr = section_content[expr_start:expr_end].strip()
                                self.symbols[f"{section_name}.{alpha_var}"] = {
                                    'type': 'expression',
                                    'section': section_key,
                                    'expression': expr
                                }
                                print(f"    Found barrier alpha expression: {alpha_var} = {expr}")
                                break
            
            elif section_type == 'mode':
                # Extract controller expressions and parameters - direct string approach
                params_start = section_content.find("params")
                if params_start >= 0:
                    print(f"  Found mode params block")
                    params_open = section_content.find("{", params_start)
                    if params_open >= 0:
                        params_close = section_content.find("}", params_open)
                        if params_close >= 0:
                            params_content = section_content[params_open+1:params_close]
                            # Process each line
                            for line in params_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if ':' in line and ';' in line:
                                    param_name = line[:line.find(':')].strip()
                                    param_value = line[line.find(':')+1:line.find(';')].strip()
                                    if param_name:
                                        symbol_key = f"{section_name}.{param_name}"
                                        self.symbols[symbol_key] = {
                                            'type': 'param',
                                            'section': section_key,
                                            'value': param_value
                                        }
                                        print(f"    Found mode param: {symbol_key} = {param_value}")
                
                # Extract controller expressions - direct string approach
                controller_start = section_content.find("controller")
                if controller_start >= 0:
                    print(f"  Found controller block")
                    controller_open = section_content.find("{", controller_start)
                    if controller_open >= 0:
                        controller_close = section_content.find("}", controller_open)
                        if controller_close >= 0:
                            controller_content = section_content[controller_open+1:controller_close]
                            # Process each line
                            for line in controller_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if '=' in line:
                                    # Extract variable name and expression
                                    parts = line.split('=', 1)
                                    var_name = parts[0].strip()
                                    expr = parts[1].strip()
                                    if expr.endswith(';'):
                                        expr = expr[:-1].strip()
                                    
                                    # Store as both qualified and unqualified name
                                    if var_name:
                                        symbol_key = f"{section_name}.{var_name}"
                                        self.symbols[symbol_key] = {
                                            'type': 'controller',
                                            'section': section_key,
                                            'expression': expr
                                        }
                                        print(f"    Found controller expr: {symbol_key} = {expr}")
                                        
                                        # Also add as direct variable for unqualified reference
                                        self.symbols[var_name] = {
                                            'type': 'controller_alias',
                                            'section': section_key,
                                            'expression': expr,
                                            'original': symbol_key
                                        }
    
    def analyze_references(self) -> None:
        """Analyze symbol references in expressions."""
        # Extract all potential identifiers from expressions
        for symbol, info in self.symbols.items():
            if 'expression' in info:
                # Find potential variable references in the expression
                # This regex matches potential variable names
                identifiers = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', info['expression'])
                
                # Store references but filter out numeric functions and constants
                common_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 
                                   'max', 'min', 'if', 'then', 'else', 'pi', 'e'}
                
                references = set()
                for identifier in identifiers:
                    if (
                        identifier not in common_functions and
                        identifier not in info.get('args', [])  # Skip function arguments
                    ):
                        references.add(identifier)
                
                self.references[symbol] = references
                
                # Check for potential aliases (two variables with identical expressions)
                expression_key = info['expression'].strip()
                if expression_key not in self.potential_aliases:
                    self.potential_aliases[expression_key] = []
                self.potential_aliases[expression_key].append(symbol)
    
    def check_for_circular_references(self) -> None:
        """Detect circular references in dynamics definitions."""
        # Build a dependency graph
        dependency_graph = {}
        for var, info in self.symbols.items():
            if 'expression' in info:
                dependency_graph[var] = set()
                for ref in self.references.get(var, set()):
                    dependency_graph[var].add(ref)
        
        # Detect direct circular references (e.g., dx = dx)
        for var, info in self.symbols.items():
            if 'expression' in info:
                # Check for exact self-reference
                pattern = r'\b' + re.escape(var) + r'\b'
                if re.search(pattern, info['expression']):
                    self.issues.append({
                        'type': 'circular_reference',
                        'severity': 'error',
                        'message': f"Direct circular reference: {var} depends on itself",
                        'variable': var,
                        'expression': info['expression']
                    })
        
        # Detect indirect circular references through path traversal
        def has_cycle(node, path=None, visited=None):
            if path is None:
                path = []
            if visited is None:
                visited = set()
                
            path.append(node)
            visited.add(node)
            
            for neighbor in dependency_graph.get(node, set()):
                if neighbor in path:
                    # Found a cycle
                    cycle_path = path[path.index(neighbor):] + [neighbor]
                    return True, cycle_path
                if neighbor not in visited:
                    result, cycle_path = has_cycle(neighbor, path.copy(), visited)
                    if result:
                        return True, cycle_path
            
            return False, []
        
        # Check each node for cycles
        for var in dependency_graph:
            has_cycle_detected, cycle_path = has_cycle(var)
            if has_cycle_detected:
                self.issues.append({
                    'type': 'circular_reference',
                    'severity': 'error',
                    'message': f"Indirect circular reference detected: {' -> '.join(cycle_path)}",
                    'variable': var,
                    'cycle_path': cycle_path
                })
    
    def check_for_derivative_consistency(self) -> None:
        """Verify that derivative relationships are consistent."""
        # Group variables by their base name using derivative patterns
        base_to_derivs = {}
        
        # First pass: Identify all base-derivative relationships
        for var, info in self.symbols.items():
            # Skip variables that are already identified as derivatives
            if 'derivative_of' in info:
                base_var = info['derivative_of']
                if base_var not in base_to_derivs:
                    base_to_derivs[base_var] = []
                base_to_derivs[base_var].append(var)
                continue
                
            # Check if this could be a derivative variable
            for pattern, transform in self.derivative_patterns:
                match = re.match(pattern, var)
                if match:
                    base_var = transform(match.group(1))
                    # Only consider it a derivative if the base variable exists
                    if base_var in self.symbols:
                        if base_var not in base_to_derivs:
                            base_to_derivs[base_var] = []
                        base_to_derivs[base_var].append(var)
                        # Store the relationship
                        self.symbols[var]['potential_derivative_of'] = base_var
                        break
        
        # Check for duplicate derivatives (multiple variables deriving the same base)
        for base_var, derivatives in base_to_derivs.items():
            if len(derivatives) > 1:
                # Multiple derivatives found for the same base variable
                self.issues.append({
                    'type': 'duplicate_derivative',
                    'severity': 'warning',
                    'message': f"Multiple derivatives defined for {base_var}: {', '.join(derivatives)}",
                    'variable': base_var,
                    'derivatives': derivatives
                })
                
        # Check dynamics equations - is every state variable's derivative defined?
        for var, info in self.symbols.items():
            if info['type'] == 'state' and var not in base_to_derivs:
                self.issues.append({
                    'type': 'missing_derivative',
                    'severity': 'error',
                    'message': f"State variable {var} has no defined derivative equation",
                    'variable': var
                })
    
    def detect_potential_aliases(self) -> None:
        """Detect variables that might be aliases of each other based on identical expressions."""
        for expr, vars_with_expr in self.potential_aliases.items():
            if len(vars_with_expr) > 1 and len(expr) > 5:  # Only report non-trivial expressions
                # Check that the variables are not derivatives of each other
                derivatives = set()
                for var in vars_with_expr:
                    info = self.symbols[var]
                    if 'derivative_of' in info:
                        derivatives.add(info['derivative_of'])
                    if 'potential_derivative_of' in info:
                        derivatives.add(info['potential_derivative_of'])
                
                # Filter out variables that are derivatives
                non_derivative_vars = [v for v in vars_with_expr if v not in derivatives]
                
                if len(non_derivative_vars) > 1:
                    self.issues.append({
                        'type': 'potential_alias',
                        'severity': 'warning',
                        'message': f"Potential aliases detected: {', '.join(non_derivative_vars)} have identical expressions",
                        'variables': non_derivative_vars,
                        'expression': expr
                    })
    
    def validate_references(self) -> None:
        """Validate that all referenced symbols are defined."""
        for symbol, references in self.references.items():
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
        dynamics_vars = set()
        
        # Find all variables that have a derivative defined
        for var, refs in self.references.items():
            if 'derivative_of' in self.symbols.get(var, {}):
                base_var = self.symbols[var]['derivative_of']
                dynamics_vars.add(base_var)
            elif 'potential_derivative_of' in self.symbols.get(var, {}):
                base_var = self.symbols[var]['potential_derivative_of']
                dynamics_vars.add(base_var)
        
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
        self.check_for_derivative_consistency()
        self.detect_potential_aliases()
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
            for i, issue in enumerate(self.issues):
                severity_marker = "ERROR" if issue['severity'] == 'error' else "WARNING"
                print(f"[{severity_marker}] {issue['message']}")
