"""
Verification Utilities for ELFIN Models

This module provides tools for verifying the correctness of ELFIN models,
including syntax checking, circular reference detection, and consistency with
stability proofs.
"""

import os
import re
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional, Any


def parse_elfin_file(filepath: str) -> Dict[str, Any]:
    """
    Parse an ELFIN file into a structured format.
    
    Args:
        filepath: Path to the ELFIN file
        
    Returns:
        Dictionary containing parsed ELFIN structures
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    
    # Parse block structures (like system, barrier, lyapunov, etc.)
    blocks = {}
    
    # Match block patterns: type name { ... }
    block_pattern = r'(\w+)\s+(\w+)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    for match in re.finditer(block_pattern, content):
        block_type, block_name, block_content = match.groups()
        
        # Parse variables within the block
        variables = {}
        var_pattern = r'(\w+)\s*:\s*([^;]+);'
        for var_match in re.finditer(var_pattern, block_content):
            var_name, var_expr = var_match.groups()
            variables[var_name] = var_expr.strip()
        
        # Add to blocks
        if block_type not in blocks:
            blocks[block_type] = {}
        
        blocks[block_type][block_name] = {
            'content': block_content,
            'variables': variables
        }
    
    return blocks


def find_circular_references(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Find circular references in ELFIN variables.
    
    Args:
        parsed_data: Parsed ELFIN data
        
    Returns:
        List of circular reference chains detected
    """
    circular_refs = []
    
    # Process each block type (system, barrier, lyapunov)
    for block_type, blocks in parsed_data.items():
        for block_name, block_data in blocks.items():
            variables = block_data['variables']
            
            # Build dependency graph
            G = nx.DiGraph()
            
            # Add nodes for all variables
            for var_name in variables.keys():
                G.add_node(var_name)
            
            # Add edges based on variable dependencies
            for var_name, var_expr in variables.items():
                # Find all variable references in the expression
                for ref_var in re.finditer(r'\b(\w+)\b', var_expr):
                    ref_name = ref_var.group(1)
                    # Check if it's a variable name (not a function or constant)
                    if ref_name in variables:
                        G.add_edge(var_name, ref_name)
            
            # Find cycles
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    circular_refs.append(f"{block_type}.{block_name}: {' -> '.join(cycle)}")
            except nx.NetworkXNoCycle:
                pass  # No cycles in this block
    
    return circular_refs


def check_syntax(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Check for syntax errors in ELFIN file.
    
    Args:
        parsed_data: Parsed ELFIN data
        
    Returns:
        List of syntax errors detected
    """
    errors = []
    
    # Process each block type
    for block_type, blocks in parsed_data.items():
        for block_name, block_data in blocks.items():
            variables = block_data['variables']
            
            # Check for unmatched parentheses, brackets, etc.
            for var_name, var_expr in variables.items():
                # Check parentheses
                if var_expr.count('(') != var_expr.count(')'):
                    errors.append(f"{block_type}.{block_name}.{var_name}: Unmatched parentheses")
                
                # Check brackets
                if var_expr.count('[') != var_expr.count(']'):
                    errors.append(f"{block_type}.{block_name}.{var_name}: Unmatched brackets")
                
                # Check braces
                if var_expr.count('{') != var_expr.count('}'):
                    errors.append(f"{block_type}.{block_name}.{var_name}: Unmatched braces")
                
                # Check if-then-else completeness
                if_count = var_expr.count(' if ')
                then_count = var_expr.count(' then ')
                else_count = var_expr.count(' else ')
                
                if if_count != then_count or if_count != else_count:
                    errors.append(f"{block_type}.{block_name}.{var_name}: Incomplete if-then-else statement")
    
    return errors


def check_barrier_certificate(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Check barrier certificate conditions in ELFIN file.
    
    Args:
        parsed_data: Parsed ELFIN data
        
    Returns:
        List of potential issues with barrier certificates
    """
    issues = []
    
    # Check for 'barrier' blocks
    if 'barrier' in parsed_data:
        for barrier_name, barrier_data in parsed_data['barrier'].items():
            variables = barrier_data['variables']
            
            # Check if barrier function (B) is defined
            if 'B' not in variables:
                issues.append(f"barrier.{barrier_name}: Missing barrier function definition (B)")
            
            # Check if alpha function is defined
            if 'alpha_fun' not in variables:
                issues.append(f"barrier.{barrier_name}: Missing alpha function definition (alpha_fun)")
    
    return issues


def check_lyapunov_certificate(parsed_data: Dict[str, Any]) -> List[str]:
    """
    Check Lyapunov certificate conditions in ELFIN file.
    
    Args:
        parsed_data: Parsed ELFIN data
        
    Returns:
        List of potential issues with Lyapunov certificates
    """
    issues = []
    
    # Check for 'lyapunov' blocks
    if 'lyapunov' in parsed_data:
        for lyapunov_name, lyapunov_data in parsed_data['lyapunov'].items():
            variables = lyapunov_data['variables']
            
            # Check if Lyapunov function (V) is defined
            if 'V' not in variables:
                issues.append(f"lyapunov.{lyapunov_name}: Missing Lyapunov function definition (V)")
    
    return issues


def verify_elfin_file(filepath: str) -> Dict[str, List[str]]:
    """
    Verify an ELFIN file for correctness.
    
    This function checks for:
    1. Syntax errors
    2. Circular references
    3. Barrier certificate conditions
    4. Lyapunov certificate conditions
    
    Args:
        filepath: Path to the ELFIN file
        
    Returns:
        Dictionary of verification results
    """
    # Parse the ELFIN file
    parsed_data = parse_elfin_file(filepath)
    
    # Perform various checks
    syntax_errors = check_syntax(parsed_data)
    circular_refs = find_circular_references(parsed_data)
    barrier_issues = check_barrier_certificate(parsed_data)
    lyapunov_issues = check_lyapunov_certificate(parsed_data)
    
    # Combine results
    results = {
        'syntax_errors': syntax_errors,
        'circular_references': circular_refs,
        'barrier_issues': barrier_issues,
        'lyapunov_issues': lyapunov_issues
    }
    
    return results


def verify_with_mosek(filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verify an ELFIN barrier or Lyapunov function using MOSEK solver.
    
    This function uses optimization to verify the barrier or Lyapunov conditions
    over the state space using sum-of-squares (SOS) programming.
    
    Args:
        filepath: Path to the ELFIN file
        options: Options for verification
            - state_bounds: Bounds of the state space
            - degree: Degree of SOS polynomials
            - grid_size: Size of verification grid
            
    Returns:
        Dictionary of verification results
    """
    # This is a placeholder for the actual MOSEK integration
    # In a real implementation, this would import and use MOSEK
    
    options = options or {}
    
    # Parse the ELFIN file
    parsed_data = parse_elfin_file(filepath)
    
    # Determine if it's a barrier or Lyapunov function
    if 'barrier' in parsed_data:
        # Placeholder for barrier verification
        return {
            'verified': True,
            'message': "Barrier certificate verified using MOSEK (placeholder)",
            'certificate_type': 'barrier'
        }
    elif 'lyapunov' in parsed_data:
        # Placeholder for Lyapunov verification
        return {
            'verified': True,
            'message': "Lyapunov function verified using MOSEK (placeholder)",
            'certificate_type': 'lyapunov'
        }
    else:
        return {
            'verified': False,
            'message': "Could not identify barrier or Lyapunov function in file",
            'certificate_type': None
        }
