#!/usr/bin/env python3
"""
ELFIN Circular Reference Analyzer

A focused implementation for detecting circular references in ELFIN files.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class CircularReferenceAnalyzer:
    """
    A focused analyzer that detects circular references in ELFIN files.
    """
    
    def __init__(self, file_path: str):
        """Initialize the analyzer with file path."""
        self.file_path = file_path
        self.content = ""
        self.variables = {}  # Variable name -> expression
        self.dependency_graph = {}  # Variable name -> set of variables it depends on
        self.issues = []
    
    def load_file(self) -> None:
        """Load the ELFIN file content."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"Loaded file: {self.file_path} ({len(self.content)} bytes)")
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
    
    def extract_assignments(self) -> None:
        """
        Extract all assignment expressions from the file.
        Looks for patterns like 'variable = expression;'
        """
        # Process the file line by line to handle comments properly
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
                    self.variables[var_name] = expr
                    print(f"Found assignment: {var_name} = {expr}")
    
    def build_dependency_graph(self) -> None:
        """
        Build a dependency graph of variables.
        Each variable points to a set of variables it depends on.
        """
        # For each variable, find all other variables it references
        for var_name, expr in self.variables.items():
            dependencies = set()
            
            # Find all potential variable references in the expression
            # Look for word boundaries to avoid partial matches
            for dep_var in self.variables.keys():
                # Use word boundary pattern to match whole words only
                pattern = r'\b' + re.escape(dep_var) + r'\b'
                if re.search(pattern, expr):
                    dependencies.add(dep_var)
            
            self.dependency_graph[var_name] = dependencies
            print(f"Dependencies for {var_name}: {dependencies}")
    
    def check_direct_circular_references(self) -> None:
        """
        Check for direct circular references (e.g., x = x + 1).
        """
        for var_name, expr in self.variables.items():
            # Check if the variable directly references itself
            pattern = r'\b' + re.escape(var_name) + r'\b'
            if re.search(pattern, expr):
                self.issues.append({
                    'type': 'circular_reference',
                    'severity': 'error',
                    'message': f"Direct circular reference: {var_name} depends on itself",
                    'variable': var_name,
                    'expression': expr
                })
                print(f"Found direct circular reference: {var_name} = {expr}")
    
    def check_indirect_circular_references(self) -> None:
        """
        Check for indirect circular references (e.g., a = b, b = c, c = a).
        Uses depth-first search to find cycles in the dependency graph.
        """
        def has_cycle(node, path=None, visited=None):
            if path is None:
                path = []
            if visited is None:
                visited = set()
            
            path.append(node)
            visited.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
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
        for var in self.dependency_graph:
            has_cycle_detected, cycle_path = has_cycle(var)
            if has_cycle_detected:
                # Only add the issue if it's a real cycle (length > 1)
                # Direct cycles are handled by check_direct_circular_references
                if len(cycle_path) > 1:
                    self.issues.append({
                        'type': 'circular_reference',
                        'severity': 'error',
                        'message': f"Indirect circular reference detected: {' -> '.join(cycle_path)}",
                        'variable': var,
                        'cycle_path': cycle_path
                    })
                    print(f"Found indirect circular reference: {' -> '.join(cycle_path)}")
    
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        self.load_file()
        self.extract_assignments()
        self.build_dependency_graph()
        self.check_direct_circular_references()
        self.check_indirect_circular_references()
    
    def export_results(self, output_file: str) -> None:
        """Export analysis results to a JSON file."""
        results = {
            'file': self.file_path,
            'variables': len(self.variables),
            'issues': self.issues
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results written to: {output_file}")
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        print("\n=== ELFIN Circular Reference Analysis Summary ===")
        print(f"File: {self.file_path}")
        print(f"Variables analyzed: {len(self.variables)}")
        
        # Count issues by severity
        error_count = sum(1 for issue in self.issues if issue['severity'] == 'error')
        warning_count = sum(1 for issue in self.issues if issue['severity'] == 'warning')
        
        print(f"Issues: {len(self.issues)} ({error_count} errors, {warning_count} warnings)")
        
        if self.issues:
            print("\n=== Issues ===")
            for i, issue in enumerate(self.issues):
                severity_marker = "ERROR" if issue['severity'] == 'error' else "WARNING"
                print(f"[{severity_marker}] {issue['message']}")
    
    def has_circular_references(self) -> bool:
        """Return True if any circular references were detected."""
        return len(self.issues) > 0


def analyze_file(file_path: str, output_file: str = None) -> bool:
    """
    Analyze a single ELFIN file for circular references.
    
    Args:
        file_path: Path to the ELFIN file to analyze
        output_file: Optional path to write JSON results
        
    Returns:
        True if circular references were found, False otherwise
    """
    analyzer = CircularReferenceAnalyzer(file_path)
    analyzer.run_analysis()
    analyzer.print_summary()
    
    if output_file:
        analyzer.export_results(output_file)
    
    return analyzer.has_circular_references()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python circular_analyzer.py <file_path> [output_file]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    has_issues = analyze_file(file_path, output_file)
    sys.exit(1 if has_issues else 0)
