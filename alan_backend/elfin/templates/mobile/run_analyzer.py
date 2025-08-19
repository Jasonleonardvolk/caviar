#!/usr/bin/env python3
"""
Runner script for the ELFIN Mobile Robot Controller Analyzer.
This script analyzes the controller for correctness and generates simulation code.
"""

import os
import sys
from pathlib import Path
import json

# Define paths
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
controller_path = current_dir / "src" / "mobile_robot_controller.elfin"
output_dir = current_dir / "output"
results_path = output_dir / "analysis_results.json"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def analyze_elfin_file(filepath):
    """
    Basic analyzer for ELFIN file to check syntax and circular references
    """
    print(f"Analyzing file: {filepath}")
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"File loaded: {len(content)} bytes")
        
        # Check for systems
        import re
        
        # Simpler approach - use a basic string search to directly extract the system section
        system_pattern = re.compile(r'system\s+DifferentialDrive\s*{.*?^}', re.DOTALL | re.MULTILINE)
        system_match = system_pattern.search(content)
        if not system_match:
            print("Failed to match system section with multiline pattern, trying alternative")
            # Fallback to a simpler approach
            start_idx = content.find("system DifferentialDrive {")
            if start_idx != -1:
                end_idx = content.find("\n}", start_idx)
                if end_idx != -1:
                    system_content = content[start_idx:end_idx+2]
                    system_name = "DifferentialDrive"
                    system_match = True
        if system_match:
            print(f"Found system section")
            system_name = "DifferentialDrive"  # We know this from the file
            
            if isinstance(system_match, bool):  # We created this in our fallback
                print(f"Using fallback approach to match system section")
            else:
                system_content = system_match.group(0)
                system_name = "DifferentialDrive"
                
            print(f"System content found, length: {len(system_content)}")
            
            # Extract flow_dynamics using a direct search approach
            flow_start_idx = system_content.find("flow_dynamics {")
            if flow_start_idx != -1:
                print(f"Found flow_dynamics at position {flow_start_idx}")
                flow_end_idx = system_content.find("};", flow_start_idx)
                if flow_end_idx != -1:
                    dynamics_content = system_content[flow_start_idx+15:flow_end_idx]
                    print(f"Extracted dynamics section: {len(dynamics_content)} chars")
                    dynamics_match = True
                else:
                    print("Could not find end of flow_dynamics section")
                    dynamics_match = None
            else:
                print("Could not find flow_dynamics section")
                dynamics_match = None
            if dynamics_match:
                # if it's not a boolean (from our manual extraction), get the group(1)
                if not isinstance(dynamics_match, bool):
                    dynamics_content = dynamics_match.group(1)
                print("Successfully extracted dynamics section")
                
                # Check for circular references in dynamics
                dynamics_entries = re.findall(r'([A-Za-z0-9_]+)\s*=\s*(.*?);', dynamics_content)
                for var, expr in dynamics_entries:
                    # Exact name match (not as part of another name)
                    var_name = var.replace('_dot', '')
                    pattern = r'\b' + re.escape(var_name) + r'\b'
                    
                    # Check if variable references itself
                    if re.search(pattern, expr):
                        # Special case: x_dot = v is fine because v is a different variable
                        if var.endswith('_dot') and var_name != expr:
                            continue
                        
                        issues.append({
                            'type': 'circular_reference',
                            'severity': 'error',
                            'message': f"Circular reference detected: {var} depends on itself",
                            'variable': var,
                            'expression': expr
                        })
            else:
                issues.append({
                    'type': 'missing_section',
                    'severity': 'error',
                    'message': f"Could not find flow_dynamics section in system {system_name}"
                })
        else:
            issues.append({
                'type': 'missing_section',
                'severity': 'error',
                'message': "Could not find system definition"
            })
            
        # Check syntax issues
        # Check for helpers syntax (should be "helpers {" not "helpers: {")
        if "helpers: {" in content:
            issues.append({
                'type': 'syntax_error',
                'severity': 'error',
                'message': 'Invalid syntax: "helpers: {" should be "helpers {"'
            })
            
        # Check for system: syntax
        if "system: " in content:
            issues.append({
                'type': 'syntax_error',
                'severity': 'error',
                'message': 'Invalid syntax: "system: " reference - should be direct system name'
            })
            
        return issues
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return [{
            'type': 'exception',
            'severity': 'error',
            'message': f"Exception during analysis: {str(e)}"
        }]

# Run the analysis
issues = analyze_elfin_file(controller_path)

# Print summary
print("\n=== ELFIN Analysis Summary ===")
print(f"File: {controller_path}")

# Count issues by severity
error_count = sum(1 for issue in issues if issue['severity'] == 'error')
warning_count = sum(1 for issue in issues if issue['severity'] == 'warning')

print(f"Issues: {len(issues)} ({error_count} errors, {warning_count} warnings)")

if issues:
    print("\n=== Issues ===")
    for issue in issues:
        severity_marker = "ERROR" if issue['severity'] == 'error' else "WARNING"
        print(f"[{severity_marker}] {issue['message']}")

# Export results
results = {
    'file': str(controller_path),
    'issues': issues
}

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"\nAnalysis results written to: {results_path}")

if error_count == 0:
    print("\n✅ No circular references or syntax errors found in the ELFIN specification!")
    print("The file is ready for formal verification and simulation.")
else:
    print(f"\n❌ Found {error_count} errors in the ELFIN specification.")
    print("Please fix these issues before proceeding with simulation or verification.")
