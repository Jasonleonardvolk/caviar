#!/usr/bin/env python3
"""
Direct debug script for ELFIN analyzer.
This bypasses most of our abstraction to directly examine the parsing issues.
"""

import re
import os
from pathlib import Path

# Create a simple test file
TEST_CONTENT = """
// Simple test file with circular reference for debugging
system TestSystem {
    continuous_state {
        x; y; // Two state variables
    }
    
    input {
        u; // One input variable
    }
    
    params {
        m: 1.0; // Mass parameter
    }
    
    flow_dynamics {
        // Simple dynamics with circular reference
        x_dot = y;
        y_dot = y_dot + u / m; // Circular reference here
    }
}
"""

def direct_debug():
    """Direct debugging of the parsing functionality."""
    script_dir = Path(__file__).parent
    test_file = script_dir / "test_files" / "direct_debug.elfin"
    os.makedirs(test_file.parent, exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(TEST_CONTENT)
    
    print(f"Created test file: {test_file}")
    
    # Read the file
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n==== Step 1: Find Sections ====")
    # Find top-level sections
    section_pattern = r'(?:helpers|system|lyapunov|barrier|mode)\s*:?\s*([A-Za-z0-9_]+)\s*{([^}]*)}'
    for match in re.finditer(section_pattern, content, re.DOTALL):
        section_type = match.group(0).split()[0].rstrip(':')
        section_name = match.group(1)
        section_content = match.group(2)
        
        print(f"Found section: {section_type}:{section_name}")
        
        # Now directly try to extract the parts of each section
        print("\n==== Step 2: Parse Section Content ====")
        
        # Try to find the continuous_state block
        # Print the section content for debugging
        print(f"\nSection content (first 100 chars): '{section_content[:100]}...'")
        
        print("\nLooking for continuous_state block...")
        # More flexible pattern that doesn't rely on balanced braces
        state_pattern = r'continuous_state\s*\{(.*?)\}'
        state_match = re.search(state_pattern, section_content, re.DOTALL)
        if state_match:
            state_block = state_match.group(1)
            print(f"Found continuous_state block: '{state_block}'")
            
            # Try to extract variables from the state block
            # First remove comments
            state_block = re.sub(r'//.*?$', '', state_block, flags=re.MULTILINE)
            print(f"After comment removal: '{state_block}'")
            
            # Split by semicolons
            state_vars = [var.strip() for var in state_block.split(';') if var.strip()]
            print(f"State variables: {state_vars}")
        else:
            print("No continuous_state block found! Trying alternate pattern...")
            # Try a simpler pattern
            state_pattern2 = r'continuous_state\s*\{([\s\S]*?)\}'
            state_match = re.search(state_pattern2, section_content)
            if state_match:
                print(f"Found with alternate pattern: '{state_match.group(1)}'")
            else:
                print("Still not found!")
        
        # Try to find the flow_dynamics block
        print("\nLooking for flow_dynamics block...")
        dynamics_pattern = r'flow_dynamics\s*\{(.*?)\}'
        dynamics_match = re.search(dynamics_pattern, section_content, re.DOTALL)
        if dynamics_match:
            dynamics_block = dynamics_match.group(1)
            print(f"Found flow_dynamics block: '{dynamics_block}'")
            
            # Remove comments and split into lines
            dynamics_block = re.sub(r'//.*?$', '', dynamics_block, flags=re.MULTILINE)
            print(f"After comment removal: '{dynamics_block}'")
            
            # Process each line
            for line in dynamics_block.strip().split('\n'):
                print(f"Processing line: '{line}'")
                line = line.strip()
                if not line:
                    print("  Empty line, skipping")
                    continue
                
                # Try to extract variable and expression
                if '=' in line:
                    parts = line.split('=', 1)
                    var_name = parts[0].strip()
                    expr = parts[1].strip()
                    if expr.endswith(';'):
                        expr = expr[:-1].strip()
                    
                    print(f"  Found dynamic variable: {var_name} = {expr}")
                    
                    # Check for self-reference
                    if var_name in expr:
                        print(f"  CIRCULAR REFERENCE DETECTED: {var_name} in {expr}")
                else:
                    print("  No assignment found in line")

if __name__ == "__main__":
    direct_debug()
