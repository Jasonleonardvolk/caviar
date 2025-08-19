#!/usr/bin/env python3
"""
Super simple debug script for ELFIN parser with minimal regex.
"""

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

def simple_debug():
    """Manually parse the file without complex regex."""
    script_dir = Path(__file__).parent
    test_file = script_dir / "test_files" / "simple_debug.elfin"
    os.makedirs(test_file.parent, exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(TEST_CONTENT)
    
    print(f"Created test file: {test_file}")
    
    # Read the file
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Manual parsing
    print("\n==== Step 1: Manual Parsing ====")
    
    # Find system block
    system_start = content.find("system")
    if system_start >= 0:
        print(f"Found system at position {system_start}")
        
        # Find opening brace
        open_brace = content.find("{", system_start)
        if open_brace >= 0:
            print(f"Found opening brace at position {open_brace}")
            
            # Find closing brace (assuming no nested braces of same level)
            brace_count = 1
            pos = open_brace + 1
            while brace_count > 0 and pos < len(content):
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                system_content = content[open_brace+1:pos-1]
                print(f"System content length: {len(system_content)} chars")
                
                # Find continuous_state
                cs_start = system_content.find("continuous_state")
                if cs_start >= 0:
                    print(f"Found continuous_state at position {cs_start}")
                    
                    # Find opening brace
                    cs_open = system_content.find("{", cs_start)
                    if cs_open >= 0:
                        # Find closing brace
                        cs_close = system_content.find("}", cs_open)
                        if cs_close >= 0:
                            cs_content = system_content[cs_open+1:cs_close]
                            print(f"Continuous state content: '{cs_content}'")
                            
                            # Split by semicolons
                            state_vars = []
                            for var_entry in cs_content.split(';'):
                                # Remove comments
                                if '//' in var_entry:
                                    var_entry = var_entry[:var_entry.find('//')]
                                
                                var_name = var_entry.strip()
                                if var_name:
                                    state_vars.append(var_name)
                            
                            print(f"State variables: {state_vars}")
                        else:
                            print("Could not find closing brace for continuous_state")
                    else:
                        print("Could not find opening brace for continuous_state")
                else:
                    print("Could not find continuous_state block")
                
                # Find flow_dynamics
                fd_start = system_content.find("flow_dynamics")
                if fd_start >= 0:
                    print(f"Found flow_dynamics at position {fd_start}")
                    
                    # Find opening brace
                    fd_open = system_content.find("{", fd_start)
                    if fd_open >= 0:
                        # Find closing brace
                        fd_close = system_content.find("}", fd_open)
                        if fd_close >= 0:
                            fd_content = system_content[fd_open+1:fd_close]
                            print(f"Flow dynamics content: '{fd_content}'")
                            
                            # Process each line
                            for line in fd_content.strip().split('\n'):
                                # Remove comments
                                if '//' in line:
                                    line = line[:line.find('//')]
                                
                                line = line.strip()
                                if not line:
                                    continue
                                
                                # Check for assignment
                                if '=' in line:
                                    parts = line.split('=', 1)
                                    var_name = parts[0].strip()
                                    expr = parts[1].strip()
                                    if expr.endswith(';'):
                                        expr = expr[:-1].strip()
                                    
                                    print(f"Found dynamic variable: {var_name} = {expr}")
                                    
                                    # Check for self-reference
                                    if var_name in expr:
                                        print(f"CIRCULAR REFERENCE DETECTED: {var_name} in {expr}")
                        else:
                            print("Could not find closing brace for flow_dynamics")
                    else:
                        print("Could not find opening brace for flow_dynamics")
                else:
                    print("Could not find flow_dynamics block")
            else:
                print("Unbalanced braces in system block")
        else:
            print("Could not find opening brace for system")
    else:
        print("Could not find system block")

if __name__ == "__main__":
    simple_debug()
