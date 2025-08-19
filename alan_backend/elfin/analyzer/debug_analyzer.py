#!/usr/bin/env python3
"""
Debug script for the ELFIN analyzer.
Creates a simple test file and runs detailed analysis.
"""

import sys
import os
from pathlib import Path
import re

# Ensure the analyzer module is in the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import the analyzer
from reference_analyzer import ELFINAnalyzer

# Create a very simple test file with a direct circular reference
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

def debug_analyzer():
    """Run a detailed debug session on the analyzer."""
    # Create a test file
    test_file = script_dir / "test_files" / "debug_test.elfin"
    os.makedirs(test_file.parent, exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(TEST_CONTENT)
    
    print(f"Created test file: {test_file}")
    
    # Create analyzer
    analyzer = ELFINAnalyzer(str(test_file))
    
    # Debug step 1: Load and parse file
    analyzer.load_file()
    analyzer.parse_sections()
    
    print("\nSections found:", len(analyzer.sections))
    for key, content in analyzer.sections.items():
        print(f"  {key}: {len(content)} chars")
    
    # Debug step 2: Extract symbols
    analyzer.extract_symbols()
    
    print("\nSymbols found:", len(analyzer.symbols))
    for name, info in analyzer.symbols.items():
        print(f"  {name}: {info['type']}")
        for k, v in info.items():
            if k != 'type':
                print(f"    {k}: {v}")
    
    # Debug step 3: Analyze references
    analyzer.analyze_references()
    
    print("\nReferences found:")
    for symbol, refs in analyzer.references.items():
        print(f"  {symbol} -> {refs}")
    
    # Debug step 4: Check for circular references
    analyzer.check_for_circular_references()
    
    # Debug step 5: Check for derivative consistency
    analyzer.check_for_derivative_consistency()
    
    # Debug step 6: Detect potential aliases
    analyzer.detect_potential_aliases()
    
    # Debug step 7: Validate references
    analyzer.validate_references()
    
    # Debug step 8: Check dynamics completeness
    analyzer.check_dynamics_completeness()
    
    # Print results
    print("\nIssues found:", len(analyzer.issues))
    for i, issue in enumerate(analyzer.issues):
        print(f"  Issue {i+1}:")
        for k, v in issue.items():
            print(f"    {k}: {v}")
    
    # Check for specific problems
    print("\nSpecific problem check:")
    if "y_dot" in analyzer.symbols:
        print("  y_dot symbol found!")
        if 'expression' in analyzer.symbols["y_dot"]:
            print(f"  Expression: {analyzer.symbols['y_dot']['expression']}")
            
            # Check if this is a direct circular reference
            if "y_dot" in analyzer.references.get("y_dot", set()):
                print("  -> FOUND CIRCULAR REFERENCE in references dictionary!")
            
            # Check if regex pattern matches
            pattern = r'\by_dot\b'
            expr = analyzer.symbols["y_dot"]['expression']
            if re.search(pattern, expr):
                print("  -> FOUND CIRCULAR REFERENCE with regex match!")
    else:
        print("  y_dot symbol NOT found!")

if __name__ == "__main__":
    debug_analyzer()
