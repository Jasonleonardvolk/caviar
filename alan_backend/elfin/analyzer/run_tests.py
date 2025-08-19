#!/usr/bin/env python3
"""
ELFIN Analyzer Test Runner

Tests the circular reference analyzer against sample ELFIN files and prints results.
"""

import os
import sys
from pathlib import Path

# Ensure the analyzer module is in the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import the analyzer
from reference_analyzer import ELFINAnalyzer

# Test content with circular references
TEST_WITH_ERRORS = """
// Test file with circular references
system RobotSystem {
    continuous_state {
        x; y; theta;
        v; omega;
    }
    
    input {
        a; alpha;
    }
    
    params {
        m: 1.0;
        J: 0.1;
        L: 0.5;
    }
    
    flow_dynamics {
        // State evolution
        x_dot = v * cos(theta);
        y_dot = v * sin(theta);
        theta_dot = omega;
        
        // Circular reference in velocity
        v_dot = v_dot + a / m;  // ERROR: Circular reference
        
        // Multiple derivatives for the same base variable
        omega_dot = alpha / J;
        dtheta = 2.0 * omega;   // WARNING: Another derivative for theta
    }
}

mode PD_Control {
    params {
        kp: 5.0;
        kd: 1.0;
    }
    
    controller {
        // Alias example - these have identical expressions
        control1 = kp * (0 - theta) + kd * (0 - omega);
        control2 = kp * (0 - theta) + kd * (0 - omega);
        
        // Indirect circular reference
        temp1 = temp2 + 1.0;  // ERROR: Indirect circular reference
        temp2 = temp3 * 2.0;
        temp3 = temp1 / 2.0;
    }
}
"""

# Test content without errors
TEST_WITHOUT_ERRORS = """
// Test file without circular references
system RobotSystem {
    continuous_state {
        x; y; theta;
        v; omega;
    }
    
    input {
        a; alpha;
    }
    
    params {
        m: 1.0;
        J: 0.1;
        L: 0.5;
    }
    
    flow_dynamics {
        // Good dynamics equations
        x_dot = v * cos(theta);
        y_dot = v * sin(theta);
        theta_dot = omega;
        v_dot = a / m;
        omega_dot = alpha / J;
    }
}

mode PD_Control {
    params {
        kp: 5.0;
        kd: 1.0;
    }
    
    controller {
        // Good controller
        control1 = kp * (0 - theta) + kd * (0 - omega);
        control2 = kp * (0 - theta) + kd * (0 - omega);
    }
}
"""

def run_test_case(test_content, name):
    """Run analyzer on test content and print results."""
    print(f"\n=== Running Test: {name} ===")
    
    # Create temporary test file
    test_dir = script_dir / "test_files"
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / f"{name}.elfin"
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"Created test file: {test_file}")
    
    # Create and run analyzer
    analyzer = ELFINAnalyzer(str(test_file))
    analyzer.run_analysis()
    
    # Print results
    print(f"\nSymbols found: {len(analyzer.symbols)}")
    for name, info in analyzer.symbols.items():
        sym_type = info.get('type', 'unknown')
        print(f"  {name}: {sym_type}")
        if 'expression' in info:
            print(f"    expression: {info['expression']}")
    
    print(f"\nReferences:")
    for var, refs in analyzer.references.items():
        if refs:
            print(f"  {var} -> {', '.join(refs)}")
    
    print(f"\nIssues found: {len(analyzer.issues)}")
    for i, issue in enumerate(analyzer.issues):
        severity = issue.get('severity', 'unknown')
        issue_type = issue.get('type', 'unknown')
        message = issue.get('message', 'no message')
        print(f"  {i+1}. [{severity.upper()}] {issue_type}: {message}")
    
    # Highlight circular references specifically
    circular_refs = [issue for issue in analyzer.issues if issue['type'] == 'circular_reference']
    if circular_refs:
        print(f"\nFound {len(circular_refs)} circular references:")
        for issue in circular_refs:
            print(f"  - {issue['message']}")
    else:
        print("\nNo circular references detected.")
    
    # Return success/failure
    return len(circular_refs) > 0

def main():
    """Run all tests."""
    print("=== ELFIN Analyzer Test Runner ===")
    
    # Run tests
    with_errors = run_test_case(TEST_WITH_ERRORS, "test_with_errors")
    without_errors = run_test_case(TEST_WITHOUT_ERRORS, "test_without_errors")
    
    # Report results
    print("\n=== Test Summary ===")
    print(f"Test with errors: {'PASS' if with_errors else 'FAIL'} (expected to find circular references)")
    print(f"Test without errors: {'PASS' if not without_errors else 'FAIL'} (expected no circular references)")
    
    if with_errors and not without_errors:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
