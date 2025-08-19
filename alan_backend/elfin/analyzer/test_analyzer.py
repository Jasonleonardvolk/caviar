#!/usr/bin/env python3
"""
ELFIN Analyzer Test Script

Tests the circular reference analyzer against sample ELFIN files.
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure the analyzer module is in the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent.parent))

# Use direct import to avoid depending on alan_backend.elfin package
try:
    from reference_analyzer import ELFINAnalyzer
except ImportError:
    # Try different import paths
    try:
        from alan_backend.elfin.analyzer.reference_analyzer import ELFINAnalyzer
    except ImportError:
        sys.path.append(str(script_dir))
        from reference_analyzer import ELFINAnalyzer


def create_test_file(test_file_path, with_errors=True):
    """
    Create a sample ELFIN file for testing with or without circular references.
    
    Args:
        test_file_path: Path to create the test file
        with_errors: Whether to include circular references
    """
    # Sample ELFIN content with circular reference
    if with_errors:
        content = """
// Test file with circular references
helpers Circle {
    // Helper function
    squared(x) = x * x;
    doubled(x) = x + x;
}

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

lyapunov Energy {
    params {
        k1: 1.0;
        k2: 2.0;
    }
    
    // Energy function
    V = 0.5 * squared(v) + 0.5 * squared(omega);
}

barrier SafetyBounds {
    params {
        x_max: 10.0;
        y_max: 10.0;
    }
    
    // Barrier function
    B = (x_max - x) * (x_max + x) * (y_max - y) * (y_max + y);
    
    // Alpha function
    alphaFun = 0.5 * B;
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
        
        a = 0.0;
        alpha = control1;  // Using one of the aliases
        
        // Indirect circular reference
        temp1 = temp2 + 1.0;  // ERROR: Indirect circular reference
        temp2 = temp3 * 2.0;
        temp3 = temp1 / 2.0;
    }
}
"""
    else:
        # Sample without circular references
        content = """
// Test file with no circular references
helpers Circle {
    // Helper function
    squared(x) = x * x;
    doubled(x) = x + x;
}

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
        
        // Good dynamics equations
        v_dot = a / m;
        omega_dot = alpha / J;
    }
}

lyapunov Energy {
    params {
        k1: 1.0;
        k2: 2.0;
    }
    
    // Energy function
    V = 0.5 * squared(v) + 0.5 * squared(omega);
}

barrier SafetyBounds {
    params {
        x_max: 10.0;
        y_max: 10.0;
    }
    
    // Barrier function
    B = (x_max - x) * (x_max + x) * (y_max - y) * (y_max + y);
    
    // Alpha function
    alphaFun = 0.5 * B;
}

mode PD_Control {
    params {
        kp: 5.0;
        kd: 1.0;
    }
    
    controller {
        // Good controller
        control1 = kp * (0 - theta) + kd * (0 - omega);
        
        a = 0.0;
        alpha = control1;
    }
}
"""
    
    # Create the file
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created test file: {test_file_path}")


def run_tests():
    """Run analyzer tests on sample files."""
    test_dir = script_dir / "test_files"
    test_dir.mkdir(exist_ok=True)
    
    # Test files
    test_with_errors = test_dir / "test_with_errors.elfin"
    test_without_errors = test_dir / "test_without_errors.elfin"
    
    # Create test files
    create_test_file(test_with_errors, with_errors=True)
    create_test_file(test_without_errors, with_errors=False)
    
    # Test file with errors - should find issues
    print("\n=== Testing file with errors ===")
    analyzer1 = ELFINAnalyzer(str(test_with_errors))
    analyzer1.run_analysis()
    analyzer1.print_summary()
    
    # Check if errors were found
    error_count1 = sum(1 for issue in analyzer1.issues if issue['severity'] == 'error')
    if error_count1 >= 2:
        print("\n✅ PASS: Successfully detected circular references")
    else:
        print("\n❌ FAIL: Failed to detect all expected circular references")
    
    # Check if alias was detected
    alias_count = sum(1 for issue in analyzer1.issues if issue['type'] == 'potential_alias')
    if alias_count >= 1:
        print("✅ PASS: Successfully detected potential aliases")
    else:
        print("❌ FAIL: Failed to detect aliases")
    
    # Test file without errors - should be clean
    print("\n=== Testing file without errors ===")
    analyzer2 = ELFINAnalyzer(str(test_without_errors))
    analyzer2.run_analysis()
    analyzer2.print_summary()
    
    # Check if no errors were found
    error_count2 = sum(1 for issue in analyzer2.issues if issue['severity'] == 'error')
    if error_count2 == 0:
        print("\n✅ PASS: No errors detected in clean file")
    else:
        print(f"\n❌ FAIL: Found {error_count2} errors in clean file")
    
    # Return success if tests passed
    return error_count1 >= 2 and alias_count >= 1 and error_count2 == 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the ELFIN analyzer")
    parser.add_argument('--run', action='store_true', help='Run the tests')
    args = parser.parse_args()
    
    if args.run:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
