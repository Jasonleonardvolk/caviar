#!/usr/bin/env python3
"""
Test script for the focused Circular Reference Analyzer.
"""

import os
import sys
from pathlib import Path

# Ensure the analyzer module is in the path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import the analyzer
from circular_analyzer import CircularReferenceAnalyzer

# Test content with direct circular reference
DIRECT_CIRCULAR = """
// Test file with direct circular reference
system Robot {
    flow_dynamics {
        x_dot = v * cos(theta);
        v_dot = v_dot + u / m;  // DIRECT circular reference
    }
}
"""

# Test content with indirect circular reference
INDIRECT_CIRCULAR = """
// Test file with indirect circular reference
system Robot {
    flow_dynamics {
        x_dot = v * cos(theta);
        v_dot = a;
    }
}

mode Control {
    controller {
        a = b;
        b = c;
        c = a;  // INDIRECT circular reference: a -> b -> c -> a
    }
}
"""

# Test content without circular references
NO_CIRCULAR = """
// Test file with no circular references
system Robot {
    flow_dynamics {
        x_dot = v * cos(theta);
        v_dot = a;
    }
}

mode Control {
    controller {
        a = v * cos(theta);
        b = a + sin(theta);
        c = b * 2.0;
    }
}
"""

def create_test_file(content, name):
    """Create a test file with the specified content."""
    test_dir = script_dir / "test_files"
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / f"{name}.elfin"
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return test_file

def run_test(content, name, expect_circular=True):
    """Run a test and return True if the test passed."""
    print(f"\n=== Running Test: {name} ===")
    test_file = create_test_file(content, name)
    
    analyzer = CircularReferenceAnalyzer(str(test_file))
    analyzer.run_analysis()
    analyzer.print_summary()
    
    has_circular = analyzer.has_circular_references()
    if has_circular == expect_circular:
        print(f"✅ PASS: {'Found' if has_circular else 'Did not find'} circular references as expected.")
        return True
    else:
        print(f"❌ FAIL: {'Found' if has_circular else 'Did not find'} circular references, but expected {'to find' if expect_circular else 'not to find'} them.")
        return False

def main():
    """Run all the tests."""
    print("=== ELFIN Circular Reference Analyzer Tests ===")
    test_results = []
    
    # Test direct circular reference - should detect
    result1 = run_test(DIRECT_CIRCULAR, "direct_circular", expect_circular=True)
    test_results.append(("Direct circular reference", result1))
    
    # Test indirect circular reference - should detect
    result2 = run_test(INDIRECT_CIRCULAR, "indirect_circular", expect_circular=True)
    test_results.append(("Indirect circular reference", result2))
    
    # Test no circular reference - should not detect
    result3 = run_test(NO_CIRCULAR, "no_circular", expect_circular=False)
    test_results.append(("No circular reference", result3))
    
    # Print summary
    print("\n=== Test Summary ===")
    for name, result in test_results:
        print(f"{name}: {'PASS' if result else 'FAIL'}")
    
    all_pass = all(result for _, result in test_results)
    print(f"\nOverall: {'SUCCESS' if all_pass else 'FAILURE'}")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
