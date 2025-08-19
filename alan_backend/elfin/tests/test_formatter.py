"""
Tests for the ELFIN code formatter.

This module tests the formatter to ensure it correctly formats ELFIN code
according to the style guidelines.
"""

import unittest
import sys
import os
import tempfile
import filecmp
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from alan_backend.elfin.formatting.elffmt import ELFINFormatter


class TestELFINFormatter(unittest.TestCase):
    """Test cases for the ELFIN formatter."""
    
    def setUp(self):
        """Set up for tests."""
        self.formatter = ELFINFormatter()
        self.test_cases_dir = Path(__file__).parent / "formatter_test_cases"
        os.makedirs(self.test_cases_dir, exist_ok=True)
    
    def test_basic_formatting(self):
        """Test basic formatting cases."""
        input_code = """
# Unformatted code
system  TestSystem {
 continuous_state : [x, y, z];
    input: [u];
      
  params{
 a:1;b: 2;
  c : 3;
}

flow_dynamics{
x_dot =v;
    y_dot= v;
  z_dot  =  v;
}
}
"""

        expected_output = """
# Unformatted code
system TestSystem {
  continuous_state: [x, y, z];
  input: [u];
  
  params {
    a: 1;
    b: 2;
    c: 3;
  }
  
  flow_dynamics {
    x_dot = v;
    y_dot = v;
    z_dot = v;
  }
}
"""
        
        formatted = self.formatter.format_string(input_code)
        self.assertEqual(formatted.strip(), expected_output.strip())
    
    def test_parameter_alignment(self):
        """Test alignment of parameters in param blocks."""
        input_code = """
system AlignmentTest {
  params {
    short_name: 1;
    very_long_parameter_name: 2;
    medium_length: 3;
    x: 4;
  }
}
"""
        
        formatted = self.formatter.format_string(input_code)
        
        # Check that the equals signs align in the formatted output
        lines = formatted.strip().split('\n')
        param_lines = [line for line in lines if ":" in line and "=" not in line]
        
        # Extract the positions of the colons
        colon_positions = [line.find(":") for line in param_lines]
        
        # All colons should be at the same position
        if colon_positions:
            self.assertEqual(len(set(colon_positions)), 1, 
                             "Colons are not aligned in param block")
    
    def test_unit_formatting(self):
        """Test compact formatting of unit annotations."""
        input_code = """
system UnitTest {
  params {
    velocity   :  [m/s]  = 10.0;
    acceleration: [m/s^2] = 9.81;
    mass: [kg] = 1.0;
  }
}
"""
        
        expected_output = """
system UnitTest {
  params {
    velocity[m/s]    = 10.0;
    acceleration[m/s^2] = 9.81;
    mass[kg]         = 1.0;
  }
}
"""
        
        formatted = self.formatter.format_string(input_code)
        self.assertEqual(formatted.strip(), expected_output.strip())
    
    def test_line_length_limit(self):
        """Test that lines are limited to 80 columns."""
        # Create a very long line
        very_long_param = "very_" * 20 + "long"
        input_code = f"""
system LineLengthTest {{
  params {{
    {very_long_param}: 1;
  }}
}}
"""
        
        formatted = self.formatter.format_string(input_code)
        
        # Check that no line exceeds 80 characters
        lines = formatted.strip().split('\n')
        for line in lines:
            self.assertLessEqual(len(line), 80, 
                                f"Line exceeds 80 characters: {line}")
    
    def test_import_statements(self):
        """Test formatting of import statements."""
        input_code = """
import   Helpers   from    "std/helpers.elfin"  ;

system ImportTest {
  params {
    a: 1;
  }
}
"""
        
        expected_output = """
import Helpers from "std/helpers.elfin";

system ImportTest {
  params {
    a: 1;
  }
}
"""
        
        formatted = self.formatter.format_string(input_code)
        self.assertEqual(formatted.strip(), expected_output.strip())
    
    def test_golden_files(self):
        """Test formatting against golden files."""
        # Get paths to golden files
        golden_dir = Path(__file__).parent / "golden" / "formatter"
        input_file = golden_dir / "unformatted_input.elfin"
        expected_file = golden_dir / "formatted_output.elfin"
        
        # Ensure golden files exist
        self.assertTrue(input_file.exists(), f"Golden input file not found: {input_file}")
        self.assertTrue(expected_file.exists(), f"Golden output file not found: {expected_file}")
        
        # Read input file
        with open(input_file, 'r') as f:
            input_code = f.read()
        
        # Format the input
        formatted = self.formatter.format_string(input_code)
        
        # Write to a temporary file for comparison
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.elfin') as tmp:
            tmp_path = tmp.name
            tmp.write(formatted)
        
        try:
            # Compare with expected output
            expected_content = expected_file.read_text()
            self.assertEqual(formatted.strip(), expected_content.strip(), 
                            "Formatted output doesn't match golden file")
            
            # Also test using filecmp for binary comparison
            self.assertTrue(filecmp.cmp(tmp_path, expected_file, shallow=False),
                          "Formatted file doesn't match golden file (binary comparison)")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
