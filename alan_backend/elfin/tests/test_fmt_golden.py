"""
Golden file tests for the ELFIN formatter.

This module contains tests that verify the formatter preserves the formatting
of carefully curated golden files. These tests ensure that changes to the
formatter don't inadvertently change the expected behavior.
"""

import unittest
from pathlib import Path

# Import formatter
from alan_backend.elfin.formatting.elffmt import ELFINFormatter


class TestFormatterGolden(unittest.TestCase):
    """Test the formatter against golden files."""
    
    def setUp(self):
        """Set up the test case."""
        self.formatter = ELFINFormatter()
        
        # Path to golden files
        self.golden_dir = Path(__file__).parent / "golden"
        
    def test_golden_files(self):
        """Test that formatting golden files doesn't change them."""
        # Find all .elfin files in the golden directory
        golden_files = list(self.golden_dir.glob("*.elfin"))
        
        # Make sure we found some golden files
        self.assertGreater(len(golden_files), 0, 
                          "No golden files found for testing")
        
        # For each golden file, verify that formatting doesn't change it
        for file_path in golden_files:
            with self.subTest(file=file_path.name):
                # Read the original content
                original = file_path.read_text()
                
                # Format the content
                formatted = self.formatter.format_string(original)
                
                # Verify the content didn't change
                self.assertEqual(
                    original, 
                    formatted,
                    f"Formatting changed the content of '{file_path.name}'"
                )
                
                # Print success message
                print(f"✓ {file_path.name} preserved after formatting")


if __name__ == "__main__":
    unittest.main()
