#!/usr/bin/env python3
"""
Test for error documentation completeness.

This test verifies that documentation exists for all error codes
used in the ELFIN framework.
"""

import pathlib
import unittest
from typing import List

# Import error handler
from alan_backend.elfin.errors import ErrorHandler
from alan_backend.elfin.errors.error_handler import VerificationError


class TestErrorDocs(unittest.TestCase):
    """Test case for error documentation."""
    
    def setUp(self):
        """Set up test case."""
        # Get the path to docs/errors
        self.error_handler = ErrorHandler()
        
        # Known error codes (these should all have documentation)
        self.known_error_codes = list(VerificationError.ERROR_CODES.keys())
    
    def test_error_doc_exists(self):
        """Test that documentation exists for all known error codes."""
        missing_docs = []
        
        for code in self.known_error_codes:
            # Check if documentation exists
            if not self.error_handler.error_doc_exists(code):
                missing_docs.append(code)
        
        # Assert that no docs are missing
        self.assertEqual(
            len(missing_docs), 0,
            f"Missing documentation for error codes: {', '.join(missing_docs)}"
        )
    
    def test_lyap_001_doc_exists(self):
        """Test that documentation exists for E-LYAP-001."""
        self.assertTrue(
            self.error_handler.error_doc_exists("LYAP_001"),
            "Missing documentation for E-LYAP-001"
        )
    
    def test_lyap_002_doc_exists(self):
        """Test that documentation exists for E-LYAP-002."""
        self.assertTrue(
            self.error_handler.error_doc_exists("LYAP_002"),
            "Missing documentation for E-LYAP-002"
        )


if __name__ == "__main__":
    unittest.main()
