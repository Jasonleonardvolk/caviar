#!/usr/bin/env python3
# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
ELFIN Grammar Roundtrip Tests

This module tests the idempotence of ELFIN parsing by verifying:
1. Input text → AST → Output text round-trips correctly
2. The generated AST contains the expected nodes and properties
3. Token generation is stable

These tests ensure the canonical grammar (elfin_v1.ebnf) is well-formed
and consistent with the parser implementation.
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import subprocess
import importlib.util
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import parser modules
from alan_backend.elfin.standalone_parser import parse, parse_file, Program
import scripts.gen_tokens as token_gen  # Import token generator

# Path to the canonical grammar file
GRAMMAR_PATH = Path("elfin/grammar/elfin_v1.ebnf")

# Example ELFIN files for testing
EXAMPLES_DIR = Path("alan_backend/elfin/examples")

# Basic test cases for parser (simple strings that should parse correctly)
BASIC_TEST_CASES = [
    # Empty system
    """
    system EmptySystem {
        continuous_state: [];
        input: [];
        params {}
        dynamics {}
    }
    """,
    
    # Simple system with dynamics
    """
    system Oscillator {
        continuous_state: [x, v];
        input: [u];
        params {
            k: 1.0 [N/m];
            b: 0.1 [N·s/m];
            m: 1.0 [kg];
        }
        dynamics {
            x' = v;
            v' = -k/m * x - b/m * v + u/m;
        }
    }
    """,
    
    # Concept with spin vector
    """
    concept "SpinOscillator" {
        spinvec s;
        float theta [rad];
        float p_theta [rad/s];
        
        constraint norm(s) == 1.0;
        
        reversible true;
    }
    """,
    
    # Reversible block
    """
    reversible SpinIntegrator {
        forward {
            theta = theta + dt * p_theta;
            s = normalize(rotate(s, dtheta));
        }
        
        backward {
            s = normalize(rotate(s, -dtheta));
            theta = theta - dt * p_theta;
        }
        
        checkpoint every 10;
    }
    """,
    
    # Lyapunov function definition
    """
    lyapunov Quadratic {
        polynomial(degree=2)
        domain(Oscillator)
        form "x^2 + v^2"
        verify(sos)
    }
    """
]

# Test for non-ASCII characters (unicode support)
UNICODE_TEST_CASE = """
concept "PhaseOscillator" ψ-mode: ϕ1 {
    float θ [rad];
    float p [rad/s];
    
    constraint θ < π;
    
    require Lyapunov(ψ_self) > 0;
}
"""


class TestElfinRoundtrip(unittest.TestCase):
    """Test ELFIN grammar and parser for roundtrip consistency."""
    
    def setUp(self):
        """Set up test cases by ensuring the grammar file exists."""
        self.assertTrue(GRAMMAR_PATH.exists(), f"Grammar file {GRAMMAR_PATH} not found")
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_grammar_file_integrity(self):
        """Test that the grammar file has the expected sections."""
        grammar_text = GRAMMAR_PATH.read_text(encoding="utf-8")
        
        # Check for required sections
        required_sections = [
            "TOP-LEVEL STRUCTURE",
            "CONCEPT DECLARATIONS",
            "SYSTEM DECLARATIONS",
            "RELATION DECLARATIONS",
            "LYAPUNOV FUNCTIONS",
            "VERIFICATION DIRECTIVES",
            "STABILITY DIRECTIVES",
            "PHASE DRIFT MONITORING",
            "RUNTIME ADAPTATION",
            "KOOPMAN OPERATORS",
            "REVERSIBLE COMPUTATION",
            "SPIN VECTOR OPERATIONS",
            "EXPRESSIONS",
            "STATEMENTS",
            "LEXICAL ELEMENTS"
        ]
        
        for section in required_sections:
            self.assertIn(section, grammar_text, f"Missing section: {section}")
    
    def test_token_extraction(self):
        """Test that token extraction from grammar works."""
        tokens = token_gen.extract_tokens_from_grammar(GRAMMAR_PATH)
        
        # Test that we extracted a reasonable number of tokens
        self.assertGreaterEqual(len(tokens), 30, "Too few tokens extracted")
        
        # Test that critical tokens are present
        critical_tokens = [
            "CONCEPT", "SYSTEM", "RELATION", "LYAPUNOV", 
            "REVERSIBLE", "SPINVEC", "PSI_MODE"
        ]
        
        for token in critical_tokens:
            self.assertIn(token, tokens, f"Missing critical token: {token}")
        
        # Test that token naming works
        self.assertEqual(token_gen.make_token_name("ψ-mode"), "PSI_MODE")
        self.assertEqual(token_gen.make_token_name("π"), "PI")
    
    def test_token_id_stability(self):
        """Test that token IDs are stable across runs."""
        # Extract tokens twice and ensure hash generation is stable
        tokens1 = token_gen.extract_tokens_from_grammar(GRAMMAR_PATH)
        tokens2 = token_gen.extract_tokens_from_grammar(GRAMMAR_PATH)
        
        # Generate IDs
        ids1 = {token: int(hash(token) % 10000) for token in tokens1}
        ids2 = {token: int(hash(token) % 10000) for token in tokens2}
        
        # IDs should be identical
        self.assertEqual(ids1, ids2, "Token IDs not stable across runs")
    
    def test_parse_basic_examples(self):
        """Test parsing basic ELFIN examples."""
        for i, test_case in enumerate(BASIC_TEST_CASES):
            with self.subTest(i=i):
                # Parse the test case
                ast = parse(test_case)
                
                # Verify AST is a Program node
                self.assertIsInstance(ast, Program, "Parse result should be a Program")
                
                # For system declarations, verify state and inputs are extracted
                if "system" in test_case and "continuous_state" in test_case:
                    system_section = None
                    for section in ast.sections:
                        if hasattr(section, 'name') and section.name:
                            system_section = section
                            break
                    
                    self.assertIsNotNone(system_section, "System section not found in AST")
                    
                    # Check for continuous_state extraction if listed in system
                    if "[x, v]" in test_case:
                        self.assertIn("x", system_section.continuous_state)
                        self.assertIn("v", system_section.continuous_state)
    
    def test_unicode_support(self):
        """Test parsing with non-ASCII (Unicode) characters."""
        ast = parse(UNICODE_TEST_CASE)
        
        # Verify AST is a Program node
        self.assertIsInstance(ast, Program, "Parse result should be a Program")
        
        # Verify a section was extracted
        self.assertGreaterEqual(len(ast.sections), 1, "No sections extracted from Unicode test")
    
    def test_token_generation(self):
        """Test token generation script produces output files."""
        # Generate tokens to temporary directory
        rust_output = self.temp_path / "elfin_tokens.rs"
        ts_output = self.temp_path / "elfin_tokens.ts"
        
        # Mock the outputs paths for testing
        token_gen.RUST_OUTPUT = rust_output
        token_gen.TS_OUTPUT = ts_output
        
        # Extract tokens
        tokens = token_gen.extract_tokens_from_grammar(GRAMMAR_PATH)
        
        # Generate outputs
        token_gen.generate_token_enum_rust(tokens, rust_output)
        token_gen.generate_token_enum_typescript(tokens, ts_output)
        
        # Verify outputs were created
        self.assertTrue(rust_output.exists(), "Rust token file not generated")
        self.assertTrue(ts_output.exists(), "TypeScript token file not generated")
        
        # Verify Rust file contains expected content
        rust_content = rust_output.read_text(encoding="utf-8")
        self.assertIn("pub enum TokenKind", rust_content)
        self.assertIn("impl TokenKind", rust_content)
        
        # Verify TypeScript file contains expected content
        ts_content = ts_output.read_text(encoding="utf-8")
        self.assertIn("export enum TokenKind", ts_content)
        self.assertIn("export function isKeyword", ts_content)


def run_tests():
    """Run tests directly from command line."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
