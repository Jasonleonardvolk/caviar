"""
Test script for verifying math-aware chunking.

This script tests that the math-aware chunking implementation
correctly preserves LaTeX formulas and prevents them from
being split across chunks.
"""

import os
import sys
import unittest
import tempfile
import pathlib
from typing import List, Dict, Any

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the math-aware extractor
from src.utils.math_aware_extractor import (
    chunk_text_preserving_math,
    METRICS
)


class TestMathAwareChunking(unittest.TestCase):
    """Test case for math-aware chunking"""
    
    def setUp(self):
        """Set up test case"""
        # Reset metrics
        for key in METRICS:
            METRICS[key] = 0
    
    def test_simple_text(self):
        """Test chunking with simple text"""
        text = "This is a simple text without any math formulas."
        chunks = chunk_text_preserving_math(text, max_len=100)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
    
    def test_inline_latex(self):
        """Test chunking with inline LaTeX formulas"""
        text = """
This is a paragraph with an inline LaTeX formula: $E = mc^2$.
This is another paragraph with a different formula: $F = ma$.
This is a third paragraph without any formulas.
"""
        chunks = chunk_text_preserving_math(text, max_len=50)
        
        # Verify no math blocks were split
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
        
        # The chunking algorithm should preserve the formulas
        # by keeping the entire paragraph containing a formula together
        for chunk in chunks:
            chunk_text = chunk["text"]
            
            # Check that no formula is split
            self.assertFalse("$E = mc" in chunk_text and "^2$" not in chunk_text)
            self.assertFalse("$F = m" in chunk_text and "a$" not in chunk_text)
            
            # Check that complete formulas are preserved
            if "$E = mc^2$" in chunk_text:
                self.assertIn("This is a paragraph with an inline LaTeX formula: $E = mc^2$", chunk_text)
            
            if "$F = ma$" in chunk_text:
                self.assertIn("This is another paragraph with a different formula: $F = ma$", chunk_text)
    
    def test_display_latex(self):
        """Test chunking with display LaTeX formulas"""
        text = """
This is a paragraph before a display LaTeX formula.

$$
\\int_{a}^{b} f(x) dx = F(b) - F(a)
$$

This is a paragraph after the formula.
"""
        chunks = chunk_text_preserving_math(text, max_len=50)
        
        # Verify no math blocks were split
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
        
        # Find the chunk with the formula
        formula_chunk = None
        for chunk in chunks:
            if "\\int_" in chunk["text"]:
                formula_chunk = chunk
                break
        
        # Verify the formula was preserved intact
        self.assertIsNotNone(formula_chunk)
        self.assertIn("$$", formula_chunk["text"])
        self.assertIn("\\int_{a}^{b} f(x) dx = F(b) - F(a)", formula_chunk["text"])
    
    def test_complex_document(self):
        """Test chunking with a complex document containing multiple formulas"""
        # Create a test document with paragraphs and formulas
        paragraphs = []
        for i in range(20):
            # Regular paragraph
            paragraphs.append(f"This is paragraph {i+1} with some text content.")
            
            # Every third paragraph, add an inline formula
            if i % 3 == 0:
                paragraphs.append(f"Paragraph {i+1} has an inline formula: $x_{i} = \\frac{{a_{i}}}{{b_{i} + c_{i}}}$.")
            
            # Every fifth paragraph, add a display formula
            if i % 5 == 0:
                paragraphs.append(f"""
Here is a display formula for paragraph {i+1}:

$$
\\begin{{align}}
y_{i} &= \\int_{{0}}^{{\\infty}} e^{{-x}} dx \\\\
&= \\lim_{{n \\to \\infty}} \\sum_{{k=0}}^{{n}} \\frac{{(-1)^k}}{{k!}}
\\end{{align}}
$$

The formula above is very important.
""")
        
        # Join paragraphs with double newlines
        text = "\n\n".join(paragraphs)
        
        # Chunk with a small max_len to force multiple chunks
        chunks = chunk_text_preserving_math(text, max_len=100)
        
        # Verify no math blocks were split
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
        
        # Verify that each chunk has complete formulas
        for chunk in chunks:
            chunk_text = chunk["text"]
            
            # Check that all $ are paired
            dollar_count = chunk_text.count("$")
            self.assertEqual(dollar_count % 2, 0, 
                            f"Chunk has odd number of $ symbols: {dollar_count}\n{chunk_text}")
            
            # Check that all display math environments are closed
            if "\\begin{align}" in chunk_text:
                self.assertIn("\\end{align}", chunk_text)
    
    def test_long_formula_handling(self):
        """Test handling of formulas longer than max_len"""
        # Create a very long formula
        long_formula = "$$\n" + "x + " * 100 + "y = 0\n$$"
        
        text = f"""
This is a paragraph before a very long formula.

{long_formula}

This is a paragraph after the formula.
"""
        
        # Set max_len smaller than the formula
        chunks = chunk_text_preserving_math(text, max_len=50)
        
        # The formula should still be preserved in a single chunk
        # even though it exceeds max_len
        formula_found = False
        for chunk in chunks:
            if "$$" in chunk["text"] and "x + " in chunk["text"] and "y = 0" in chunk["text"]:
                formula_found = True
                break
        
        self.assertTrue(formula_found, "Long formula was not preserved in a single chunk")
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
    
    def test_mixed_content(self):
        """Test chunking with mixed content"""
        text = """
# Introduction

This document contains various types of content including text and math.

## Math Section

Let's look at the formula for the area of a circle:

$$A = \pi r^2$$

Where:
- $r$ is the radius of the circle
- $\pi$ is approximately 3.14159

## Another Section

Here's another formula, the quadratic formula:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

This formula is used to solve quadratic equations of the form $ax^2 + bx + c = 0$.

## Final Section

This section has no formulas, just regular text content.
"""
        
        chunks = chunk_text_preserving_math(text, max_len=150)
        
        # Verify no math blocks were split
        self.assertEqual(METRICS['math_blocks_split_total'], 0)
        
        # Verify that all formulas are preserved correctly
        circle_formula_found = False
        quadratic_formula_found = False
        
        for chunk in chunks:
            chunk_text = chunk["text"]
            
            if "$$A = \\pi r^2$$" in chunk_text:
                circle_formula_found = True
                
                # Check that related inline formulas are in the same chunk
                self.assertIn("$r$", chunk_text)
                self.assertIn("$\\pi$", chunk_text)
            
            if "$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$" in chunk_text:
                quadratic_formula_found = True
                
                # Check that related content is in the same chunk
                self.assertIn("quadratic formula", chunk_text.lower())
                self.assertIn("$ax^2 + bx + c = 0$", chunk_text)
        
        self.assertTrue(circle_formula_found, "Circle area formula not found in chunks")
        self.assertTrue(quadratic_formula_found, "Quadratic formula not found in chunks")


if __name__ == "__main__":
    unittest.main()
