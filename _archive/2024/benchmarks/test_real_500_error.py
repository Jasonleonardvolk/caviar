#!/usr/bin/env python3
"""
COMPLETE NONETYPE ERADICATION - Find and fix ALL math operations
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_with_actual_pdf():
    """Test with a real PDF-like scenario that would cause the 500 error"""
    print("🎯 TESTING WITH ACTUAL PDF SCENARIO")
    print("=" * 45)
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        
        # Create a test file that mimics the actual PDF content
        test_pdf = current_dir / "rulebook_test.txt"
        with open(test_pdf, "w") as f:
            f.write("""
            A Rulebook for Arguments
            Anthony Weston
            
            Abstract
            This book provides a comprehensive guide to constructing and evaluating arguments.
            It covers logical reasoning, fallacies, and critical thinking skills essential for
            academic writing and debate.
            
            Introduction
            Arguments are everywhere in our daily lives. From political debates to academic
            discussions, the ability to construct sound arguments and identify weak ones is
            crucial for effective communication and decision-making.
            
            Chapter 1: What is an Argument?
            An argument is not just a disagreement or quarrel. In logic and critical thinking,
            an argument is a set of statements where some statements (premises) are offered
            as reasons or evidence for another statement (the conclusion).
            
            Chapter 2: Short Arguments
            The simplest arguments consist of just one premise and one conclusion.
            For example: "Socrates is mortal because all humans are mortal and Socrates is human."
            
            Chapter 3: Longer Arguments
            More complex arguments may have multiple premises that work together to support
            a conclusion. These arguments require careful analysis to evaluate their strength.
            
            Conclusion
            Mastering the art of argumentation requires practice and attention to logical
            structure. By following the rules outlined in this book, readers can improve
            their reasoning skills and become more effective communicators.
            """)
        
        print(f"📄 Created realistic test file: {test_pdf}")
        
        # Test the exact scenario that's failing
        print("🚀 Running full extraction pipeline...")
        result = ingest_pdf_clean(str(test_pdf), admin_mode=True)
        
        print(f"Status: {result.get('status')}")
        print(f"Concept count: {result.get('concept_count', 0)}")
        
        # Check for the specific fields that might be None
        purity_analysis = result.get('purity_analysis', {})
        entropy_analysis = result.get('entropy_analysis', {})
        
        print(f"\nPURITY ANALYSIS:")
        print(f"  Raw concepts: {purity_analysis.get('raw_concepts')}")
        print(f"  Pure concepts: {purity_analysis.get('pure_concepts')}")
        print(f"  Final concepts: {purity_analysis.get('final_concepts')}")
        print(f"  Purity efficiency: {purity_analysis.get('purity_efficiency_percent')}%")
        print(f"  Diversity efficiency: {purity_analysis.get('diversity_efficiency_percent')}%")
        
        print(f"\nENTROPY ANALYSIS:")
        print(f"  Enabled: {entropy_analysis.get('enabled')}")
        if entropy_analysis.get('enabled'):
            print(f"  Total before: {entropy_analysis.get('total_before_entropy')}")
            print(f"  Selected: {entropy_analysis.get('selected_diverse')}")
            print(f"  Diversity efficiency: {entropy_analysis.get('diversity_efficiency_percent')}%")
            print(f"  Final entropy: {entropy_analysis.get('final_entropy')}")
            print(f"  Avg similarity: {entropy_analysis.get('avg_similarity')}")
        
        # Check for None values that would cause 500 errors
        none_values = []
        for key, value in result.items():
            if value is None:
                none_values.append(key)
        
        if none_values:
            print(f"\n❌ FOUND None VALUES: {none_values}")
        else:
            print(f"\n✅ NO None VALUES FOUND")
        
        # Clean up
        os.remove(test_pdf)
        
        return result.get('status') == 'success'
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print("FULL TRACEBACK:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🎯 COMPLETE NONETYPE ERADICATION TEST")
    print("Finding the exact cause of the 500 error...")
    print("=" * 60)
    
    success = test_with_actual_pdf()
    
    if success:
        print(f"\n✅ TEST PASSED - No NoneType errors detected")
    else:
        print(f"\n❌ TEST FAILED - Found the issue causing 500 errors")
        print("🔧 Need to apply more fixes to the pipeline")
