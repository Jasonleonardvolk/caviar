#!/usr/bin/env python3
"""
🧪 TEST PYTHON EXTRACTION DIRECTLY
Test the Python extraction with Windows protection to see why it's failing
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_extraction_with_sample_text():
    """Test extraction with the exact text that was failing"""
    
    print("🧪 TESTING PYTHON EXTRACTION WITH SAMPLE TEXT")
    print("=" * 50)
    
    # Sample text similar to what was causing crashes
    sample_text = """
    IEEE, VOL. X, NO. X, X X 1
    Reinforcement learning Based Automated Design of Differential
    Evolution Algorithm for Black-box Optimization
    Xu Yang, Rui Wang, Kaiwen Li, and Ling Wang,
    Abstract —Differential evolution (DE) algorithm is
    recognized as one of the most effective evolutionary algorithms, designed for solving
    continuous optimization problems. However, the performance of DE largely depends on
    the selection of control parameters and mutation strategies. In this work, we propose
    a reinforcement learning based automated design method for DE algorithm, called RL-DE.
    The proposed method employs deep Q-network (DQN) to automatically select the most
    suitable mutation strategy and control parameters during the evolutionary process.
    Machine learning artificial intelligence neural networks deep learning optimization
    algorithm reinforcement learning differential evolution black-box optimization
    evolutionary computation adaptive parameter control automated algorithm design.
    """
    
    try:
        # Import the extraction function
        from ingest_pdf.extractConceptsFromDocument import extractConceptsFromDocument
        print("✅ Successfully imported extractConceptsFromDocument")
        
        print(f"🔬 Testing with {len(sample_text)} characters")
        print(f"📄 Sample preview: {sample_text[:100]}...")
        
        # Test with timeout protection (same as Windows server)
        def extract_with_timeout(text, threshold=0.0, timeout_seconds=30):
            """Test extraction with timeout"""
            
            print(f"🔧 EXTRACTION TEST: Processing {len(text)} characters with {timeout_seconds}s timeout")
            
            def extract_function():
                return extractConceptsFromDocument(text, threshold)
            
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(extract_function)
                    result = future.result(timeout=timeout_seconds)
                    return result
            except FuturesTimeoutError:
                print(f"⏰ TIMEOUT after {timeout_seconds}s")
                return []
        
        # Test extraction
        start_time = time.time()
        result = extract_with_timeout(sample_text, 0.0, 30)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"✅ Extraction completed in {processing_time:.2f}s")
        print(f"📊 Result type: {type(result)}")
        
        if isinstance(result, list):
            print(f"📊 Found {len(result)} concepts")
            if result:
                print("📋 First few concepts:")
                for i, concept in enumerate(result[:5]):
                    print(f"   {i+1}. {concept}")
            else:
                print("❌ No concepts found!")
        else:
            print(f"⚠️ Unexpected result type: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ EXTRACTION TEST FAILED: {e}")
        import traceback
        print("❌ Full traceback:")
        traceback.print_exc()
        return []

def test_small_vs_large_text():
    """Test if text size affects extraction success"""
    
    print("\n🧪 TESTING TEXT SIZE IMPACT")
    print("-" * 30)
    
    small_text = "machine learning artificial intelligence neural networks optimization"
    large_text = small_text * 100  # Make it much larger
    
    try:
        from ingest_pdf.extractConceptsFromDocument import extractConceptsFromDocument
        
        # Test small text
        print(f"🔬 Testing small text ({len(small_text)} chars)")
        start = time.time()
        small_result = extractConceptsFromDocument(small_text, 0.0)
        small_time = time.time() - start
        print(f"✅ Small text: {len(small_result) if isinstance(small_result, list) else 'error'} concepts in {small_time:.2f}s")
        
        # Test large text
        print(f"🔬 Testing large text ({len(large_text)} chars)")
        start = time.time()
        large_result = extractConceptsFromDocument(large_text, 0.0)
        large_time = time.time() - start
        print(f"✅ Large text: {len(large_result) if isinstance(large_result, list) else 'error'} concepts in {large_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Size test failed: {e}")
        return False

def main():
    """Run extraction tests"""
    
    print("🧪 PYTHON EXTRACTION DEBUGGING")
    print("=" * 50)
    print("Testing why Python extraction fails even with Windows protection")
    print()
    
    # Test 1: Sample text extraction
    result1 = test_extraction_with_sample_text()
    
    # Test 2: Size impact
    result2 = test_small_vs_large_text()
    
    print("\n📊 TEST SUMMARY:")
    print("=" * 30)
    if result1:
        print("✅ Sample text extraction: SUCCESS")
        print(f"   Found {len(result1)} concepts")
    else:
        print("❌ Sample text extraction: FAILED")
    
    print(f"✅ Size impact test: {'SUCCESS' if result2 else 'FAILED'}")
    
    print("\n🎯 NEXT STEPS:")
    if result1:
        print("✅ Python extraction works - issue might be:")
        print("   1. PDF text extraction producing bad input")
        print("   2. Text encoding issues")
        print("   3. API timeout too short")
    else:
        print("❌ Python extraction failing - need to:")
        print("   1. Check extraction function dependencies")
        print("   2. Test with simpler text")
        print("   3. Debug the extraction pipeline")

if __name__ == "__main__":
    main()
