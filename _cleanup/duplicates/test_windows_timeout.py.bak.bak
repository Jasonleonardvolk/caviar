#!/usr/bin/env python3
"""
🧪 WINDOWS TIMEOUT TEST
Test the Windows-compatible timeout mechanism before running the full server
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_timeout_mechanism():
    """Test that the Windows timeout mechanism works"""
    
    print("🧪 TESTING WINDOWS TIMEOUT MECHANISM")
    print("=" * 40)
    
    def slow_function(duration):
        """A function that takes a specific amount of time"""
        print(f"😴 Starting slow function (will run for {duration}s)")
        time.sleep(duration)
        print(f"✅ Slow function completed after {duration}s")
        return f"Success after {duration}s"
    
    def test_with_timeout(duration, timeout):
        """Test the timeout mechanism"""
        print(f"\n🔬 Test: {duration}s function with {timeout}s timeout")
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(slow_function, duration)
                result = future.result(timeout=timeout)
                print(f"✅ Result: {result}")
                return True
        except FuturesTimeoutError:
            print(f"⏰ TIMEOUT after {timeout}s (expected)")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # Test 1: Function completes within timeout
    success1 = test_with_timeout(2, 5)  # 2s function, 5s timeout
    
    # Test 2: Function times out
    success2 = not test_with_timeout(8, 3)  # 8s function, 3s timeout (should timeout)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"✅ Fast function test: {'PASS' if success1 else 'FAIL'}")
    print(f"⏰ Timeout test: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n🎉 WINDOWS TIMEOUT MECHANISM WORKING!")
        print("✅ Ready to run the protected server")
        return True
    else:
        print("\n❌ TIMEOUT MECHANISM FAILED")
        print("⚠️ May need alternative approach")
        return False

def test_extraction_import():
    """Test that we can import the extraction function"""
    
    print("\n🔬 TESTING EXTRACTION IMPORT")
    print("-" * 30)
    
    try:
        from ingest_pdf.extractConceptsFromDocument import extractConceptsFromDocument
        print("✅ Successfully imported extractConceptsFromDocument")
        
        # Test with small text
        test_text = "machine learning artificial intelligence neural networks"
        print(f"🧪 Testing with small text: '{test_text}'")
        
        start_time = time.time()
        result = extractConceptsFromDocument(test_text, 0.0)
        end_time = time.time()
        
        print(f"✅ Extraction completed in {end_time - start_time:.2f}s")
        print(f"📊 Result type: {type(result)}")
        if isinstance(result, list):
            print(f"📊 Found {len(result)} concepts")
            if result:
                print(f"📊 First concept: {result[0] if result else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🧪 WINDOWS COMPATIBILITY TEST SUITE")
    print("=" * 50)
    
    # Test 1: Timeout mechanism
    timeout_works = test_timeout_mechanism()
    
    # Test 2: Extraction import
    import_works = test_extraction_import()
    
    print("\n📋 FINAL RESULTS:")
    print("=" * 30)
    print(f"⏰ Timeout mechanism: {'✅ WORKING' if timeout_works else '❌ FAILED'}")
    print(f"🧬 Extraction import: {'✅ WORKING' if import_works else '❌ FAILED'}")
    
    if timeout_works and import_works:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Ready to run the Windows-protected server:")
        print('   "C:\\Users\\jason\\Desktop\\tori\\kha\\START_WINDOWS_PROTECTED.bat"')
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Need to debug further before running protected server")

if __name__ == "__main__":
    main()
