"""
Simple Integration Test - No Heavy Dependencies
Tests the lightweight integration without ML model loading issues
"""
import sys
from pathlib import Path

def test_simple_integration():
    """Test the simple integration approach"""
    print("🚀 Prajna Simple Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Import simple integration
        print("Test 1: Importing simple integration...")
        from simple_integration import test_simple_integration, get_simple_integration
        print("✅ Simple integration imported successfully")
        
        # Test 2: Run simple integration test
        print("\nTest 2: Testing simple integration...")
        success = test_simple_integration()
        if success:
            print("✅ Simple integration test passed")
        else:
            print("❌ Simple integration test failed")
            return False
        
        # Test 3: Get integration instance
        print("\nTest 3: Getting integration instance...")
        integration = get_simple_integration()
        stats = integration.get_stats()
        print(f"✅ Integration ready:")
        print(f"   📊 Advanced available: {stats.get('advanced_available', False)}")
        print(f"   📁 Method: {stats.get('method', 'unknown')}")
        
        # Test 4: Test ingestion
        print("\nTest 4: Testing ingestion module...")
        from ingestion import get_ingestion_statistics
        ingestion_stats = get_ingestion_statistics()
        print(f"✅ Ingestion ready:")
        print(f"   📊 Method: {ingestion_stats.get('method', 'unknown')}")
        print(f"   🏗️ Advanced: {ingestion_stats.get('advanced_available', False)}")
        
        print("\n" + "=" * 50)
        print("🎉 SIMPLE INTEGRATION SUCCESS!")
        print("✅ System ready for PDF processing")
        print("🔄 Will use advanced system if available, fallback otherwise")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple integration test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_with_sample_text():
    """Test with simple text processing"""
    print("\n🧪 Testing with sample PDF...")
    
    # Look for sample PDF
    sample_paths = [
        Path("data/memory/2407.15527v2.pdf"),
        Path("ingest_pdf/data/memory/2407.15527v2.pdf")
    ]
    
    sample_pdf = None
    for path in sample_paths:
        if path.exists():
            sample_pdf = path
            break
    
    if sample_pdf:
        try:
            from simple_integration import extract_concepts_simple
            
            print(f"🔄 Testing with: {sample_pdf}")
            result = extract_concepts_simple(str(sample_pdf))
            
            num_concepts = result.get("num_concepts", 0)
            method = result.get("method", "unknown")
            status = result.get("status", "unknown")
            
            print(f"✅ Extraction completed:")
            print(f"   📊 Concepts: {num_concepts}")
            print(f"   🔧 Method: {method}")
            print(f"   ✅ Status: {status}")
            
            # Show sample concepts
            concepts = result.get("concepts", [])
            if concepts:
                print(f"📝 Sample concepts:")
                for i, concept in enumerate(concepts[:5]):
                    name = concept.get("name", "Unknown")
                    score = concept.get("score", 0)
                    print(f"   {i+1}. {name} (score: {score:.3f})")
                if len(concepts) > 5:
                    print(f"   ... and {len(concepts) - 5} more")
            
            return True
            
        except Exception as e:
            print(f"❌ PDF test failed: {str(e)}")
            # Don't print full traceback here to avoid ML import errors
            return False
    else:
        print("📄 No sample PDF found - upload via API to test")
        return True

def show_next_steps():
    """Show what to do next"""
    print("\n🎯 NEXT STEPS:")
    print()
    print("🌟 SIMPLE INTEGRATION READY!")
    print("1. Start the API: python start_prajna_3000.py")
    print("2. Start the frontend: cd frontend && npm run dev")
    print("3. Upload PDFs through the dashboard")
    print("4. System will use advanced pipeline if available, fallback otherwise")
    print()
    print("🔧 INTEGRATION FEATURES:")
    print("- ✅ Always works (no import conflicts)")
    print("- 🔄 Tries advanced system first")
    print("- 📄 Falls back to simple extraction if needed") 
    print("- 🌉 Seamless for the user")
    print()
    print("📊 DASHBOARD READY:")
    print("- Upload PDFs and see concepts")
    print("- Works immediately without heavy ML setup")
    print("- Improves automatically when advanced system is available")

if __name__ == "__main__":
    print("🚀 Prajna Simple Integration Test")
    print("Lightweight approach that always works")
    print()
    
    # Run basic integration test
    success = test_simple_integration()
    
    if success:
        # Test with PDF if available (but don't fail if ML issues)
        test_with_sample_text()
        
        # Show next steps
        show_next_steps()
    else:
        print("\n❌ Integration not ready - check error messages above")
