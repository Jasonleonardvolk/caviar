"""
Advanced Integration Test - Test Your 4000-Hour System
This specifically tests the robust integration with your sophisticated pipeline
"""
import sys
from pathlib import Path

def test_advanced_integration():
    """Test the robust advanced integration"""
    print("🚀 Prajna Advanced Integration Test")
    print("Testing robust connection to Jason's 4000-hour sophisticated system")
    print("=" * 70)
    
    try:
        # Test 1: Import advanced integration
        print("Test 1: Importing advanced integration...")
        from advanced_integration import test_advanced_integration, get_advanced_integration
        print("✅ Advanced integration imported successfully")
        
        # Test 2: Run advanced integration test
        print("\nTest 2: Testing advanced system detection...")
        success = test_advanced_integration()
        if success:
            print("✅ Advanced integration test passed")
        else:
            print("❌ Advanced integration test failed")
            return False
        
        # Test 3: Get integration instance and detailed stats
        print("\nTest 3: Getting integration instance and stats...")
        integration = get_advanced_integration()
        stats = integration.get_stats()
        
        print(f"✅ Advanced Integration Status:")
        print(f"   📊 Advanced system available: {stats.get('advanced_available', False)}")
        print(f"   📁 Integration method: {stats.get('method', 'unknown')}")
        print(f"   📈 Documents processed: {stats.get('total_documents_processed', 0)}")
        print(f"   🎯 Advanced successes: {stats.get('total_advanced_successes', 0)}")
        print(f"   🔄 Fallback uses: {stats.get('total_fallback_uses', 0)}")
        print(f"   📊 Success rate: {stats.get('success_rate', 0):.1f}%")
        
        # Test 4: Test ingestion module
        print("\nTest 4: Testing ingestion module...")
        from ingestion import get_ingestion_statistics
        ingestion_stats = get_ingestion_statistics()
        
        print(f"✅ Ingestion System Ready:")
        print(f"   📊 Method: {ingestion_stats.get('method', 'unknown')}")
        print(f"   🏗️ Advanced available: {ingestion_stats.get('advanced_available', False)}")
        print(f"   📈 Success rate: {ingestion_stats.get('success_rate', 0):.1f}%")
        
        print("\n" + "=" * 70)
        
        # Determine status
        advanced_available = stats.get('advanced_available', False)
        
        if advanced_available:
            print("🎉 FULL ADVANCED INTEGRATION SUCCESS!")
            print("🌟 Your 4000-hour sophisticated system is ready to use!")
            print("📊 Will provide rich analytics: purity analysis, context extraction, etc.")
        else:
            print("⚠️ ADVANCED SYSTEM NOT AVAILABLE - ENHANCED FALLBACK READY")
            print("🔄 System will use enhanced universal extraction as fallback")
            print("📋 Advanced system detection failed - check logs for details")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced integration test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_with_sample_pdf():
    """Test with a sample PDF using the advanced system"""
    
    # Look for sample PDFs
    sample_paths = [
        Path("data/memory/2407.15527v2.pdf"),
        Path("ingest_pdf/data/memory/2407.15527v2.pdf"),
        Path("C:/Users/jason/Desktop/tori/kha/data/memory/2407.15527v2.pdf")
    ]
    
    sample_pdf = None
    for path in sample_paths:
        if path.exists():
            sample_pdf = path
            break
    
    if not sample_pdf:
        print(f"\n📄 No sample PDF found in expected locations:")
        for path in sample_paths:
            print(f"   - {path}")
        print("   Upload a PDF through the API to test extraction")
        return True
    
    print(f"\n🧪 Testing with sample PDF: {sample_pdf}")
    
    try:
        from advanced_integration import extract_concepts_advanced
        
        print("🔄 Running extraction through robust advanced integration...")
        result = extract_concepts_advanced(str(sample_pdf))
        
        num_concepts = result.get("num_concepts", 0)
        method = result.get("method", "unknown")
        status = result.get("status", "unknown")
        advanced_used = "advanced_4000h_system" in method
        
        print(f"✅ Extraction completed:")
        print(f"   📊 Concepts: {num_concepts}")
        print(f"   🔧 Method: {method}")
        print(f"   ✅ Status: {status}")
        print(f"   🎯 Advanced system used: {'✅ YES' if advanced_used else '🔄 NO (fallback)'}")
        
        # Show sample concepts
        concepts = result.get("concepts", [])
        if concepts:
            print(f"\n📝 Sample concepts:")
            for i, concept in enumerate(concepts[:5]):
                name = concept.get("name", "Unknown")
                score = concept.get("score", 0)
                method_used = concept.get("method", "unknown")
                print(f"   {i+1}. {name} (score: {score:.3f}, method: {method_used})")
            if len(concepts) > 5:
                print(f"   ... and {len(concepts) - 5} more")
        
        # Show advanced analytics if available
        advanced_analytics = result.get("advanced_analytics", {})
        if advanced_analytics and advanced_used:
            print(f"\n🏆 Advanced Analytics Available:")
            
            purity_analysis = advanced_analytics.get("purity_analysis", {})
            if purity_analysis:
                print(f"   📊 Purity Analysis:")
                print(f"      - Raw concepts: {purity_analysis.get('raw_concepts', 'N/A')}")
                print(f"      - Pure concepts: {purity_analysis.get('pure_concepts', 'N/A')}")
                print(f"      - Efficiency: {purity_analysis.get('purity_efficiency', 'N/A')}")
            
            context_extraction = advanced_analytics.get("context_extraction", {})
            if context_extraction:
                print(f"   📍 Context Extraction:")
                print(f"      - Title extracted: {context_extraction.get('title_extracted', False)}")
                print(f"      - Abstract found: {context_extraction.get('abstract_extracted', False)}")
                sections = context_extraction.get('sections_identified', [])
                print(f"      - Sections: {', '.join(sections) if sections else 'None'}")
            
            performance = advanced_analytics.get("processing_time", 0)
            if performance:
                print(f"   ⚡ Processing time: {performance:.1f}s")
        
        elif not advanced_used:
            print(f"\n🔄 Used fallback extraction - advanced analytics not available")
            print(f"   Check the advanced system setup if you expected to use the 4000-hour system")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def show_next_steps():
    """Show what to do next"""
    print(f"\n🎯 NEXT STEPS:")
    print()
    
    from advanced_integration import get_advanced_integration
    integration = get_advanced_integration()
    stats = integration.get_stats()
    advanced_available = stats.get('advanced_available', False)
    
    if advanced_available:
        print("🌟 FULL ADVANCED INTEGRATION READY!")
        print("Your 4000-hour sophisticated system is fully operational!")
        print()
        print("1. 🚀 Start the API: python start_prajna_3000.py")
        print("2. 🌐 Start the frontend: cd frontend && npm run dev")
        print("3. 📄 Upload PDFs through the dashboard")
        print("4. 🏆 Enjoy rich analytics from your advanced system!")
        print()
        print("🎉 FEATURES AVAILABLE:")
        print("- ✅ Purity-based concept extraction")
        print("- ✅ Context-aware section detection")
        print("- ✅ Universal domain coverage")
        print("- ✅ Smart filtering and consensus analysis")
        print("- ✅ Database auto-prefill")
        print("- ✅ Rich dashboard analytics")
    else:
        print("⚠️ ADVANCED SYSTEM NOT AVAILABLE")
        print("Enhanced fallback system ready as backup!")
        print()
        print("1. 🚀 Start the API: python start_prajna_3000.py")
        print("2. 🌐 Start the frontend: cd frontend && npm run dev") 
        print("3. 📄 Upload PDFs (will use enhanced fallback)")
        print()
        print("🔧 TO ENABLE ADVANCED SYSTEM:")
        print("- Check that all files exist in ingest_pdf/")
        print("- Install missing dependencies if any")
        print("- Check error logs for specific issues")
        print("- Re-run this test to verify")
        print()
        print("🎯 CURRENT CAPABILITIES:")
        print("- ✅ Universal concept extraction (fallback)")
        print("- ✅ API and dashboard functional")
        print("- 🔄 Missing: purity analysis, context awareness, etc.")

if __name__ == "__main__":
    print("🚀 Prajna Advanced Integration Test")
    print("Testing robust connection to Jason's 4000-hour sophisticated system")
    print()
    
    # Run basic integration test
    success = test_advanced_integration()
    
    if success:
        # Test with PDF if available
        test_with_sample_pdf()
        
        # Show next steps
        show_next_steps()
    else:
        print("\n❌ Advanced integration not ready - check error messages above")
        print("💡 Try fixing any import or dependency issues mentioned")
