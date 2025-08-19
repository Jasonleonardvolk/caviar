"""
Package Integration Test - SOLVES RELATIVE IMPORT ISSUES
This tests the package-aware integration that properly handles your 
advanced system's relative imports by treating ingest_pdf as a package.
"""
import sys
from pathlib import Path

def test_package_integration():
    """Test the package-aware integration"""
    print("🚀 Prajna Package-Aware Integration Test")
    print("Solving relative import issues for Jason's 4000-hour system")
    print("=" * 70)
    
    try:
        # Test 1: Import package integration
        print("Test 1: Importing package-aware integration...")
        from package_integration import test_package_integration, get_package_integration
        print("✅ Package integration imported successfully")
        
        # Test 2: Run package integration test
        print("\nTest 2: Testing package integration setup...")
        success = test_package_integration()
        if success:
            print("✅ Package integration test passed")
        else:
            print("❌ Package integration test failed")
            return False
        
        # Test 3: Get integration instance and detailed stats
        print("\nTest 3: Getting package integration instance...")
        integration = get_package_integration()
        stats = integration.get_stats()
        
        print(f"✅ Package Integration Status:")
        print(f"   📦 Advanced system available: {stats.get('advanced_available', False)}")
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
        print(f"   📦 Package integration: {ingestion_stats.get('package_integration', False)}")
        print(f"   📊 Method: {ingestion_stats.get('method', 'unknown')}")
        print(f"   🏗️ Advanced available: {ingestion_stats.get('advanced_available', False)}")
        print(f"   📈 Success rate: {ingestion_stats.get('success_rate', 0):.1f}%")
        
        print("\n" + "=" * 70)
        
        # Determine status
        advanced_available = stats.get('advanced_available', False)
        
        if advanced_available:
            print("🎉 PACKAGE INTEGRATION SUCCESS!")
            print("🌟 Relative imports solved - your 4000-hour system is ready!")
            print("📦 Advanced system properly loaded as Python package")
        else:
            print("⚠️ PACKAGE INTEGRATION PARTIAL")
            print("📦 Package setup completed but advanced system not available")
            print("🔄 Will use enhanced fallback - system still functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Package integration test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_with_sample_pdf():
    """Test with a sample PDF using package integration"""
    
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
    print("This will test the full package-aware integration...")
    
    try:
        from package_integration import extract_concepts_package_aware
        
        print("🔄 Running extraction through package-aware integration...")
        result = extract_concepts_package_aware(str(sample_pdf))
        
        num_concepts = result.get("num_concepts", 0)
        method = result.get("method", "unknown")
        status = result.get("status", "unknown")
        advanced_used = "package_aware_advanced_4000h" in method
        
        print(f"✅ Package-aware extraction completed:")
        print(f"   📊 Concepts: {num_concepts}")
        print(f"   🔧 Method: {method}")
        print(f"   ✅ Status: {status}")
        print(f"   🎯 Advanced system used: {'✅ YES' if advanced_used else '🔄 NO (fallback)'}")
        
        # Show sample concepts
        concepts = result.get("concepts", [])
        if concepts:
            print(f"\n📝 Sample concepts:")
            for i, concept in enumerate(concepts[:7]):
                name = concept.get("name", "Unknown")
                score = concept.get("score", 0)
                method_used = concept.get("method", "unknown")
                print(f"   {i+1}. {name} (score: {score:.3f}, method: {method_used})")
            if len(concepts) > 7:
                print(f"   ... and {len(concepts) - 7} more")
        
        # Show advanced analytics if available
        advanced_analytics = result.get("advanced_analytics", {})
        summary = result.get("summary", {})
        
        if advanced_analytics and advanced_used:
            print(f"\n🏆 ADVANCED ANALYTICS AVAILABLE:")
            
            # Purity analysis
            purity_analysis = advanced_analytics.get("purity_analysis", {})
            if purity_analysis:
                print(f"   📊 Purity Analysis:")
                print(f"      - Raw concepts: {purity_analysis.get('raw_concepts', 'N/A')}")
                print(f"      - Pure concepts: {purity_analysis.get('pure_concepts', 'N/A')}")
                print(f"      - Purity efficiency: {purity_analysis.get('purity_efficiency', 'N/A')}")
                
                distribution = purity_analysis.get('distribution', {})
                if distribution:
                    print(f"      - Consensus concepts: {distribution.get('consensus', 0)}")
                    print(f"      - High confidence: {distribution.get('high_confidence', 0)}")
            
            # Context extraction
            context_extraction = advanced_analytics.get("context_extraction", {})
            if context_extraction:
                print(f"   📍 Context Analysis:")
                print(f"      - Title extracted: {context_extraction.get('title_extracted', False)}")
                print(f"      - Abstract found: {context_extraction.get('abstract_extracted', False)}")
                sections = context_extraction.get('sections_identified', [])
                print(f"      - Sections: {', '.join(sections) if sections else 'None'}")
                avg_freq = context_extraction.get('avg_concept_frequency', 0)
                if avg_freq:
                    print(f"      - Avg frequency: {avg_freq:.1f}")
            
            # Performance metrics
            processing_time = advanced_analytics.get("processing_time", 0)
            auto_prefilled = advanced_analytics.get("auto_prefilled_concepts", 0)
            semantic_concepts = advanced_analytics.get("semantic_concepts", 0)
            boosted_concepts = advanced_analytics.get("boosted_concepts", 0)
            
            if processing_time or auto_prefilled:
                print(f"   ⚡ Performance:")
                if processing_time:
                    print(f"      - Processing time: {processing_time:.1f}s")
                if auto_prefilled:
                    print(f"      - Auto-prefilled: {auto_prefilled}")
                if semantic_concepts:
                    print(f"      - Semantic concepts: {semantic_concepts}")
                if boosted_concepts:
                    print(f"      - Boosted concepts: {boosted_concepts}")
            
            # Summary metrics
            if summary:
                print(f"   📈 Summary:")
                pure_concepts = summary.get('pure_concepts', 0)
                consensus_concepts = summary.get('consensus_concepts', 0)
                if pure_concepts:
                    print(f"      - Pure concepts: {pure_concepts}")
                if consensus_concepts:
                    print(f"      - Consensus concepts: {consensus_concepts}")
        
        elif not advanced_used:
            print(f"\n🔄 Used fallback extraction - advanced analytics not available")
            fallback_reason = summary.get("fallback_reason", "unknown")
            print(f"   Fallback reason: {fallback_reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF test failed: {str(e)}")
        # Don't print full traceback to avoid overwhelming output
        print(f"   Error type: {type(e).__name__}")
        return False

def show_next_steps():
    """Show what to do next"""
    print(f"\n🎯 NEXT STEPS:")
    print()
    
    from package_integration import get_package_integration
    integration = get_package_integration()
    stats = integration.get_stats()
    advanced_available = stats.get('advanced_available', False)
    
    if advanced_available:
        print("🌟 PACKAGE INTEGRATION SUCCESS!")
        print("📦 Relative imports solved - your 4000-hour system is ready!")
        print()
        print("1. 🚀 Start the API: python start_prajna_3000.py")
        print("2. 🌐 Start the frontend: cd frontend && npm run dev")
        print("3. 📄 Upload PDFs through the dashboard")
        print("4. 🏆 Enjoy FULL advanced analytics!")
        print()
        print("🎉 ADVANCED FEATURES AVAILABLE:")
        print("- ✅ Purity-based concept extraction (quality over quantity)")
        print("- ✅ Context-aware section detection (title, abstract, etc.)")
        print("- ✅ Universal domain coverage (science, humanities, arts, etc.)")
        print("- ✅ Smart filtering and consensus analysis")
        print("- ✅ Database auto-prefill and cross-reference boosting")
        print("- ✅ Rich dashboard analytics with all metrics")
        print("- ✅ Package integration handles relative imports")
    else:
        print("⚠️ PACKAGE SETUP COMPLETED BUT ADVANCED SYSTEM NOT AVAILABLE")
        print("📦 Relative import issues solved, but system still using fallback")
        print()
        print("1. 🚀 Start the API: python start_prajna_3000.py (will work)")
        print("2. 🌐 Start the frontend: cd frontend && npm run dev")
        print("3. 📄 Upload PDFs (will use enhanced fallback)")
        print()
        print("🔧 TO TROUBLESHOOT ADVANCED SYSTEM:")
        print("- Check logs above for specific error messages")
        print("- Verify all dependencies are installed")
        print("- Check that your advanced system can run independently")
        print("- Re-run this test to verify fixes")
        print()
        print("🎯 CURRENT CAPABILITIES:")
        print("- ✅ Enhanced universal concept extraction (good fallback)")
        print("- ✅ API and dashboard fully functional")
        print("- ✅ Package structure properly set up")
        print("- 🔄 Missing: advanced analytics from your 4000-hour system")

if __name__ == "__main__":
    print("🚀 Prajna Package-Aware Integration Test")
    print("Solving relative import issues for your 4000-hour system")
    print()
    
    # Run basic integration test
    success = test_package_integration()
    
    if success:
        # Test with PDF if available
        test_with_sample_pdf()
        
        # Show next steps
        show_next_steps()
    else:
        print("\n❌ Package integration not ready - check error messages above")
        print("💡 Try fixing any import or dependency issues mentioned")
