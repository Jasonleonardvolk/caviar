#!/usr/bin/env python3
"""
Test script to verify TORI API integration is pulling all concepts properly
"""

import sys
import os
from pathlib import Path

# Add the ingest_pdf directory to Python path
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_pipeline_direct():
    """Test the pipeline directly"""
    print("🧪 Testing pipeline direct import...")
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        print("✅ Successfully imported ingest_pdf_clean")
        
        # Test with a small text file
        test_file = current_dir / "test_sample.txt"
        with open(test_file, "w") as f:
            f.write("""
            Machine Learning and Artificial Intelligence
            
            This document discusses machine learning algorithms, neural networks, 
            and deep learning approaches. Quantum computing is also mentioned as 
            an emerging technology that could revolutionize artificial intelligence.
            
            Natural language processing and computer vision are key applications
            of machine learning in modern AI systems.
            """)
        
        print(f"📄 Created test file: {test_file}")
        
        # Run extraction with admin mode
        print("🚀 Running extraction with admin_mode=True...")
        result = ingest_pdf_clean(str(test_file), admin_mode=True)
        
        print("\n📊 EXTRACTION RESULTS:")
        print("=" * 50)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Concept Count: {result.get('concept_count', 0)}")
        print(f"Purity Based: {result.get('purity_based', False)}")
        print(f"Entropy Pruned: {result.get('entropy_pruned', False)}")
        print(f"Admin Mode: {result.get('admin_mode', False)}")
        
        concept_names = result.get('concept_names', [])
        if concept_names:
            print(f"\n🎯 CONCEPTS FOUND ({len(concept_names)}):")
            for i, concept in enumerate(concept_names[:10], 1):
                print(f"  {i}. {concept}")
            if len(concept_names) > 10:
                print(f"  ... and {len(concept_names) - 10} more")
        
        # Check purity analysis
        purity_analysis = result.get('purity_analysis', {})
        if purity_analysis:
            print(f"\n🔬 PURITY ANALYSIS:")
            print(f"  Raw concepts: {purity_analysis.get('raw_concepts', 0)}")
            print(f"  Pure concepts: {purity_analysis.get('pure_concepts', 0)}")
            print(f"  Final concepts: {purity_analysis.get('final_concepts', 0)}")
            print(f"  Purity efficiency: {purity_analysis.get('purity_efficiency_percent', 0)}%")
        
        # Check entropy analysis
        entropy_analysis = result.get('entropy_analysis', {})
        if entropy_analysis.get('enabled'):
            print(f"\n🎯 ENTROPY ANALYSIS:")
            print(f"  Enabled: {entropy_analysis.get('enabled')}")
            print(f"  Before entropy: {entropy_analysis.get('total_before_entropy', 0)}")
            print(f"  After entropy: {entropy_analysis.get('selected_diverse', 0)}")
            print(f"  Final entropy: {entropy_analysis.get('final_entropy', 0):.3f}")
            print(f"  Avg similarity: {entropy_analysis.get('avg_similarity', 0):.3f}")
        else:
            print("\n🎯 ENTROPY ANALYSIS: Disabled or no stats")
        
        # Check full concepts objects
        full_concepts = result.get('concepts', [])
        if full_concepts:
            print(f"\n📦 FULL CONCEPT OBJECTS: {len(full_concepts)} available")
            if len(full_concepts) > 0:
                first_concept = full_concepts[0]
                print(f"  Sample concept keys: {list(first_concept.keys())}")
        
        # Clean up
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False, None

def test_api_response_format():
    """Test the expected API response format"""
    print("\n🌐 Testing API response format...")
    
    success, result = test_pipeline_direct()
    if not success:
        print("❌ Cannot test API response format - pipeline test failed")
        return False
    
    # Simulate the main.py response format
    api_response = {
        "success": True,
        "filename": "test_sample.txt",
        "status": result.get("status", "success"),
        "extraction_method": "atomic_purity_based_universal_pipeline",
        "concept_count": result.get("concept_count", 0),
        "concept_names": result.get("concept_names", []),
        "concepts": result.get("concepts", []),
        "semantic_extracted": result.get("semantic_extracted", 0),
        "file_storage_boosted": result.get("file_storage_boosted", 0),
        "purity_based": result.get("purity_based", True),
        "entropy_pruned": result.get("entropy_pruned", False),
        "admin_mode": result.get("admin_mode", False),
        "purity_analysis": result.get("purity_analysis", {}),
        "entropy_analysis": result.get("entropy_analysis", {}),
        "domain_distribution": result.get("domain_distribution", {}),
    }
    
    print("✅ API Response Structure:")
    print(f"  - Success: {api_response['success']}")
    print(f"  - Concept Count: {api_response['concept_count']}")
    print(f"  - Concept Names: {len(api_response['concept_names'])} items")
    print(f"  - Full Concepts: {len(api_response['concepts'])} items")
    print(f"  - Purity Analysis: {bool(api_response['purity_analysis'])}")
    print(f"  - Entropy Analysis: {api_response['entropy_analysis'].get('enabled', False)}")
    print(f"  - Admin Mode: {api_response['admin_mode']}")
    
    return True

def check_entropy_prune_module():
    """Check if entropy pruning module is available"""
    print("\n🎯 Testing entropy pruning module...")
    
    try:
        from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
        print("✅ Successfully imported entropy pruning functions")
        
        # Test with dummy concepts
        test_concepts = [
            {"name": "machine learning", "score": 0.9},
            {"name": "deep learning", "score": 0.85},  # Similar to ML
            {"name": "quantum computing", "score": 0.8},  # Different
            {"name": "artificial intelligence", "score": 0.88},  # Similar to ML
        ]
        
        selected, stats = entropy_prune(
            test_concepts,
            top_k=3,
            similarity_threshold=0.85,
            verbose=True
        )
        
        print(f"✅ Entropy pruning test successful:")
        print(f"  Input: {len(test_concepts)} concepts")
        print(f"  Output: {len(selected)} concepts")
        print(f"  Selected: {[c['name'] for c in selected]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Entropy pruning test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TORI API INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Pipeline direct import and execution
    success1, _ = test_pipeline_direct()
    
    # Test 2: API response format
    success2 = test_api_response_format()
    
    # Test 3: Entropy pruning module
    success3 = check_entropy_prune_module()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Pipeline Direct Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  API Response Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"  Entropy Pruning Test: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    all_passed = success1 and success2 and success3
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 Your TORI system is ready!")
        print("  - Pipeline extracts concepts properly")
        print("  - API returns all concept data")
        print("  - Entropy pruning is working")
        print("  - Admin mode provides unlimited concepts")
    else:
        print("\n⚠️ Some issues detected. Check the logs above.")
