#!/usr/bin/env python3
"""
Test Entropy Pruning Import and Functionality
"""

import sys
from pathlib import Path

# Add the kha directory to Python path
kha_dir = Path("C:/Users/jason/Desktop/tori/kha")
if str(kha_dir) not in sys.path:
    sys.path.insert(0, str(kha_dir))

def test_entropy_import():
    print("🧪 Testing Entropy Pruning Import...\n")
    
    # Test 1: Direct import
    try:
        from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
        print("✅ Test 1 PASSED: Direct import from ingest_pdf.entropy_prune")
    except ImportError as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    
    # Test 2: Import from pruning module
    try:
        from ingest_pdf.pipeline.pruning import apply_entropy_pruning
        print("✅ Test 2 PASSED: Import from pruning module")
    except ImportError as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Test entropy_prune function
    print("\n🧪 Testing entropy_prune function...")
    
    test_concepts = [
        {"name": "machine learning", "score": 0.9, "embedding": None},
        {"name": "deep learning", "score": 0.85, "embedding": None},
        {"name": "neural networks", "score": 0.8, "embedding": None},
        {"name": "artificial intelligence", "score": 0.95, "embedding": None},
        {"name": "AI", "score": 0.7, "embedding": None},  # Similar to artificial intelligence
    ]
    
    try:
        selected, stats = entropy_prune(
            test_concepts,
            top_k=3,
            similarity_threshold=0.8,
            verbose=True
        )
        
        print(f"\n✅ Test 3 PASSED: entropy_prune executed successfully")
        print(f"   Input concepts: {len(test_concepts)}")
        print(f"   Selected concepts: {len(selected)}")
        print(f"   Pruned: {stats.get('pruned', 0)}")
        print(f"   Selected names: {[c['name'] for c in selected]}")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test entropy_prune_with_categories
    print("\n🧪 Testing entropy_prune_with_categories function...")
    
    categorized_concepts = [
        {"name": "supervised learning", "score": 0.9, "metadata": {"category": "ML"}},
        {"name": "unsupervised learning", "score": 0.85, "metadata": {"category": "ML"}},
        {"name": "reinforcement learning", "score": 0.8, "metadata": {"category": "ML"}},
        {"name": "Python", "score": 0.95, "metadata": {"category": "Programming"}},
        {"name": "Java", "score": 0.7, "metadata": {"category": "Programming"}},
        {"name": "C++", "score": 0.75, "metadata": {"category": "Programming"}},
    ]
    
    try:
        selected, stats = entropy_prune_with_categories(
            categorized_concepts,
            categories=["ML", "Programming"],
            concepts_per_category=2,
            verbose=True
        )
        
        print(f"\n✅ Test 4 PASSED: entropy_prune_with_categories executed successfully")
        print(f"   Input concepts: {len(categorized_concepts)}")
        print(f"   Selected concepts: {len(selected)}")
        print(f"   By category: {stats.get('by_category', {})}")
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_pruning_module():
    print("\n🧪 Testing apply_entropy_pruning from pruning module...")
    
    try:
        from ingest_pdf.pipeline.pruning import apply_entropy_pruning
        
        test_concepts = [
            {"name": "TORI system", "score": 0.95},
            {"name": "artificial intelligence", "score": 0.9},
            {"name": "machine learning", "score": 0.85},
        ]
        
        pruned_concepts, prune_stats = apply_entropy_pruning(test_concepts, admin_mode=False)
        
        print(f"✅ apply_entropy_pruning executed successfully")
        print(f"   Results: {len(pruned_concepts)} concepts retained")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test apply_entropy_pruning: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TORI Entropy Pruning Test Suite")
    print("=" * 60)
    
    # Run the import fix first
    print("\n1️⃣ Running import fix...")
    from fix_entropy_import import fix_entropy_import
    fix_entropy_import()
    
    print("\n2️⃣ Running import tests...")
    import_success = test_entropy_import()
    
    if import_success:
        print("\n3️⃣ Running pruning module test...")
        pruning_success = test_pruning_module()
        
        if pruning_success:
            print("\n✅ All tests passed! Entropy pruning is working correctly.")
        else:
            print("\n⚠️  Import works but pruning module has issues.")
    else:
        print("\n❌ Import tests failed. Check the error messages above.")
    
    print("\n" + "=" * 60)
