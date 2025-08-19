"""
Test Suite for Saigon Mesh-to-Text Generator
============================================

Comprehensive tests for the Saigon character-level language generation system.
"""

import sys
import os
import time
import json
import torch
from typing import Dict, List, Any

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from saigon import SaigonGenerator, saigon_generate, mesh_to_text
    from saigon_utils import (
        validate_mesh_path, mesh_to_text_enhanced, 
        format_mesh_relation, mesh_statistics, get_available_relations
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def test_mesh_validation():
    """Test mesh path validation functionality."""
    print("🧪 Testing mesh path validation...")
    
    # Valid mesh paths
    valid_mesh = [
        {"concept": "soliton", "relation": "implies", "context": "memory"},
        {"concept": "coherence", "relation": "supports", "context": "physics"}
    ]
    assert validate_mesh_path(valid_mesh), "Valid mesh should pass validation"
    
    # Invalid mesh paths
    invalid_mesh_1 = "not a list"
    assert not validate_mesh_path(invalid_mesh_1), "String should fail validation"
    
    invalid_mesh_2 = [{"no_concept": "test"}]
    assert not validate_mesh_path(invalid_mesh_2), "Missing concept should fail validation"
    
    print("✅ Mesh validation tests passed")


def test_relationship_templates():
    """Test relationship template formatting."""
    print("🧪 Testing relationship templates...")
    
    # Test available relations
    relations = get_available_relations()
    print(f"Available relations: {relations}")
    assert len(relations) > 0, "Should have available relations"
    
    # Test relationship formatting
    result = format_mesh_relation("concept A", "concept B", "implies")
    assert "concept A" in result and "concept B" in result, "Should contain both concepts"
    
    # Test unknown relation
    result_unknown = format_mesh_relation("A", "B", "unknown_relation")
    assert "A" in result_unknown and "B" in result_unknown, "Should handle unknown relations"
    
    print("✅ Relationship template tests passed")


def test_mesh_to_text_conversion():
    """Test basic mesh-to-text conversion."""
    print("🧪 Testing mesh-to-text conversion...")
    
    test_mesh = [
        {"concept": "entropy", "relation": "implies", "context": "information theory"},
        {"concept": "complexity", "relation": "supports", "context": "systems"}
    ]
    
    # Test basic conversion
    basic_result = mesh_to_text(test_mesh)
    assert len(basic_result) > 0, "Should produce non-empty text"
    
    # Test enhanced conversion
    enhanced_result = mesh_to_text_enhanced(test_mesh)
    assert len(enhanced_result) > 0, "Enhanced conversion should produce text"
    assert len(enhanced_result) > len(basic_result), "Enhanced should be more detailed"
    
    print(f"Basic result: {basic_result}")
    print(f"Enhanced result: {enhanced_result}")
    print("✅ Mesh-to-text conversion tests passed")


def test_saigon_generator_fallback():
    """Test Saigon generator with graceful fallback."""
    print("🧪 Testing Saigon generator fallback...")
    
    # Create generator (may not have trained model yet)
    generator = SaigonGenerator()
    
    test_mesh = [
        {"concept": "consciousness", "relation": "emerges_from", "context": "neural networks"},
        {"concept": "awareness", "relation": "extends", "context": "cognition"}
    ]
    
    # Test generation (should fallback gracefully if no model)
    result = generator.generate(test_mesh, smoothing=True)
    
    assert "text" in result, "Should return text field"
    assert "method" in result, "Should specify generation method"
    assert "audit" in result, "Should include audit information"
    assert len(result["text"]) > 0, "Should produce non-empty text"
    
    print(f"Generation method: {result['method']}")
    print(f"Generated text: {result['text'][:100]}...")
    print("✅ Saigon generator fallback tests passed")


def test_saigon_with_model():
    """Test Saigon generator with trained model if available."""
    print("🧪 Testing Saigon with trained model...")
    
    generator = SaigonGenerator()
    
    # Try to load the model
    model_loaded = generator.load_model()
    print(f"Model loaded: {model_loaded}")
    
    if model_loaded:
        test_mesh = [
            {"concept": "phase coherence", "relation": "enables", "context": "quantum systems"},
            {"concept": "entanglement", "relation": "supports", "context": "information transfer"}
        ]
        
        # Test with LSTM smoothing
        result = generator.generate(test_mesh, smoothing=True, temperature=1.0)
        
        assert result["method"] == "lstm_smoothed", "Should use LSTM when available"
        assert len(result["text"]) > len(result["base_text"]), "LSTM should expand text"
        
        print(f"Base text: {result['base_text']}")
        print(f"LSTM text: {result['text']}")
        print("✅ LSTM generation tests passed")
    else:
        print("⚠️ No trained model available - skipping LSTM tests")


def test_legacy_compatibility():
    """Test legacy function compatibility."""
    print("🧪 Testing legacy compatibility...")
    
    test_mesh = [
        {"concept": "emergence", "relation": "transcends", "context": "complexity"},
        {"concept": "synthesis", "relation": "unifies", "context": "knowledge"}
    ]
    
    # Test legacy function
    result = saigon_generate(test_mesh, smoothing=False)
    assert isinstance(result, str), "Legacy function should return string"
    assert len(result) > 0, "Should produce non-empty result"
    
    print(f"Legacy result: {result}")
    print("✅ Legacy compatibility tests passed")


def test_mesh_statistics():
    """Test mesh statistics functionality."""
    print("🧪 Testing mesh statistics...")
    
    test_mesh = [
        {"concept": "recursion", "relation": "implies", "context": "mathematics"},
        {"concept": "self-reference", "relation": "implies", "context": "logic"},
        {"concept": "iteration", "relation": "supports", "context": "computation"}
    ]
    
    stats = mesh_statistics(test_mesh)
    
    assert stats["total_concepts"] == 3, "Should count all concepts"
    assert stats["unique_concepts"] == 3, "Should count unique concepts"
    assert stats["relation_types"] >= 1, "Should identify relation types"
    
    print(f"Mesh statistics: {stats}")
    print("✅ Mesh statistics tests passed")


def performance_benchmark():
    """Run performance benchmarks."""
    print("🧪 Running performance benchmarks...")
    
    # Create larger mesh for performance testing
    large_mesh = []
    concepts = ["memory", "processing", "storage", "retrieval", "encoding", "decoding"]
    relations = ["implies", "supports", "extends", "enables"]
    
    for i in range(20):
        large_mesh.append({
            "concept": concepts[i % len(concepts)],
            "relation": relations[i % len(relations)],
            "context": f"domain_{i}"
        })
    
    generator = SaigonGenerator()
    
    # Benchmark mesh-to-text conversion
    start_time = time.time()
    for _ in range(100):
        mesh_to_text_enhanced(large_mesh)
    mesh_time = time.time() - start_time
    
    print(f"Mesh-to-text (100 iterations): {mesh_time:.3f}s")
    
    # Benchmark full generation
    start_time = time.time()
    result = generator.generate(large_mesh)
    gen_time = time.time() - start_time
    
    print(f"Full generation: {gen_time:.3f}s")
    print(f"Generated {len(result['text'])} characters")
    print("✅ Performance benchmarks completed")


def main():
    """Run all tests."""
    print("🚀 Starting Saigon Test Suite")
    print("=" * 50)
    
    try:
        test_mesh_validation()
        test_relationship_templates()
        test_mesh_to_text_conversion()
        test_saigon_generator_fallback()
        test_saigon_with_model()
        test_legacy_compatibility()
        test_mesh_statistics()
        performance_benchmark()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ Saigon system is ready for integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
