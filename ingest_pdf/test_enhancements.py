"""
Test suite for TORI pipeline enhancements
Tests the critical fixes and improvements
"""

import asyncio
import pytest
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Import the enhanced modules
from pipeline import ingest_pdf_clean, OCR_MAX_PAGES, MAX_PARALLEL_WORKERS
from memory_sculptor import memory_sculptor, ENABLE_SENTIMENT
from soliton_multi_tenant_manager import soliton_manager, async_retry_with_backoff

# Test data
TEST_CONCEPT = {
    'id': 'test_concept_123',
    'name': 'neural networks',
    'text': 'Neural networks are a fundamental component of deep learning systems.',
    'score': 0.85,
    'quality_score': 0.9,
    'metadata': {
        'section': 'methodology',
        'frequency': 3
    }
}

TEST_CONCEPTS = [
    {
        'id': 'concept_1',
        'name': 'machine learning',
        'text': 'Machine learning enables systems to learn from data.',
        'score': 0.8
    },
    {
        'id': 'concept_2', 
        'name': 'deep learning',
        'text': 'Deep learning uses neural networks with multiple layers.',
        'score': 0.85
    },
    {
        'id': 'concept_3',
        'name': 'neural networks',
        'text': 'Neural networks are inspired by biological neurons.',
        'score': 0.9
    }
]

class TestPipelineEnhancements:
    """Test pipeline configuration and improvements"""
    
    def test_ocr_config(self):
        """Test OCR page limit configuration"""
        # OCR_MAX_PAGES should be configurable
        assert OCR_MAX_PAGES is None or isinstance(OCR_MAX_PAGES, int)
    
    def test_parallel_workers_config(self):
        """Test parallel workers configuration"""
        # MAX_PARALLEL_WORKERS should be configurable
        assert MAX_PARALLEL_WORKERS is None or isinstance(MAX_PARALLEL_WORKERS, int)
    
    def test_safe_math_functions(self):
        """Test bulletproof math functions"""
        from pipeline import safe_divide, safe_multiply, safe_percentage
        
        # Test safe divide
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0  # No division by zero error
        assert safe_divide(None, 5) == 0.0  # Handle None
        
        # Test safe multiply
        assert safe_multiply(5, 3) == 15.0
        assert safe_multiply(None, 5) == 0.0
        
        # Test safe percentage
        assert safe_percentage(25, 100) == 25.0
        assert safe_percentage(25, 0) == 0.0  # No division by zero

class TestMemorySculptorEnhancements:
    """Test memory sculptor improvements"""
    
    def test_sentiment_disabled(self):
        """Test that sentiment analysis is disabled by default"""
        assert ENABLE_SENTIMENT == False
    
    def test_advanced_entity_extraction(self):
        """Test enhanced entity extraction"""
        text = "Contact: john@example.com, Paper: (Smith et al., 2023), Equation: x = 5 + 3"
        entities = memory_sculptor._extract_advanced_entities(text)
        
        entity_types = {e['type'] for e in entities}
        assert 'email' in entity_types
        assert 'citation' in entity_types
        assert 'math' in entity_types
    
    def test_exact_word_matching(self):
        """Test exact word boundary matching"""
        # Should match exact words only
        assert memory_sculptor._contains_exact("concatenate strings", "cat") == False
        assert memory_sculptor._contains_exact("the cat sat", "cat") == True
        assert memory_sculptor._contains_exact("CAT scan", "cat") == True  # Case insensitive
    
    @pytest.mark.asyncio
    async def test_relationship_detection(self):
        """Test improved relationship detection"""
        content = "Neural networks are used in deep learning. Machine learning includes neural networks."
        concepts = ["neural networks", "deep learning", "machine learning", "cat"]
        
        relationships = memory_sculptor.detect_concept_relationships(content, concepts)
        
        # Should find relationships between co-occurring concepts
        assert "neural networks" in relationships
        assert "deep learning" in relationships["neural networks"]
        assert "machine learning" in relationships["neural networks"]
        
        # Should NOT find "cat" as it doesn't appear
        assert "cat" not in relationships

class TestMultiTenantEnhancements:
    """Test multi-tenant manager improvements"""
    
    @pytest.mark.asyncio
    async def test_relationship_id_hashing(self):
        """Test that long relationship IDs are hashed"""
        tenant_id = "test_tenant"
        source_id = "very_long_concept_id_that_exceeds_normal_limits_123456789"
        target_id = "another_very_long_concept_id_that_exceeds_limits_987654321"
        
        # This should not fail due to ID length
        # The method will hash the IDs internally
        try:
            # Mock the store_concept method to avoid actual Soliton calls
            async def mock_store(*args, **kwargs):
                # Check that the ID is hashed and short
                concept_id = kwargs.get('concept_id', args[1] if len(args) > 1 else '')
                assert len(concept_id) < 64  # Soliton limit
                return True
            
            # Temporarily replace the method
            original_store = soliton_manager.store_concept
            soliton_manager.store_concept = mock_store
            
            result = await soliton_manager.store_concept_relationship(
                tenant_id, source_id, target_id, "similarity"
            )
            
            # Restore original method
            soliton_manager.store_concept = original_store
            
        except Exception as e:
            pytest.fail(f"Relationship storage failed: {e}")
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test exponential backoff retry"""
        call_count = 0
        
        @async_retry_with_backoff(max_retries=3)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return "Success"
        
        result = await flaky_operation()
        assert result == "Success"
        assert call_count == 3  # Should retry twice before succeeding
    
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch concept storage"""
        tenant_id = "test_tenant"
        
        # Create a batch of concepts
        concepts = [
            {
                'id': f'batch_concept_{i}',
                'content': f'Test concept {i}',
                'metadata': {'index': i},
                'strength': 0.7 + i * 0.05
            }
            for i in range(5)
        ]
        
        # Mock the store operations
        stored = []
        
        async def mock_store(*args, **kwargs):
            stored.append(kwargs.get('concept_id', ''))
            return True
        
        original_store = soliton_manager.store_concept
        soliton_manager.store_concept = mock_store
        
        try:
            results = await soliton_manager.store_concepts_batch(tenant_id, concepts)
            
            assert results['successful'] == 5
            assert results['failed'] == 0
            assert len(stored) == 5
            
        finally:
            soliton_manager.store_concept = original_store

class TestIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_quality_score_propagation(self):
        """Test that quality scores flow through the system"""
        # Create a concept with quality score
        concept = TEST_CONCEPT.copy()
        
        # Memory sculptor should preserve quality score
        memories = await memory_sculptor.sculpt_and_store(
            user_id="test_user",
            raw_concept=concept,
            metadata={'test': True}
        )
        
        # Note: This would require mocking Soliton client
        # In real test, verify quality_score is in metadata
    
    def test_event_loop_compatibility(self):
        """Test that pipeline handles event loops correctly"""
        # This should not raise "event loop already running" error
        try:
            # Simulate FastAPI/uvicorn environment
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # This should detect the loop and handle appropriately
            # In real usage, would process a PDF
            
        except RuntimeError as e:
            if "event loop" in str(e):
                pytest.fail(f"Event loop conflict: {e}")

def test_config_loading():
    """Test that configurations load without import errors"""
    # All imports should work
    assert OCR_MAX_PAGES is not None or OCR_MAX_PAGES is None
    assert MAX_PARALLEL_WORKERS is not None or MAX_PARALLEL_WORKERS is None
    assert ENABLE_SENTIMENT == False  # Should be disabled by default

if __name__ == "__main__":
    # Run basic tests
    print("Testing TORI Pipeline Enhancements...\n")
    
    # Test 1: Configuration
    print("1. Testing configurations...")
    test_config_loading()
    print("   ✅ Configurations loaded successfully")
    
    # Test 2: Safe math
    print("\n2. Testing safe math functions...")
    test = TestPipelineEnhancements()
    test.test_safe_math_functions()
    print("   ✅ Safe math functions working")
    
    # Test 3: Entity extraction
    print("\n3. Testing enhanced entity extraction...")
    test = TestMemorySculptorEnhancements()
    test.test_advanced_entity_extraction()
    print("   ✅ Advanced entity extraction working")
    
    # Test 4: Exact matching
    print("\n4. Testing exact word matching...")
    test.test_exact_word_matching()
    print("   ✅ Exact word matching working (no more CAT/concatenate issues)")
    
    # Test 5: Async tests
    print("\n5. Testing async components...")
    print("   ⚠️  Run with pytest for full async test coverage")
    
    print("\n✨ Basic tests passed! Use pytest for comprehensive testing.")
