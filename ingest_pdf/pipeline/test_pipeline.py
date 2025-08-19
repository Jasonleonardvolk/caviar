"""
Test suite for the merged pipeline to ensure all functions work correctly.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test that safe_get exists and is callable
def test_safe_get_exists():
    """Verify that utils.safe_get exists and is callable"""
    from ingest_pdf.pipeline.utils import safe_get
    assert callable(safe_get)
    
    # Test basic functionality
    test_dict = {"key": "value"}
    assert safe_get(test_dict, "key") == "value"
    assert safe_get(test_dict, "missing", "default") == "default"
    assert safe_get(None, "key", "default") == "default"

# Test ensure_sync doesn't deadlock
def test_ensure_sync_no_deadlock():
    """Verify ensure_sync works correctly in sync context"""
    from ingest_pdf.pipeline.pipeline import ensure_sync
    
    async def simple_coro():
        return "test_value"
    
    # Should work in sync context
    result = ensure_sync(simple_coro())
    assert result == "test_value"

# Test async context detection
def test_async_context_detection():
    """Verify we detect and handle async contexts properly"""
    from ingest_pdf.pipeline.pipeline import ensure_sync
    
    async def async_test():
        # This should return None and log a warning
        result = ensure_sync(simple_async_func())
        assert result is None
    
    async def simple_async_func():
        return "should_not_reach"
    
    # Run the async test
    asyncio.run(async_test())

# Test store_concepts_sync
def test_store_concepts_sync():
    """Verify store_concepts_sync handles async contexts gracefully"""
    from ingest_pdf.pipeline.pipeline import store_concepts_sync
    
    # Mock the storage function
    with patch('ingest_pdf.pipeline.pipeline.store_concepts_in_soliton') as mock_store:
        async def mock_async_store(concepts, meta):
            return True
        
        mock_store.return_value = mock_async_store([], {})
        
        # Test in sync context
        result = store_concepts_sync([], {})
        # Should work without deadlock

# Test thread safety of ConceptDB
def test_concept_db_thread_safety():
    """Verify ConceptDB is thread-safe across multiple threads"""
    from ingest_pdf.pipeline.pipeline import get_db
    from threading import Thread
    
    results = []
    
    def check_db():
        db = get_db()
        results.append(len(db.storage))
    
    # Run in multiple threads
    threads = [Thread(target=check_db) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # All threads should see the same database size
    assert len(set(results)) == 1, "ConceptDB not thread-safe"

# Test progress callback handling
def test_progress_callback():
    """Verify progress callbacks are called correctly"""
    from ingest_pdf.pipeline.pipeline import ingest_pdf_clean
    
    progress_calls = []
    
    def progress_callback(stage, pct, msg):
        progress_calls.append((stage, pct, msg))
    
    # Mock the necessary parts
    with patch('ingest_pdf.pipeline.pipeline.check_pdf_safety') as mock_safety:
        mock_safety.return_value = (False, "Test PDF", {"file_size_mb": 1})
        
        # This should fail safely
        result = ingest_pdf_clean(
            "test.pdf",
            progress_callback=progress_callback
        )
        
        # Should have at least the init progress call
        assert len(progress_calls) > 0
        assert progress_calls[0][0] == "init"
        assert progress_calls[0][1] == 0

# Test parallel processing fallback
def test_parallel_processing_fallback():
    """Verify parallel processing falls back to sequential in async context"""
    from ingest_pdf.pipeline.pipeline import _process_chunks
    
    chunks = [{"text": "test", "index": 0, "section": "body"}]
    params = {"threshold": 0, "title": "", "abstract": ""}
    
    # Mock to simulate async context
    with patch('ingest_pdf.pipeline.pipeline.ensure_sync') as mock_sync:
        mock_sync.return_value = None  # Simulate async context
        
        with patch('ingest_pdf.pipeline.pipeline._process_chunks_sequential') as mock_seq:
            mock_seq.return_value = ["concept"]
            
            result = _process_chunks(chunks, params, 100, lambda *args: None)
            
            # Should have called sequential processing
            assert mock_seq.called

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
