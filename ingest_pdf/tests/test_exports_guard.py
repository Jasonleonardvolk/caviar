"""
Export guard tests to ensure public API consistency.
Run with: pytest tests/test_exports_guard.py
"""

import pytest
from ingest_pdf.pipeline._exports import PUBLIC_API, PIPELINE_EXPORTS


def test_pipeline_exports_match():
    """Ensure pipeline.py exports match the expected list."""
    from ingest_pdf.pipeline import pipeline
    
    if hasattr(pipeline, '__all__'):
        # Check that pipeline.py exports exactly match PIPELINE_EXPORTS
        assert set(pipeline.__all__) == set(PIPELINE_EXPORTS), (
            f"pipeline.py __all__ has drifted!\n"
            f"Expected: {sorted(PIPELINE_EXPORTS)}\n"
            f"Actual: {sorted(pipeline.__all__)}\n"
            f"Difference: {set(pipeline.__all__).symmetric_difference(set(PIPELINE_EXPORTS))}"
        )


def test_package_exports_complete():
    """Ensure all expected exports are available at package level."""
    from ingest_pdf import pipeline
    
    missing_exports = []
    for export in PUBLIC_API:
        if not hasattr(pipeline, export):
            missing_exports.append(export)
    
    assert not missing_exports, (
        f"Missing exports from ingest_pdf.pipeline:\n"
        f"{missing_exports}\n"
        f"Update pipeline/__init__.py to import these from the appropriate modules."
    )


def test_no_unexpected_exports():
    """Warn about exports not in the approved list (optional strictness)."""
    from ingest_pdf import pipeline
    
    if hasattr(pipeline, '__all__'):
        approved_exports = set(PUBLIC_API)
        actual_exports = set(pipeline.__all__)
        unexpected = actual_exports - approved_exports
        
        if unexpected:
            pytest.skip(
                f"Found unexpected exports (not necessarily bad):\n"
                f"{sorted(unexpected)}\n"
                f"Consider adding to _exports.py if intentional."
            )


def test_critical_functions_callable():
    """Ensure critical functions are not just present but actually callable."""
    from ingest_pdf import pipeline
    
    # Test get_db returns ConceptDB
    db = pipeline.get_db()
    assert hasattr(db, 'storage'), "get_db() should return ConceptDB with 'storage' attribute"
    assert hasattr(db, 'search_concepts'), "get_db() should return ConceptDB with search_concepts method"
    
    # Test ProgressTracker is instantiable
    progress = pipeline.ProgressTracker(total=10)
    assert hasattr(progress, 'update_sync'), "ProgressTracker should have update_sync method"
    assert hasattr(progress, 'get_state'), "ProgressTracker should have get_state method"
    
    # Test run_sync is callable (don't actually run it)
    assert callable(pipeline.run_sync), "run_sync should be callable"


def test_backwards_compatibility():
    """Ensure legacy imports still work."""
    # Old style imports should still work
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        assert callable(ingest_pdf_clean)
    except ImportError:
        pytest.fail("Legacy import 'from ingest_pdf.pipeline import ingest_pdf_clean' failed")
    
    try:
        from ingest_pdf.pipeline.config import settings
        assert hasattr(settings, 'entropy_threshold')
    except ImportError:
        pytest.fail("Legacy import 'from ingest_pdf.pipeline.config import settings' failed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
