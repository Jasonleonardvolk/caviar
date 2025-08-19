# Fix for asyncio.run() error in PDF processing pipeline

## Problem
The error `asyncio.run() cannot be called from a running event loop` was occurring when trying to process PDFs through the Prajna API. This happens because:

1. The FastAPI endpoint runs in an async context (with an existing event loop)
2. The `safe_pdf_processing` function was calling `ingest_pdf_clean`
3. `ingest_pdf_clean` was trying to use `asyncio.run()` to run the async `ingest_pdf_async` function
4. You cannot call `asyncio.run()` when there's already a running event loop

## Root Cause Found
The `ingest_pdf_async` function was not being exported from the `ingest_pdf.pipeline` module's `__init__.py` file, causing an ImportError when trying to import it.

## Solution Applied

### 1. Fixed pipeline.py
Modified `ingest_pdf_clean` to detect if there's already a running event loop and handle it appropriately:

```python
def ingest_pdf_clean(...):
    # Check if there's already a running event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context (like FastAPI), use run_sync helper
        # which properly handles running async code from sync context
        return run_sync(ingest_pdf_async(
            pdf_path, doc_id, extraction_threshold,
            admin_mode, use_ocr, progress_callback
        ))
    except RuntimeError:
        # No event loop running, we can use asyncio.run()
        return asyncio.run(ingest_pdf_async(
            pdf_path, doc_id, extraction_threshold,
            admin_mode, use_ocr, progress_callback
        ))
```

### 2. Fixed __init__.py in ingest_pdf/pipeline/
Added `ingest_pdf_async` to the module exports:

```python
# Import statement
from .pipeline import ingest_pdf_clean, ingest_pdf_async, preload_concept_database, ProgressTracker, get_db

# Added to __all__ list
__all__ = [
    # Main function
    'ingest_pdf_clean',
    'ingest_pdf_async',  # <-- Added this
    'preload_concept_database',
    # ... rest of exports
]
```

### 3. Fixed prajna_api.py
Made two important changes:

a) Changed `safe_pdf_processing` to be async and use the async version directly:
```python
async def safe_pdf_processing(file_path: str, filename: str) -> Dict[str, Any]:
    # Import and use the async version
    from ingest_pdf.pipeline import ingest_pdf_async
    result = await ingest_pdf_async(
        file_path, 
        extraction_threshold=0.0, 
        admin_mode=True
    )
```

b) Added `await` when calling `safe_pdf_processing`:
```python
extraction_result = await safe_pdf_processing(str(temp_file_path), safe_filename)
```

## Why This Works

1. The `run_sync` helper from `execution_helpers.py` uses `asyncio.run_coroutine_threadsafe` which properly handles running async code when there's already an event loop
2. By making `safe_pdf_processing` async and using `ingest_pdf_async` directly, we avoid the nested event loop issue entirely
3. The fix maintains backward compatibility - `ingest_pdf_clean` still works from sync contexts (CLI, scripts) while also working from async contexts (FastAPI)
4. Adding `ingest_pdf_async` to the module exports ensures it can be imported properly

## Testing
After applying these fixes, the PDF upload should work without the `asyncio.run()` error. The pipeline will:
- Process PDFs successfully through the API
- Extract concepts as expected
- Handle both sync and async contexts properly

## Files Modified
1. `${IRIS_ROOT}\ingest_pdf\pipeline\pipeline.py`
2. `${IRIS_ROOT}\ingest_pdf\pipeline\__init__.py`
3. `${IRIS_ROOT}\prajna\api\prajna_api.py`
