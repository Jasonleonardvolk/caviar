# Code Review Fixes - Implementation Summary

Based on your excellent code review, I've implemented the following quick fixes:

## 1. ✅ Fixed Thread-Pool Churn in `run_sync`

**Issue**: Creating a new event loop for every `run_sync` call was wasteful.

**Fix**: Added a module-level thread pool executor with a single thread:
```python
_sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tori-sync")
```

This reuses the same thread for all sync executions, preventing thread churn.

## 2. ✅ Enhanced Error Logging with Explicit Paths

**Issue**: Error messages didn't include the exact file paths that failed.

**Fix**: Now logs the complete paths for both concept files:
```
Failed to load concept database from any source! Check that data files exist in one of these locations:
1. Python package: ingest_pdf.data/concept_file_storage.json
2. Namespace package: /path/to/ingest_pdf/data/concept_file_storage.json
3. Relative paths:
   - Main concepts: /full/path/to/concept_file_storage.json
   - Universal seeds: /full/path/to/concept_seed_universal.json
```

## 3. ✅ Added Progress Tracking with Throttling

**Issue**: Progress updates could flood logs with duplicate percentages.

**Fix**: Created a `ProgressTracker` class that only reports when progress changes significantly:

```python
# Example usage in pipeline
progress = ProgressTracker(total=len(chunks), min_change=1.0)  # Only report 1% changes

# In processing loop
for i, chunk in enumerate(chunks):
    # Process chunk...
    
    # Update progress (async context)
    if pct := await progress.update():
        logger.info(f"Processing progress: {pct:.0f}%")
        
# Or in sync context
if pct := progress.update_sync():
    logger.info(f"Processing progress: {pct:.0f}%")
```

Features:
- Thread-safe with async support
- Configurable minimum change threshold
- Returns None if change is below threshold
- Prevents duplicate percentage logging

## 4. ✅ Added Documentation for `run_sync` Behavior

Added clear warning about blocking behavior in FastAPI contexts:
```python
NOTE: When called inside FastAPI request handlers, this will block
that worker thread, potentially serializing requests on that worker.
Consider using async handlers with 'await' instead.
```

## Example: Using Progress Tracking in the Pipeline

Here's how to integrate the progress tracker into PDF processing:

```python
async def process_pdf_with_progress(pdf_path: str):
    # Extract chunks
    chunks = await extract_chunks(pdf_path)
    
    # Create progress tracker
    progress = ProgressTracker(
        total=len(chunks),
        min_change=5.0  # Only log every 5%
    )
    
    # Process with progress updates
    results = []
    for chunk in chunks:
        result = await process_chunk(chunk)
        results.append(result)
        
        # Update progress
        if pct := await progress.update():
            logger.info(f"📊 Processing {pdf_path}: {pct:.0f}% complete")
            
            # Could also send to websocket/SSE
            if websocket:
                await websocket.send_json({"progress": pct})
    
    return results
```

## Future Enhancements Ready to Implement

Based on your suggestions, here are the next improvements ready to go:

### 1. Secrets Model
```python
class SecureSettings(BaseSettings):
    api_key: SecretStr = Field(..., env="TORI_API_KEY")
    db_password: SecretStr = Field(..., env="TORI_DB_PASSWORD")
    
    class Config:
        # Load from HashiCorp Vault or AWS Secrets Manager
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                vault_settings,  # Highest priority
                env_settings,
                file_secret_settings,
                init_settings,
            )
```

### 2. YAML Support
```python
def yaml_config_settings(settings_cls, yaml_file: str):
    import yaml
    with open(yaml_file) as f:
        return settings_cls(**yaml.safe_load(f))

# In Settings.Config
if yaml_path := os.getenv("TORI_CONFIG_FILE"):
    settings = yaml_config_settings(Settings, yaml_path)
```

### 3. Per-Request Overrides
```python
@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile,
    config_overrides: dict = None
):
    # Merge with global settings
    request_settings = settings.copy(update=config_overrides or {})
    
    # Use request-specific settings
    result = await ingest_pdf_clean(
        file.filename,
        settings=request_settings
    )
    return result
```

All fixes have been implemented and tested. The pipeline is now more efficient and operator-friendly!
