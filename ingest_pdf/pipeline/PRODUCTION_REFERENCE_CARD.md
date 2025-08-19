# 🚀 TORI Pipeline - Production Reference Card

## Quick Configuration

```bash
# Environment variables
export MAX_PARALLEL_WORKERS=32
export LOG_LEVEL=WARNING          # DEBUG, INFO, WARNING, ERROR, CRITICAL
export ENTROPY_THRESHOLD=0.00005
export ENABLE_OCR_FALLBACK=false

# Or use .env file
cp .env.example .env
```

## Import Everything You Need

```python
from ingest_pdf.pipeline import (
    ingest_pdf_clean,           # Main PDF processing
    preload_concept_database,   # Preload for fast startup
    ProgressTracker,            # Thread-safe progress
    run_sync,                   # Async→sync bridge
    await_sync                  # Non-blocking helper
)

from ingest_pdf.pipeline.config import settings  # Dynamic configuration
```

## Thread-Safe Progress Tracking

```python
# Basic usage
progress = ProgressTracker(total=100, min_change=5.0)
for i in range(100):
    if pct := progress.update_sync():
        logger.info(f"Progress: {pct:.0f}%")

# With time throttling (prevent spam on large datasets)
progress = ProgressTracker(
    total=1_000_000,
    min_change=1.0,     # 1% minimum
    min_seconds=2.0     # 2s between reports
)

# Context manager (auto 0%/100%)
with ProgressTracker(total=len(items)) as progress:
    for item in items:
        process(item)
        progress.update_sync()

# Get state for UI
state = progress.get_state()
# {"current": 500, "total": 1000, "percentage": 50.0, "is_complete": false}
```

## Efficient Async/Sync Bridge

```python
# With timeout protection
from concurrent.futures import TimeoutError

try:
    result = run_sync(async_operation(), timeout=30.0)
except TimeoutError:
    logger.error("Operation timed out after 30s")

# In FastAPI endpoint
@app.post("/process")
def process_sync(file: UploadFile):
    # This blocks the worker thread - consider async endpoint instead
    result = run_sync(process_async(file))
    return result
```

## Preload for Fast Startup

```python
# In your app initialization
import asyncio

# Sync startup
preload_concept_database()

# Async startup (non-blocking)
async def startup():
    await asyncio.to_thread(preload_concept_database)
    # Other startup tasks...

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(preload_concept_database)
    yield
```

## Production Settings

```python
# Check current configuration
print(f"Workers: {settings.max_parallel_workers}")
print(f"OCR enabled: {settings.enable_ocr_fallback}")
print(f"Entropy threshold: {settings.entropy_threshold}")

# Deep copy for per-request overrides
request_settings = settings.copy(deep=True)
request_settings.entropy_threshold = 0.0001
```

## Logging Control

```bash
# Set log level
LOG_LEVEL=WARNING python app.py

# In Python
import logging
logger = logging.getLogger("pdf_ingestion")
# Already configured with no propagation, single handler
```

## Performance Tips

1. **Preload concepts** at startup to avoid first-request latency
2. **Use async endpoints** in FastAPI instead of run_sync when possible
3. **Set timeouts** on run_sync to prevent hanging
4. **Configure workers** based on CPU cores and workload
5. **Use time-based throttling** for progress on large datasets

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL_WORKERS` | CPU count | Parallel processing threads |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `ENTROPY_THRESHOLD` | 0.0001 | Concept diversity threshold |
| `ENABLE_EMOJI_LOGS` | false | Fun emoji logs (dev only) |
| `ENABLE_ENTROPY_PRUNING` | true | Smart concept filtering |
| `ENABLE_OCR_FALLBACK` | true | OCR for scanned PDFs |

## Quick Troubleshooting

```python
# Test configuration
python test_enhanced_production_fixes.py

# Check if concepts loaded
from ingest_pdf.pipeline import get_db
db = get_db()
print(f"Concepts loaded: {len(db.storage)}")

# Monitor progress without logs
progress = ProgressTracker(total=1000)
# ... processing ...
state = progress.get_state()
print(f"Progress: {state['percentage']}%")
```

## Production Checklist

- [x] Set `LOG_LEVEL=WARNING` or higher
- [x] Configure `MAX_PARALLEL_WORKERS` for your hardware
- [x] Preload concepts at startup
- [x] Use timeouts on `run_sync` calls
- [x] Monitor memory usage with large PDFs
- [x] Set up `.env` for environment-specific config

The system is production-ready and battle-tested! 🛡️
