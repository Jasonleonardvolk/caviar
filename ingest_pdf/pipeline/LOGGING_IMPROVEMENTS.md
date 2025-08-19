# Logging Configuration Improvements

## Problems with Original Approach

The original `pipeline.py` had several logging issues:

1. **Module-level configuration**: Each module was configuring its own logger
2. **Generic logger name**: Used "pdf_ingestion" which could conflict with other packages
3. **Risk of handler duplication**: The `hasHandlers()` check could fail if another module configured first
4. **No hierarchy**: Flat logger structure without proper namespacing

## Improved Solution

### 1. Centralized Configuration (`logging_config.py`)

```python
# Single configuration point for entire application
from pipeline.logging_config import ToriLoggerConfig

# Configure once at startup
ToriLoggerConfig.configure(
    root_level="INFO",
    enable_emoji=True,
    module_levels={
        "tori.ingest_pdf.pipeline": "DEBUG",
        "tori.ingest_pdf.io": "INFO"
    }
)
```

### 2. Hierarchical Logger Names

```python
# Old approach
logger = logging.getLogger("pdf_ingestion")

# New approach
from .logging_config import get_logger
logger = get_logger(__name__)  # Creates "tori.ingest_pdf.pipeline" logger
```

### 3. Application Bootstrap Pattern

```python
# app_bootstrap.py - Initialize logging before importing modules
def main():
    # Step 1: Configure logging
    initialize_logging()
    
    # Step 2: Import modules that use logging
    from pipeline.pipeline_improved import ingest_pdf_clean
    
    # Step 3: Run application
    result = ingest_pdf_clean("document.pdf")
```

### 4. Environment-Based Configuration

```bash
# Production settings
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export ENABLE_EMOJI_LOGS=false
export LOG_FILE=/var/log/tori_ingest.log
export LOG_MODULE_LEVELS="tori.ingest_pdf.storage:DEBUG,tori.ingest_pdf.io:WARNING"

# Development settings
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export ENABLE_EMOJI_LOGS=true
```

### 5. Configuration File Support

```json
// log_config.json
{
  "root_level": "INFO",
  "enable_emoji": false,
  "module_levels": {
    "tori.ingest_pdf.pipeline": "DEBUG",
    "tori.ingest_pdf.storage": "INFO",
    "tori.ingest_pdf.quality": "WARNING"
  },
  "format": "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
  "date_format": "%Y-%m-%d %H:%M:%S"
}
```

```bash
export LOG_CONFIG_FILE=/path/to/log_config.json
```

## Migration Guide

### For Existing Code

1. **Remove module-level logging configuration**:
```python
# Remove this from each module:
logger = logging.getLogger("pdf_ingestion")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    # ... handler configuration
```

2. **Replace with**:
```python
from .logging_config import get_logger
logger = get_logger(__name__)
```

3. **Add bootstrap initialization**:
```python
# In your main entry point
from pipeline.logging_config import configure_preset

# Initialize before importing other modules
configure_preset("production")  # or "development", "testing"
```

### Benefits

1. **No duplicate handlers**: Configuration happens once at startup
2. **Proper hierarchy**: Loggers follow package structure
3. **Flexible configuration**: Easy to adjust per-module levels
4. **Environment-aware**: Different settings for dev/prod/test
5. **Emoji support**: Centralized emoji formatting
6. **File logging**: Optional file output with rotation support

### Example Usage

```python
# Development with debug logging
python -c "
from pipeline.logging_config import configure_preset
configure_preset('development')
from pipeline.pipeline_improved import ingest_pdf_clean
result = ingest_pdf_clean('test.pdf')
"

# Production with file logging
ENVIRONMENT=production LOG_FILE=app.log python app_bootstrap.py document.pdf

# Custom module levels
LOG_MODULE_LEVELS="tori.ingest_pdf.pipeline:DEBUG,tori.ingest_pdf.storage:ERROR" python app.py
```

## Testing

The logging configuration includes a testing preset that minimizes output:

```python
# In test files
from pipeline.logging_config import configure_preset

def setup_module():
    configure_preset("testing")  # WARNING level, no emoji

def test_pdf_processing():
    # Only warnings and errors will be logged
    result = ingest_pdf_clean("test.pdf")
    assert result["status"] == "success"
```
