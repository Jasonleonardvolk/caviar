# TORI Pipeline Dynamic Configuration

## Overview

The TORI pipeline now uses Pydantic BaseSettings for dynamic configuration. This allows you to override any configuration value through environment variables or a `.env` file without modifying code.

## Key Benefits

1. **Zero Code Changes for Config Updates**: Change thresholds, limits, and feature flags without redeploying
2. **Environment-Specific Settings**: Different settings for dev, staging, and production
3. **Type Safety**: Pydantic validates and converts all configuration values
4. **Backward Compatibility**: Existing code continues to work without modifications

## Usage

### 1. Using Environment Variables

Set any configuration value through environment variables:

```bash
# Disable entropy pruning for this run
export ENABLE_ENTROPY_PRUNING=false

# Increase parallel workers
export MAX_PARALLEL_WORKERS=32

# Run your pipeline
python process_pdf.py document.pdf
```

### 2. Using .env File

Create a `.env` file in the pipeline directory:

```bash
cd ingest_pdf/pipeline
cp .env.example .env
# Edit .env with your preferred settings
```

### 3. In Docker/Kubernetes

```yaml
# docker-compose.yml
services:
  tori:
    environment:
      - MAX_PARALLEL_WORKERS=16
      - ENTROPY_THRESHOLD=0.0001
      - ENABLE_OCR_FALLBACK=true
```

## Configuration Options

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CONTEXT_EXTRACTION` | `true` | Extract context from documents |
| `ENABLE_FREQUENCY_TRACKING` | `true` | Track concept frequency |
| `ENABLE_SMART_FILTERING` | `true` | Apply smart filtering algorithms |
| `ENABLE_ENTROPY_PRUNING` | `true` | Prune concepts based on entropy |
| `ENABLE_OCR_FALLBACK` | `true` | Use OCR for scanned PDFs |
| `ENABLE_PARALLEL_PROCESSING` | `true` | Process chunks in parallel |
| `ENABLE_ENHANCED_MEMORY_STORAGE` | `true` | Use enhanced memory storage |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL_WORKERS` | `None` | Max parallel workers (None = CPU count) |
| `OCR_MAX_PAGES` | `None` | Max pages to OCR (None = unlimited) |

### Entropy Pruning

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTROPY_THRESHOLD` | `0.0001` | Lower = more aggressive pruning |
| `SIMILARITY_THRESHOLD` | `0.85` | Higher = keep more similar concepts |
| `MAX_DIVERSE_CONCEPTS` | `None` | Max concepts to keep (None = unlimited) |
| `CONCEPTS_PER_CATEGORY` | `None` | Max per category (None = unlimited) |

### File Size Limits

Configure processing limits based on file size:

```bash
# Small files (< 2MB)
SMALL_FILE_MB=2
SMALL_CHUNKS=400
SMALL_CONCEPTS=300

# Medium files (2-10MB)
MEDIUM_FILE_MB=10
MEDIUM_CHUNKS=800
MEDIUM_CONCEPTS=1000

# Large files (10-50MB)
LARGE_FILE_MB=50
LARGE_CHUNKS=2000
LARGE_CONCEPTS=2500

# Extra large files (>50MB)
XLARGE_CHUNKS=3000
XLARGE_CONCEPTS=5000
```

### Section Weights

Adjust importance of different document sections:

```bash
# As JSON
SECTION_WEIGHTS_JSON='{"title":2.5,"abstract":2.0,"introduction":1.5}'

# As comma-separated pairs
SECTION_WEIGHTS='title=2.5,abstract=2.0,introduction=1.5'
```

## Advanced Usage

### Programmatic Access

```python
from ingest_pdf.pipeline.config import settings

# Access any setting
print(f"Max workers: {settings.max_parallel_workers}")
print(f"Entropy threshold: {settings.entropy_threshold}")

# Modify at runtime (not recommended in production)
settings.entropy_threshold = 0.0002
```

### Custom Validation

```python
from ingest_pdf.pipeline.config import Settings

# Create custom settings with validation
custom_settings = Settings(
    max_parallel_workers=64,
    entropy_threshold=0.001
)
```

### Per-Request Configuration

```python
from ingest_pdf.pipeline import ingest_pdf_clean

# Override settings for a specific request
result = ingest_pdf_clean(
    "document.pdf",
    extraction_params={
        "threshold": 0.001,
        "max_concepts": 1000
    }
)
```

## Migration from Static Config

The system maintains full backward compatibility. Code that imports constants continues to work:

```python
# Old style - still works
from ingest_pdf.pipeline.config import ENABLE_ENTROPY_PRUNING
if ENABLE_ENTROPY_PRUNING:
    apply_pruning()

# New style - recommended
from ingest_pdf.pipeline.config import settings
if settings.enable_entropy_pruning:
    apply_pruning()
```

## Production Best Practices

1. **Use .env files for local development** - Never commit them to version control
2. **Use environment variables in production** - Set through your deployment system
3. **Document your settings** - Keep a record of what values work best
4. **Monitor performance** - Track how setting changes affect processing
5. **Start conservative** - Begin with default values and tune gradually

## Troubleshooting

### Settings not taking effect?

1. Check environment variable names (case-insensitive)
2. Ensure .env file is in the correct directory
3. Verify no typos in variable names
4. Check that values are valid (Pydantic will raise errors for invalid types)

### View current configuration:

```python
from ingest_pdf.pipeline.config import settings, CONFIG
import json

# Pretty print all settings
print(json.dumps(CONFIG, indent=2))

# Or check specific settings
print(f"Entropy pruning enabled: {settings.enable_entropy_pruning}")
print(f"Current threshold: {settings.entropy_threshold}")
```

## Environment Variable Reference

See `.env.example` for a complete list of all available environment variables with descriptions and example values.
