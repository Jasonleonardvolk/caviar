# Enhanced Error Handling Guide

## Overview

The enhanced error handling system provides:
- Specific exception types for different error categories
- Automatic retry logic for recoverable errors
- Detailed error context and logging
- Consistent error response format
- Error categorization and severity levels

## Exception Hierarchy

```
PipelineError (base)
├── IOError - File I/O errors
├── PDFParseError - PDF parsing errors
├── ExtractionError - Concept extraction errors
├── MemoryError - Memory-related errors
├── TimeoutError - Processing timeouts
├── ValidationError - Input validation errors
└── ConfigurationError - Configuration errors
```

## Error Categories & Recovery

### IOError
- **When**: File not found, permission denied, disk errors
- **Severity**: HIGH
- **Recoverable**: Sometimes (e.g., temporary disk issues)
- **Strategy**: Retry with exponential backoff

### PDFParseError
- **When**: Corrupted PDF, invalid format
- **Severity**: MEDIUM
- **Recoverable**: No
- **Strategy**: Fallback to OCR or abort

### ExtractionError
- **When**: Chunk processing failures
- **Severity**: MEDIUM
- **Recoverable**: Sometimes
- **Strategy**: Skip failed chunks, continue

### MemoryError
- **When**: Out of memory
- **Severity**: HIGH/CRITICAL
- **Recoverable**: No
- **Strategy**: Degrade functionality or abort

### ValidationError
- **When**: Invalid input parameters
- **Severity**: MEDIUM
- **Recoverable**: No
- **Strategy**: Abort with clear message

## Usage Examples

### Basic Error Handling
```python
from pipeline.error_handling import ErrorContext, error_handler

def process_pdf(pdf_path):
    context = ErrorContext(operation="pdf_processing", file_path=pdf_path)
    
    try:
        # Processing logic
        result = extract_chunks(pdf_path)
    except Exception as e:
        strategy = error_handler.handle_error(e, context)
        
        if strategy == RecoveryStrategy.RETRY:
            # Retry the operation
            pass
        elif strategy == RecoveryStrategy.ABORT:
            # Abort processing
            raise
```

### Using Retry Decorator
```python
from pipeline.error_handling import with_retry, PipelineIOError

@with_retry(max_retries=3, delay=1.0, recoverable_errors=[PipelineIOError])
def read_file(path):
    with open(path, 'rb') as f:
        return f.read()
```

### Specific Exception Handling
```python
try:
    metadata = extract_pdf_metadata(pdf_path)
except PipelineIOError as e:
    if e.recoverable:
        # Try alternative approach
        metadata = get_basic_metadata(pdf_path)
    else:
        # Fatal error, cannot continue
        return error_response(e)
except PDFParseError as e:
    # Try OCR fallback
    return process_with_ocr(pdf_path)
```

## Error Response Format

All errors maintain consistent response schema:

```json
{
  "filename": "document.pdf",
  "concept_count": 0,
  "concepts": [],
  "status": "error",
  "error": {
    "message": "PDF file not found: document.pdf",
    "category": "io_error",
    "severity": "high",
    "recoverable": false,
    "operation": "safety_check",
    "retry_count": 0,
    "details": {
      "path": "document.pdf"
    }
  },
  "processing_time_seconds": 0.15
}
```

## Configuration

### Environment Variables
```bash
# Enable error logging to files
export LOG_ERRORS_TO_FILE=true
export ERROR_LOG_DIR=./error_logs

# Enable retries
export ENABLE_ERROR_RETRIES=true
export MAX_RETRY_ATTEMPTS=3
```

### Error Handler Configuration
```python
from pipeline.error_handling import ErrorHandler

# Custom error handler
handler = ErrorHandler(
    enable_retries=True,
    log_errors_to_file=True,
    error_log_dir="./my_error_logs"
)
```

## Error Logging

Errors are logged with appropriate severity:
- **LOW**: Warnings (recoverable issues)
- **MEDIUM**: Errors (functionality impaired)
- **HIGH**: Errors (major issues)
- **CRITICAL**: Critical errors (processing cannot continue)

Example log output:
```
2024-01-15 10:23:45 | ERROR | io_error in chunk_extraction: PDF file not found: test.pdf [Severity: high, Recoverable: False] Details: {"path": "test.pdf"}
```

## Partial Success Handling

When some operations fail but processing continues:

```python
# Build partial success response
response = ErrorResponseBuilder.build_partial_success_response(
    file_path=pdf_path,
    results=partial_results,
    errors=non_critical_errors,
    start_time=start_time,
    admin_mode=admin_mode
)
```

Result includes warnings:
```json
{
  "status": "partial_success",
  "concept_count": 150,
  "warnings": [
    {
      "message": "Failed to extract text from page 5",
      "category": "parse_error",
      "severity": "low"
    }
  ]
}
```

## Best Practices

1. **Catch specific exceptions** where possible
2. **Use error context** to track operation state
3. **Log with appropriate severity** based on impact
4. **Implement retry logic** for transient failures
5. **Provide actionable error messages** to users
6. **Save error details** for debugging (in admin mode)

## Error Analysis

Analyze saved error logs:
```python
import json
from pathlib import Path

error_dir = Path("./error_logs")
errors = []

for file in error_dir.glob("error_*.json"):
    with open(file) as f:
        errors.append(json.load(f))

# Analyze by category
from collections import Counter
categories = Counter(e["category"] for e in errors)
print(f"Error distribution: {dict(categories)}")

# Find patterns
io_errors = [e for e in errors if e["category"] == "io_error"]
```

## Migration from Broad Exceptions

### Before:
```python
try:
    # Complex operation
    process_everything()
except Exception as e:
    logger.error(f"Failed: {e}")
    return {"error": str(e)}
```

### After:
```python
context = ErrorContext(operation="processing")

try:
    # File operations
    validate_input(pdf_path)
except ValidationError as e:
    return ErrorResponseBuilder.build_error_response(
        pdf_path, e, context, start_time, admin_mode
    )

try:
    # PDF operations
    extract_metadata(pdf_path)
except PipelineIOError as e:
    if e.recoverable and context.should_retry(e):
        # Retry logic
        time.sleep(1)
        extract_metadata(pdf_path)
    else:
        raise
except PDFParseError as e:
    # Try fallback
    logger.warning(f"PDF parse failed, trying OCR: {e}")
    use_ocr_fallback = True
```

## Error Recovery Patterns

### 1. Retry Pattern
```python
@with_retry(max_retries=3, delay=1.0, backoff=2.0)
def unstable_operation():
    # May fail intermittently
    result = external_api_call()
    return result
```

### 2. Fallback Pattern
```python
def extract_with_fallback(pdf_path):
    try:
        # Primary method
        return extract_with_pypdf2(pdf_path)
    except PDFParseError:
        try:
            # Fallback to alternative library
            return extract_with_pdfplumber(pdf_path)
        except Exception:
            # Final fallback to OCR
            return extract_with_ocr(pdf_path)
```

### 3. Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise ServiceUnavailable("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                
            raise
```

### 4. Graceful Degradation
```python
def process_with_degradation(pdf_path, context):
    full_features = True
    
    try:
        # Try full processing
        concepts = extract_all_concepts(pdf_path)
    except MemoryError as e:
        # Degrade to basic processing
        logger.warning("Memory limit reached, degrading to basic mode")
        full_features = False
        concepts = extract_basic_concepts(pdf_path, limit=100)
    except TimeoutError as e:
        # Skip non-essential processing
        logger.warning("Timeout reached, skipping enhancement")
        concepts = extract_concepts_no_enhancement(pdf_path)
        
    return concepts, full_features
```

## Testing Error Handling

### Unit Tests
```python
import pytest
from pipeline.error_handling import PipelineIOError, ValidationError

def test_file_not_found_handling():
    with pytest.raises(PipelineIOError) as exc_info:
        extract_pdf_metadata("nonexistent.pdf")
    
    assert exc_info.value.category == ErrorCategory.IO_ERROR
    assert not exc_info.value.recoverable

def test_retry_decorator():
    attempt_count = 0
    
    @with_retry(max_retries=2, delay=0.1)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise PipelineIOError("Temporary failure", recoverable=True)
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert attempt_count == 3
```

### Integration Tests
```python
def test_error_response_format():
    # Test that error responses maintain schema
    response = ingest_pdf_clean("invalid.pdf")
    
    assert response["status"] == "error"
    assert "error" in response
    assert "message" in response["error"]
    assert "category" in response["error"]
    assert response["concept_count"] == 0
    assert isinstance(response["concepts"], list)
```

## Monitoring & Alerting

### Error Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
error_counter = Counter(
    'pipeline_errors_total',
    'Total number of pipeline errors',
    ['category', 'severity', 'operation']
)

error_recovery_histogram = Histogram(
    'pipeline_error_recovery_seconds',
    'Time spent recovering from errors',
    ['strategy']
)

# Track errors
def track_error(error: PipelineError, context: ErrorContext):
    error_counter.labels(
        category=error.category.value,
        severity=error.severity.value,
        operation=context.operation
    ).inc()
```

### Alerting Rules
```yaml
# Prometheus alerting rules
groups:
  - name: pipeline_errors
    rules:
      - alert: HighErrorRate
        expr: rate(pipeline_errors_total[5m]) > 0.1
        annotations:
          summary: "High error rate in PDF pipeline"
          
      - alert: CriticalErrors
        expr: pipeline_errors_total{severity="critical"} > 0
        annotations:
          summary: "Critical errors detected in pipeline"
```

## Summary

The enhanced error handling system provides:
1. **Specific exceptions** for different error types
2. **Automatic retry** for recoverable errors
3. **Detailed context** for debugging
4. **Consistent responses** for API consumers
5. **Flexible recovery** strategies
6. **Comprehensive logging** and monitoring

This approach makes the pipeline more robust, easier to debug, and provides better user experience through clear, actionable error messages.
