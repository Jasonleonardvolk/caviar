# Concurrency Improvements Summary

## Overview

The pipeline has been enhanced with a dedicated ConcurrencyManager that provides:
- Separate executors for CPU-bound, I/O-bound, and chunk processing tasks
- Unified async/sync abstraction layer
- Auto-throttling based on system resources
- Comprehensive metrics and monitoring
- Graceful shutdown handling

## Key Components

### 1. ConcurrencyManager (`concurrency_manager.py`)

**Features:**
- **Three Dedicated Executors:**
  - `cpu_executor`: ProcessPoolExecutor for CPU-intensive tasks (concept analysis, pruning)
  - `io_executor`: ThreadPoolExecutor for I/O operations (file reading, API calls)
  - `chunk_executor`: ThreadPoolExecutor with higher concurrency for chunk processing

- **Auto-Throttling:**
  - Monitors CPU and memory usage
  - Dynamically adjusts concurrency when system is under load
  - Configurable thresholds (default: 85% CPU, 80% memory)

- **Unified Interface:**
  - `run_cpu()`: Execute CPU-bound functions
  - `run_io()`: Execute I/O-bound functions
  - `run_chunk()`: Execute chunk processing functions
  - `map_*()`: Batch processing methods

- **Metrics Collection:**
  - Tasks submitted/completed/failed
  - Average execution time
  - Max concurrent tasks
  - Success rate

### 2. Enhanced Pipeline Processing

**Synchronous Processing (`process_chunks_sync`):**
- Uses ConcurrencyManager for parallel chunk processing
- Batches chunks for efficient processing
- Progress tracking with optional tqdm support
- Early exit when concept limit reached

**Asynchronous Processing (`process_chunks_async`):**
- Fully async implementation using ConcurrencyManager
- Adaptive strategy based on workload size
- Efficient batching for large datasets
- Periodic yielding to prevent event loop blocking

### 3. Error Handling Integration

- Specific exception types (IOError, PDFParseError, etc.)
- Retry logic with exponential backoff
- Detailed error context and recovery strategies
- Consistent error response format

## Configuration

### Environment Variables

```bash
# Chunk processing
CHUNK_BATCH_SIZE=10              # Chunks per batch
MAX_PARALLEL_WORKERS=16          # Max concurrent workers
USE_PROCESS_POOL=true           # Use processes for large workloads

# Auto-throttling
CPU_THRESHOLD=85                 # CPU usage threshold (%)
MEMORY_THRESHOLD=80              # Memory usage threshold (%)
ENABLE_AUTO_THROTTLE=true        # Enable dynamic throttling

# Metrics
ENABLE_METRICS=true              # Collect executor statistics
```

### ConcurrencyConfig

```python
config = ConcurrencyConfig(
    cpu_workers=os.cpu_count() - 1,    # Leave one core free
    io_workers=20,                      # I/O thread pool size
    chunk_processor_workers=16,         # Chunk processing threads
    enable_auto_throttle=True,
    cpu_threshold=85.0,
    memory_threshold=80.0
)
```

## Usage Examples

### Basic Usage

```python
from pipeline import get_concurrency_manager

# Get the global manager
manager = get_concurrency_manager()

# Run CPU-intensive task
result = await manager.run_cpu(analyze_concepts, concepts_list)

# Run I/O operation
pdf_data = await manager.run_io(read_pdf_file, pdf_path)

# Process chunks in parallel
results = await manager.map_chunks(extract_concepts, chunks)
```

### With Progress Tracking

```python
async with ProgressTracker(len(chunks), description="Processing chunks") as tracker:
    for batch in batches:
        results = await manager.map_chunks(process_batch, batch)
        await tracker.update_async(len(batch))
```

### Circuit Breaker Pattern

```python
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

try:
    result = await breaker.call(manager.run_io, unstable_api_call)
except Exception as e:
    logger.error(f"Circuit breaker open: {e}")
```

## Performance Benefits

1. **Resource Isolation**: CPU and I/O tasks don't interfere with each other
2. **Optimal Concurrency**: Different limits for different task types
3. **System Protection**: Auto-throttling prevents overload
4. **Better Throughput**: Batching reduces overhead
5. **Monitoring**: Built-in metrics for performance analysis

## Migration Guide

### Old Code:
```python
# Direct async/sync mixing
result = await asyncio.to_thread(cpu_intensive_func, data)

# Manual executor management
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(func, item) for item in items]
```

### New Code:
```python
# Using ConcurrencyManager
manager = get_concurrency_manager()
result = await manager.run_cpu(cpu_intensive_func, data)

# Batch processing
results = await manager.map_cpu(func, items)
```

## Monitoring & Debugging

### Get Executor Statistics
```python
stats = manager.get_stats()
for executor_name, executor_stats in stats.items():
    print(f"{executor_name}: {executor_stats}")

# Output:
# cpu: {'tasks_submitted': 150, 'tasks_completed': 148, 'success_rate': 98.7, ...}
# io: {'tasks_submitted': 523, 'tasks_completed': 520, 'success_rate': 99.4, ...}
# chunk: {'tasks_submitted': 1250, 'tasks_completed': 1250, 'success_rate': 100.0, ...}
```

### Debug Logging
```python
import logging
logging.getLogger("tori.ingest_pdf.concurrency").setLevel(logging.DEBUG)
```

### Performance Profiling
```python
# Enable profiling in chunk processing
config = ProcessingConfig(enable_profiling=True)

# View detailed timing information
if logger.isEnabledFor(logging.DEBUG):
    stats = processor.get_stats()
    logger.debug(f"Processing stats: {stats}")
```

## Best Practices

### 1. Choose the Right Executor
```python
# CPU-intensive (concept analysis, ML inference)
await manager.run_cpu(analyze_concepts, data)

# I/O-bound (file operations, API calls)
await manager.run_io(read_file, path)

# Mixed workload (chunk processing)
await manager.run_chunk(process_chunk, chunk)
```

### 2. Batch Operations
```python
# Instead of individual calls
for item in items:
    await manager.run_cpu(process, item)

# Use batch processing
results = await manager.map_cpu(process, items)
```

### 3. Handle Failures Gracefully
```python
results = await asyncio.gather(
    *[manager.run_chunk(process, chunk) for chunk in chunks],
    return_exceptions=True
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Chunk {i} failed: {result}")
```

### 4. Monitor System Resources
```python
# Check throttling status
if manager._throttle_factor < 1.0:
    logger.warning(f"System under load, throttled to {manager._throttle_factor:.1%}")
```

### 5. Graceful Shutdown
```python
# Automatic via atexit
# Or manual:
manager.shutdown(wait=True)

# Or with context manager:
async with ConcurrencyManager() as manager:
    # Use manager
    pass  # Automatically shut down
```

## Advanced Patterns

### 1. Priority Processing
```python
# Process important chunks first
important_chunks = [c for c in chunks if c.get('priority') == 'high']
regular_chunks = [c for c in chunks if c.get('priority') != 'high']

# Process high priority first
high_priority_results = await manager.map_chunks(process, important_chunks)
regular_results = await manager.map_chunks(process, regular_chunks)
```

### 2. Adaptive Batch Sizing
```python
# Adjust batch size based on system load
batch_size = 10 if manager._throttle_factor >= 0.8 else 5

for i in range(0, len(items), batch_size):
    batch = items[i:i + batch_size]
    await process_batch(batch)
```

### 3. Cascading Executors
```python
# I/O followed by CPU processing
async def process_file(path):
    # Read file using I/O executor
    data = await manager.run_io(read_file, path)
    
    # Process data using CPU executor
    result = await manager.run_cpu(analyze_data, data)
    
    return result
```

### 4. Timeout Handling
```python
try:
    result = await asyncio.wait_for(
        manager.run_cpu(long_running_task, data),
        timeout=30.0
    )
except asyncio.TimeoutError:
    logger.error("Task timed out after 30 seconds")
```

## Troubleshooting

### High Memory Usage
- Reduce `io_workers` and `chunk_processor_workers`
- Enable auto-throttling
- Use smaller batch sizes

### Slow Processing
- Increase worker counts if CPU/memory allows
- Check throttling status
- Profile to identify bottlenecks

### Deadlocks
- Avoid nested executor calls
- Don't wait for async operations in sync contexts
- Use proper async/await patterns

### Process Pool Issues
- Ensure functions are picklable
- Initialize resources in process initializer
- Handle signal interrupts properly

## Summary

The enhanced concurrency system provides:
- **Better Performance**: Optimized for different workload types
- **Resource Protection**: Prevents system overload
- **Easy Migration**: Simple API with powerful features
- **Production Ready**: Monitoring, metrics, and graceful shutdown
- **Flexible Configuration**: Adaptable to different environments

This implementation follows the principle of "Refine Concurrency: Move to dedicated executors; unify async/sync" by providing:
1. Clear separation of concerns with dedicated executors
2. Unified interface for both async and sync code
3. Automatic resource management and monitoring
4. Production-grade error handling and recovery

The system is designed to scale from small single-file processing to large batch operations while maintaining stability and performance.
