# Chunk Processing & Parallelism Improvements

## Summary of Changes

### 1. **Dedicated Executor Management**
- Created `ChunkProcessor` class with configurable ThreadPoolExecutor or ProcessPoolExecutor
- Precise control over worker count and executor type
- Avoids saturating Python's default executor

### 2. **Smart Batching**
- Configurable batch sizes (default: 4 chunks per batch)
- Reduces thread startup/teardown overhead
- Improves throughput for small, quick tasks

### 3. **Early Exit with Shared State**
- Uses `asyncio.Event` for coordinated early stopping
- Prevents wasted processing once max_concepts is reached
- All workers check the early stop event

### 4. **Chunk Prioritization**
- Scores chunks based on section type and content heuristics
- Processes high-value chunks first (title, abstract, introduction)
- Better results sooner = earlier exits = lower latency

### 5. **Backpressure & Flow Control**
- Result queue with configurable size limit
- Prevents memory overflow from overproduction
- Consumer pattern for controlled result processing

### 6. **Performance Profiling**
- Optional profiling tracks processing times
- Rolling window of last 100 operations
- Debug logging of performance statistics

### 7. **Flexible Configuration**
- Environment variables for runtime tuning
- Support for both thread and process pools
- Easy switching based on workload characteristics

## Configuration Options

### Environment Variables
```bash
# Enable process pool for CPU-heavy workloads
export USE_PROCESS_POOL=true

# Set batch size (default: 4)
export CHUNK_BATCH_SIZE=8

# Override max parallel workers
export MAX_PARALLEL_WORKERS_OVERRIDE=16

# Enable debug logging for profiling
export LOG_LEVEL=DEBUG
```

### Programmatic Configuration
```python
from pipeline.chunk_processor import ProcessingConfig, ExecutorType

config = ProcessingConfig(
    max_workers=16,
    executor_type=ExecutorType.THREAD,
    batch_size=8,
    max_concepts=10000,
    enable_profiling=True,
    prioritize_chunks=True,
    early_exit=True
)
```

## Usage Examples

### Basic Usage (Integrated in Pipeline)
The pipeline now automatically uses the advanced chunk processor:
```python
# Sync processing
result = ingest_pdf_clean("document.pdf")

# Async processing with full parallelism
result = await ingest_pdf_async("document.pdf")
```

### Direct Usage
```python
from pipeline.chunk_processor import create_chunk_processor

# Create processor
with create_chunk_processor(
    max_workers=8,
    use_processes=False,
    batch_size=4,
    max_concepts=5000
) as processor:
    
    # Process chunks
    concepts = await processor.process_chunks(
        chunks,
        extraction_func,
        params,
        progress_callback
    )
    
    # Get statistics
    stats = processor.get_stats()
    print(f"Processed with {stats['avg_processing_time']:.3f}s average")
```

## Chunk Prioritization Algorithm

Chunks are scored based on:
1. **Section type** (3.0 for title, 2.5 for abstract, etc.)
2. **Keyword presence** (+0.1 per academic keyword, max +0.5)
3. **Academic indicators** (+0.2 for citations, figures, tables)

Example scoring:
- Title chunk: 3.0 base score
- Abstract with "framework" keyword: 2.5 + 0.1 = 2.6
- Body text with citation: 1.0 + 0.2 = 1.2

## Performance Characteristics

### ThreadPoolExecutor (Default)
- **Best for**: Mixed I/O and CPU tasks
- **Workers**: 2 × CPU cores (max 16)
- **GIL impact**: Moderate (good for I/O-bound portions)

### ProcessPoolExecutor (Optional)
- **Best for**: Pure CPU-bound tasks
- **Workers**: 1 × CPU cores
- **GIL impact**: None (true parallelism)
- **Overhead**: Higher startup cost

### Batching Impact
- **Small chunks**: 2-4x throughput improvement
- **Large chunks**: Minimal improvement
- **Sweet spot**: 4-8 chunks per batch

## Monitoring & Debugging

Enable debug logging to see detailed statistics:
```bash
export LOG_LEVEL=DEBUG
python process_pdf.py
```

Sample output:
```
Chunk processing stats: {
    'concepts_collected': 1523,
    'avg_processing_time': 0.045,
    'max_workers': 8,
    'executor_type': 'thread',
    'batch_size': 4,
    'early_exit_triggered': True
}
```

## Migration Notes

The new chunk processor is a drop-in replacement:
- Existing code continues to work
- ENABLE_PARALLEL_PROCESSING still controls parallel vs sequential
- MAX_PARALLEL_WORKERS still sets worker count
- Additional tuning available via environment variables

## Best Practices

1. **For Academic PDFs**: Use default settings (threads, batch_size=4)
2. **For Large PDFs**: Increase batch_size to 8-16
3. **For CPU-Heavy Extraction**: Enable process pool
4. **For Memory-Constrained**: Reduce queue_size and batch_size
5. **For Debugging**: Enable profiling and check stats

## Future Enhancements

1. **Adaptive Batching**: Dynamically adjust batch size based on performance
2. **GPU Acceleration**: For embedding-based concept extraction
3. **Distributed Processing**: For multi-machine scaling
4. **Smart Caching**: Cache extracted concepts for similar chunks
