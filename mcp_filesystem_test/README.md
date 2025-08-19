# MCP Filesystem Test Suite

This directory demonstrates the full capabilities of the MCP Filesystem Server with production-ready, scalable code.

## Features Demonstrated

### 1. Directory Operations
- Creating directories
- Listing directory contents
- Tree view of directory structure

### 2. File Operations
- Creating new files
- Reading file contents
- Editing existing files
- Moving/renaming files
- Searching for files

### 3. Scalable Processing
- Asynchronous data processing with asyncio
- Parallel processing with multiprocessing
- Configurable worker pools
- Batch processing for large datasets
- Resource utilization optimization

## Project Structure

```
mcp_filesystem_test/
├── scalable_data_processor.py  # Main processing module
├── config.json                 # Configuration file
├── README.md                   # This file
└── results/                    # Output directory (created at runtime)
    ├── async_results.json
    └── parallel_results.json
```

## Usage

### Basic Example

```python
from scalable_data_processor import ScalableDataProcessor

# Create processor instance
with ScalableDataProcessor(max_workers=8) as processor:
    data = list(range(10000))
    results = processor.process_data_parallel(data)
    processor.save_results(results, 'output.json')
```

### Async Processing

```python
import asyncio
from scalable_data_processor import ScalableDataProcessor

async def process_async():
    processor = ScalableDataProcessor(use_processes=False)
    data = ['item_' + str(i) for i in range(1000)]
    results = await processor.process_data_async(data)
    return results

results = asyncio.run(process_async())
```

## Configuration

The `config.json` file contains all configurable parameters:

- **processing**: Batch size, worker count, timeout settings
- **storage**: Directory paths for results, logs, and temporary files
- **features**: Toggle various processing capabilities
- **performance**: Memory limits, CPU affinity, buffer sizes

## Performance Optimization

1. **Parallel Processing**: Utilizes all available CPU cores
2. **Batch Processing**: Processes data in configurable chunks
3. **Memory Management**: Configurable memory limits and garbage collection
4. **Async I/O**: Non-blocking operations for I/O-bound tasks

## Requirements

- Python 3.8+
- No external dependencies (uses standard library only)

## Testing

Run the demonstration:

```bash
python scalable_data_processor.py
```

This will:
1. Process 100 items asynchronously
2. Process 1000 items in parallel
3. Save results to JSON files
4. Log all operations

## Future Enhancements

- [ ] Add streaming data support
- [ ] Implement distributed processing
- [ ] Add data validation pipeline
- [ ] Create monitoring dashboard
- [ ] Add support for multiple data formats

## License

This is a demonstration project for testing MCP Filesystem Server capabilities.
