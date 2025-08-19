# ProgressTracker Improvements

## Summary of Changes

### 1. **Unified Update Method**
- Removed duplicate `update()` and `update_sync()` methods
- Single `update()` method works in both sync and async contexts
- Uses `threading.RLock` which is compatible with both environments
- `update_sync` is now just an alias for backward compatibility

### 2. **Optional tqdm Integration**
- Automatically detects if tqdm is available
- Uses tqdm for better CLI progress display when in a TTY
- Falls back to logging when tqdm is not available or not in TTY
- Can be controlled with `use_tqdm` parameter

### 3. **Simplified Context Managers**
- Unified implementation for both sync and async context managers
- Async versions simply delegate to sync versions
- Proper cleanup with `close()` method for tqdm

### 4. **Enhanced Features**
- Added `description` parameter for better progress messages
- `set_description()` method to update progress description dynamically
- Improved `get_state()` to include description
- Better cleanup handling with `close()` method

## Usage Examples

### Basic Usage (Sync)
```python
# Simple progress tracking
with ProgressTracker(total=100, description="Processing files") as tracker:
    for i in range(100):
        # Do work...
        tracker.update(1)

# Without context manager
tracker = ProgressTracker(total=50, min_change=5.0)  # Report every 5%
for i in range(50):
    if pct := tracker.update(1):
        print(f"Reached {pct}%")
tracker.close()
```

### Async Usage
```python
# Async context manager works the same way
async with ProgressTracker(total=100, description="Async processing") as tracker:
    for i in range(100):
        # Do async work...
        await some_async_operation()
        await tracker.update_async(1)  # or just tracker.update(1)
```

### Controlling tqdm
```python
# Force tqdm on (if available)
tracker = ProgressTracker(total=1000, use_tqdm=True)

# Force tqdm off (always use logging)
tracker = ProgressTracker(total=1000, use_tqdm=False)

# Auto-detect (default - uses tqdm if available and in TTY)
tracker = ProgressTracker(total=1000)
```

### Dynamic Descriptions
```python
tracker = ProgressTracker(total=len(files), description="Processing")
for i, file in enumerate(files):
    tracker.set_description(f"Processing {file.name}")
    process_file(file)
    tracker.update(1)
tracker.close()
```

## Benefits

1. **Cleaner Code**: No more duplicate async/sync implementations
2. **Better UX**: tqdm provides a nice progress bar in terminals
3. **Flexible**: Works in notebooks, terminals, and non-TTY environments
4. **Backward Compatible**: All existing code continues to work
5. **Thread-Safe**: Single locking mechanism works everywhere

## Migration Guide

### For Existing Code
```python
# Old code - still works!
tracker.update_sync(1)  # Works - update_sync is an alias

# New preferred way
tracker.update(1)  # Works in both sync and async contexts
```

### For Async Code
```python
# Old code
async with tracker:
    await tracker.update(1)

# Still works exactly the same!
# But now update() doesn't need to be awaited
async with tracker:
    tracker.update(1)  # No await needed
```

## Implementation Details

The key insight is that `threading.RLock` works fine in async contexts for our use case:
- Progress updates are very fast operations
- The lock is held for microseconds
- No actual async I/O happens inside the lock
- This simplifies the code significantly

The tqdm integration is optional and gracefully degrades:
- Auto-detects TTY for appropriate display
- Falls back to logging when not in a terminal
- Can be explicitly controlled if needed
