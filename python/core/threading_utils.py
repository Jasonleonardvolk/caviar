"""
Job-stealing thread pool for better Strassen parallelism
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import os

# Global executor with optimal worker count
_WORKERS = min(7, max(1, os.cpu_count() - 1))  # Leave one core free
_EXEC = ThreadPoolExecutor(max_workers=_WORKERS)
_TASKQ = queue.Queue()

def strassen_parallel(tasks):
    """
    Execute Strassen subtasks with job-stealing semantics.
    
    Parameters
    ----------
    tasks : list of tuples
        Each tuple is (function, args, kwargs)
        
    Notes
    -----
    Workers that finish early will steal work from the queue,
    avoiding the "long tail" problem of unbalanced trees.
    """
    if not tasks:
        return []
    
    # Submit all tasks to the pool
    futures = []
    for func, args, kwargs in tasks:
        future = _EXEC.submit(func, *args, **kwargs)
        futures.append(future)
    
    # Wait for all to complete IN ORDER (critical!)
    # as_completed returns results out of order, which scrambles P1-P7
    results = []
    for future in futures:
        results.append(future.result())
    
    return results


def strassen_parallel_advanced(tasks):
    """
    Advanced job-stealing with dynamic work queue.
    Better for very unbalanced trees (e.g., after padding).
    """
    done = threading.Event()
    results = queue.Queue()
    
    def worker():
        while not done.is_set():
            try:
                func, args, kwargs = _TASKQ.get(timeout=0.001)
                result = func(*args, **kwargs)
                results.put(result)
                _TASKQ.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                results.put(e)
                _TASKQ.task_done()
    
    # Fill task queue
    for task in tasks:
        _TASKQ.put(task)
    
    # Start workers
    threads = []
    for _ in range(_WORKERS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Wait for completion
    _TASKQ.join()
    done.set()
    
    # Clean up threads
    for t in threads:
        t.join()
    
    # Collect results
    output = []
    while not results.empty():
        r = results.get()
        if isinstance(r, Exception):
            raise r
        output.append(r)
    
    return output


def shutdown():
    """Shutdown the thread pool gracefully"""
    _EXEC.shutdown(wait=True)


__all__ = ['strassen_parallel', 'strassen_parallel_advanced', 'shutdown']
