"""
chunk_processor.py

Advanced chunk processing with parallelism optimizations.
Implements smart batching, early exit, prioritization, and backpressure.
"""

import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import multiprocessing

# Get logger
logger = logging.getLogger("tori.ingest_pdf.chunk_processor")

# Executor types
class ExecutorType(Enum):
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"

# Configuration
@dataclass
class ProcessingConfig:
    max_workers: Optional[int] = None
    executor_type: ExecutorType = ExecutorType.THREAD
    batch_size: int = 4
    max_concepts: Optional[int] = None
    queue_size: int = 100
    enable_profiling: bool = False
    prioritize_chunks: bool = True
    early_exit: bool = True
    
    def __post_init__(self):
        if self.max_workers is None:
            # Default based on executor type
            cpu_count = os.cpu_count() or 1
            if self.executor_type == ExecutorType.PROCESS:
                self.max_workers = cpu_count
            else:
                # 2x CPU for thread pool (good for mixed I/O and CPU)
                self.max_workers = min(16, cpu_count * 2)


class ChunkProcessor:
    """Advanced chunk processor with optimizations."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.executor = None
        self.early_stop_event = asyncio.Event()
        self.concepts_collected = 0
        self.processing_times = deque(maxlen=100)  # Track last 100 processing times
        
        # Initialize executor based on type
        if config.executor_type == ExecutorType.THREAD:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        elif config.executor_type == ExecutorType.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=config.max_workers)
        # ASYNC type doesn't need an executor
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            
    async def process_chunks(
        self,
        chunks: List[Dict],
        extraction_func: Callable,
        extraction_params: Dict,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Process chunks with advanced optimizations."""
        
        # Reset state
        self.early_stop_event.clear()
        self.concepts_collected = 0
        
        # Prioritize chunks if enabled
        if self.config.prioritize_chunks:
            chunks = self._prioritize_chunks(chunks)
            
        # Create result queue with backpressure
        result_queue = asyncio.Queue(maxsize=self.config.queue_size)
        
        # Start consumer task
        consumer_task = asyncio.create_task(
            self._consume_results(result_queue, progress_callback)
        )
        
        # Process chunks
        try:
            if self.config.batch_size > 1:
                # Batch processing
                await self._process_batched(
                    chunks, extraction_func, extraction_params, result_queue
                )
            else:
                # Individual processing
                await self._process_individual(
                    chunks, extraction_func, extraction_params, result_queue
                )
        finally:
            # Signal completion
            await result_queue.put(None)
            
        # Wait for consumer to finish
        all_concepts = await consumer_task
        
        return all_concepts
        
    def _prioritize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Prioritize chunks based on likely concept yield."""
        
        def chunk_score(chunk: Dict) -> float:
            """Score chunk based on heuristics."""
            text = chunk.get('text', '')
            section = chunk.get('section', 'body')
            
            # Prioritize by section
            section_scores = {
                'title': 3.0,
                'abstract': 2.5,
                'introduction': 2.0,
                'conclusion': 1.8,
                'methodology': 1.5,
                'body': 1.0
            }
            score = section_scores.get(section, 1.0)
            
            # Boost for certain keywords
            boost_keywords = ['define', 'concept', 'theory', 'framework', 'model']
            text_lower = text.lower()
            keyword_boost = sum(0.1 for kw in boost_keywords if kw in text_lower)
            score += min(keyword_boost, 0.5)
            
            # Boost for academic indicators
            if any(indicator in text for indicator in ['et al.', 'Fig.', 'Table']):
                score += 0.2
                
            return score
            
        # Sort by score descending
        return sorted(chunks, key=chunk_score, reverse=True)
        
    async def _process_batched(
        self,
        chunks: List[Dict],
        extraction_func: Callable,
        extraction_params: Dict,
        result_queue: asyncio.Queue
    ):
        """Process chunks in batches."""
        batches = list(self._batch_chunks(chunks, self.config.batch_size))
        tasks = []
        
        for batch_idx, batch in enumerate(batches):
            if self.config.early_exit and self.early_stop_event.is_set():
                logger.info(f"Early exit triggered at batch {batch_idx}")
                break
                
            task = asyncio.create_task(
                self._process_batch(
                    batch, batch_idx, extraction_func, 
                    extraction_params, result_queue
                )
            )
            tasks.append(task)
            
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _process_individual(
        self,
        chunks: List[Dict],
        extraction_func: Callable,
        extraction_params: Dict,
        result_queue: asyncio.Queue
    ):
        """Process chunks individually."""
        tasks = []
        
        for idx, chunk in enumerate(chunks):
            if self.config.early_exit and self.early_stop_event.is_set():
                logger.info(f"Early exit triggered at chunk {idx}")
                break
                
            task = asyncio.create_task(
                self._process_single_chunk(
                    chunk, idx, extraction_func,
                    extraction_params, result_queue
                )
            )
            tasks.append(task)
            
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _process_batch(
        self,
        batch: List[Dict],
        batch_idx: int,
        extraction_func: Callable,
        extraction_params: Dict,
        result_queue: asyncio.Queue
    ):
        """Process a batch of chunks."""
        if self.early_stop_event.is_set():
            return
            
        start_time = time.perf_counter()
        
        try:
            # Process batch based on executor type
            if self.config.executor_type == ExecutorType.ASYNC:
                # Pure async processing
                results = []
                for chunk in batch:
                    concepts = await self._extract_concepts_async(
                        chunk, extraction_func, extraction_params
                    )
                    results.extend(concepts)
            else:
                # Thread/Process pool processing
                loop = asyncio.get_event_loop()
                
                # Create tasks for batch
                tasks = []
                for chunk in batch:
                    task = loop.run_in_executor(
                        self.executor,
                        self._extract_concepts_sync,
                        chunk,
                        extraction_func,
                        extraction_params
                    )
                    tasks.append(task)
                    
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks)
                results = [c for concepts in batch_results for c in concepts]
                
            # Track timing
            elapsed = time.perf_counter() - start_time
            if self.config.enable_profiling:
                self.processing_times.append(elapsed)
                
            # Queue results
            await result_queue.put({
                'batch_idx': batch_idx,
                'concepts': results,
                'elapsed': elapsed,
                'chunk_count': len(batch)
            })
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            
    async def _process_single_chunk(
        self,
        chunk: Dict,
        idx: int,
        extraction_func: Callable,
        extraction_params: Dict,
        result_queue: asyncio.Queue
    ):
        """Process a single chunk."""
        if self.early_stop_event.is_set():
            return
            
        start_time = time.perf_counter()
        
        try:
            # Process based on executor type
            if self.config.executor_type == ExecutorType.ASYNC:
                concepts = await self._extract_concepts_async(
                    chunk, extraction_func, extraction_params
                )
            else:
                loop = asyncio.get_event_loop()
                concepts = await loop.run_in_executor(
                    self.executor,
                    self._extract_concepts_sync,
                    chunk,
                    extraction_func,
                    extraction_params
                )
                
            # Track timing
            elapsed = time.perf_counter() - start_time
            if self.config.enable_profiling:
                self.processing_times.append(elapsed)
                
            # Queue results
            await result_queue.put({
                'chunk_idx': idx,
                'concepts': concepts,
                'elapsed': elapsed,
                'chunk_count': 1
            })
            
        except Exception as e:
            logger.error(f"Error processing chunk {idx}: {e}")
            
    def _extract_concepts_sync(
        self,
        chunk: Dict,
        extraction_func: Callable,
        extraction_params: Dict
    ) -> List[Dict]:
        """Synchronous concept extraction."""
        chunk_text = chunk.get('text', '')
        chunk_index = chunk.get('index', 0)
        chunk_section = chunk.get('section', 'body')
        
        return extraction_func(
            chunk_text,
            extraction_params['threshold'],
            chunk_index,
            chunk_section,
            extraction_params['title'],
            extraction_params['abstract']
        )
        
    async def _extract_concepts_async(
        self,
        chunk: Dict,
        extraction_func: Callable,
        extraction_params: Dict
    ) -> List[Dict]:
        """Async concept extraction (if extraction_func is async)."""
        # This assumes extraction_func can be async
        # For now, wrap sync function
        return await asyncio.to_thread(
            self._extract_concepts_sync,
            chunk,
            extraction_func,
            extraction_params
        )
        
    async def _consume_results(
        self,
        result_queue: asyncio.Queue,
        progress_callback: Optional[Callable]
    ) -> List[Dict]:
        """Consume results from queue with early exit logic."""
        all_concepts = []
        chunks_processed = 0
        
        while True:
            result = await result_queue.get()
            
            # Check for completion signal
            if result is None:
                break
                
            concepts = result['concepts']
            chunks_processed += result['chunk_count']
            
            # Add concepts
            all_concepts.extend(concepts)
            self.concepts_collected = len(all_concepts)
            
            # Progress callback
            if progress_callback:
                try:
                    progress_callback(
                        "concepts",
                        min(90, 40 + int((chunks_processed / 100) * 50)),
                        f"Processed {chunks_processed} chunks, {len(all_concepts)} concepts"
                    )
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")
                    
            # Check early exit
            if self.config.max_concepts and len(all_concepts) >= self.config.max_concepts:
                logger.info(f"Reached max concepts limit: {len(all_concepts)}")
                self.early_stop_event.set()
                # Trim to max
                all_concepts = all_concepts[:self.config.max_concepts]
                break
                
            # Log profiling info periodically
            if self.config.enable_profiling and chunks_processed % 10 == 0:
                avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                logger.debug(f"Avg processing time: {avg_time:.3f}s per chunk/batch")
                
        return all_concepts
        
    @staticmethod
    def _batch_chunks(chunks: List[Dict], batch_size: int):
        """Yield batches of chunks."""
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'concepts_collected': self.concepts_collected,
            'avg_processing_time': avg_time,
            'max_workers': self.config.max_workers,
            'executor_type': self.config.executor_type.value,
            'batch_size': self.config.batch_size,
            'early_exit_triggered': self.early_stop_event.is_set()
        }


# Factory function for easy creation
def create_chunk_processor(
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    batch_size: int = 4,
    max_concepts: Optional[int] = None,
    enable_profiling: bool = False
) -> ChunkProcessor:
    """Create a chunk processor with sensible defaults."""
    
    config = ProcessingConfig(
        max_workers=max_workers,
        executor_type=ExecutorType.PROCESS if use_processes else ExecutorType.THREAD,
        batch_size=batch_size,
        max_concepts=max_concepts,
        enable_profiling=enable_profiling
    )
    
    return ChunkProcessor(config)
