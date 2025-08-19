#!/usr/bin/env python3
"""
Distributed Processing with Ray for TORI/KHA
Scales chaos computing across multiple machines without containers
File-based coordination compatible with MCP servers
"""

import asyncio
import json
import time
import logging
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import threading
from concurrent.futures import ProcessPoolExecutor
import socket
import platform

# Try to import Ray
try:
    import ray
    from ray import serve
    from ray.util.state import list_nodes
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available - using file-based distributed processing")

# Import TORI components
from python.core.chaos_control_layer import (
    ChaosTask, ChaosResult, ChaosMode, 
    DarkSolitonProcessor, AttractorHopper, PhaseExplosionEngine
)
from python.core.eigensentry.core import InstabilityEvent, InstabilityType
from python.core.gpu_eigenvalue_monitor import GPUEigenvalueMonitor

logger = logging.getLogger(__name__)

# ========== File-Based Coordination ==========

@dataclass
class DistributedTask:
    """Task for distributed processing"""
    task_id: str
    task_type: str  # "chaos", "eigenvalue", "memory", "query"
    payload: Dict[str, Any]
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, assigned, processing, completed, failed
    created_at: float = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class FileBasedCoordinator:
    """
    File-based distributed task coordinator
    Alternative to Ray when not available
    """
    
    def __init__(self, coordination_path: Path = Path("data/distributed")):
        self.coordination_path = coordination_path
        self.tasks_path = coordination_path / "tasks"
        self.workers_path = coordination_path / "workers"
        self.results_path = coordination_path / "results"
        self.locks_path = coordination_path / "locks"
        
        # Create directories
        for path in [self.tasks_path, self.workers_path, self.results_path, self.locks_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Worker info
        self.worker_id = f"{platform.node()}_{socket.gethostname()}_{hash(time.time())}"
        self.worker_file = self.workers_path / f"{self.worker_id}.json"
        
        # Task queue
        self.pending_tasks = deque()
        self.active_tasks = {}
        
        # Start heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        logger.info(f"FileBasedCoordinator initialized with worker_id: {self.worker_id}")
    
    def _heartbeat_loop(self):
        """Maintain worker heartbeat"""
        while True:
            try:
                worker_info = {
                    "worker_id": self.worker_id,
                    "hostname": socket.gethostname(),
                    "last_heartbeat": time.time(),
                    "active_tasks": len(self.active_tasks),
                    "status": "healthy"
                }
                
                with open(self.worker_file, 'w') as f:
                    json.dump(worker_info, f)
                
                time.sleep(5)  # Heartbeat every 5 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed processing"""
        # Save task to file
        task_file = self.tasks_path / f"{task.task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(asdict(task), f)
        
        # Add to pending queue
        self.pending_tasks.append(task.task_id)
        
        logger.info(f"Task {task.task_id} submitted for distributed processing")
        return task.task_id
    
    def claim_task(self) -> Optional[DistributedTask]:
        """Claim a task for processing"""
        # Look for pending tasks
        for task_file in sorted(self.tasks_path.glob("*.json")):
            try:
                # Try to acquire lock
                lock_file = self.locks_path / f"{task_file.stem}.lock"
                
                if lock_file.exists():
                    continue  # Already locked
                
                # Atomic lock creation
                try:
                    lock_file.touch(exist_ok=False)
                except FileExistsError:
                    continue
                
                # Read task
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                
                task = DistributedTask(**task_data)
                
                # Check if still pending
                if task.status != "pending":
                    lock_file.unlink()
                    continue
                
                # Claim task
                task.assigned_worker = self.worker_id
                task.status = "assigned"
                
                # Update task file
                with open(task_file, 'w') as f:
                    json.dump(asdict(task), f)
                
                # Release lock
                lock_file.unlink()
                
                self.active_tasks[task.task_id] = task
                return task
                
            except Exception as e:
                logger.error(f"Error claiming task: {e}")
                if lock_file.exists():
                    lock_file.unlink()
        
        return None
    
    def complete_task(self, task_id: str, result: Any):
        """Mark task as completed with result"""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not in active tasks")
            return
        
        task = self.active_tasks[task_id]
        task.status = "completed"
        task.completed_at = time.time()
        task.result = result
        
        # Save result
        result_file = self.results_path / f"{task_id}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Update task file
        task_file = self.tasks_path / f"{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(asdict(task), f)
        
        # Remove from active
        del self.active_tasks[task_id]
        
        logger.info(f"Task {task_id} completed")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task.status = "failed"
        task.completed_at = time.time()
        task.error = error
        
        # Update task file
        task_file = self.tasks_path / f"{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(asdict(task), f)
        
        # Remove from active
        del self.active_tasks[task_id]
        
        logger.error(f"Task {task_id} failed: {error}")
    
    def get_task_result(self, task_id: str, timeout: float = 60.0) -> Optional[Any]:
        """Get task result (blocking)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check task status
            task_file = self.tasks_path / f"{task_id}.json"
            
            if task_file.exists():
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                
                if task_data['status'] == 'completed':
                    # Load result
                    result_file = self.results_path / f"{task_id}.pkl"
                    if result_file.exists():
                        with open(result_file, 'rb') as f:
                            return pickle.load(f)
                elif task_data['status'] == 'failed':
                    raise Exception(f"Task failed: {task_data.get('error', 'Unknown error')}")
            
            time.sleep(0.5)
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    def list_workers(self) -> List[Dict[str, Any]]:
        """List active workers"""
        workers = []
        current_time = time.time()
        
        for worker_file in self.workers_path.glob("*.json"):
            try:
                with open(worker_file, 'r') as f:
                    worker_info = json.load(f)
                
                # Check if alive (heartbeat within 30s)
                if current_time - worker_info['last_heartbeat'] < 30:
                    workers.append(worker_info)
                else:
                    # Clean up dead worker
                    worker_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error reading worker file: {e}")
        
        return workers

# ========== Ray-based Distributed Processing ==========

if RAY_AVAILABLE:
    @ray.remote(num_cpus=1, num_gpus=0)
    class RayChaosWorker:
        """Ray actor for chaos computation"""
        
        def __init__(self, worker_id: str):
            self.worker_id = worker_id
            self.soliton_processor = DarkSolitonProcessor()
            self.attractor_hopper = AttractorHopper()
            self.phase_engine = PhaseExplosionEngine()
            self.processed_count = 0
            
        async def process_chaos_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            """Process chaos computation task"""
            start_time = time.time()
            
            mode = ChaosMode[task['mode']]
            input_data = np.array(task['input_data'])
            parameters = task.get('parameters', {})
            
            try:
                if mode == ChaosMode.DARK_SOLITON:
                    # Dark soliton processing
                    encoded = self.soliton_processor.encode_memory(input_data)
                    trajectory = self.soliton_processor.propagate(encoded, 
                                                                 parameters.get('time_steps', 100))
                    output = self.soliton_processor.decode_memory(trajectory[-1])
                    
                elif mode == ChaosMode.ATTRACTOR_HOP:
                    # Attractor hopping
                    target = parameters.get('target', np.zeros_like(input_data))
                    objective = lambda s: -np.linalg.norm(s - target)
                    output, best_value = self.attractor_hopper.search(
                        objective, input_data, parameters.get('max_hops', 50)
                    )
                    
                elif mode == ChaosMode.PHASE_EXPLOSION:
                    # Phase explosion
                    self.phase_engine.phases[:len(input_data)] = input_data % (2 * np.pi)
                    trajectory = self.phase_engine.trigger_explosion(
                        parameters.get('explosion_strength', 1.0)
                    )
                    patterns = self.phase_engine.extract_patterns(trajectory)
                    output = np.array([len(p) for p in patterns[:len(input_data)]])
                    
                else:
                    raise ValueError(f"Unknown chaos mode: {mode}")
                
                self.processed_count += 1
                
                return {
                    'success': True,
                    'output_data': output.tolist(),
                    'processing_time': time.time() - start_time,
                    'worker_id': self.worker_id,
                    'processed_count': self.processed_count
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time,
                    'worker_id': self.worker_id
                }
        
        def get_stats(self) -> Dict[str, Any]:
            """Get worker statistics"""
            return {
                'worker_id': self.worker_id,
                'processed_count': self.processed_count,
                'status': 'active'
            }
    
    @ray.remote(num_cpus=2, num_gpus=0.5 if torch.cuda.is_available() else 0)
    class RayEigenvalueWorker:
        """Ray actor for eigenvalue computation"""
        
        def __init__(self, worker_id: str):
            self.worker_id = worker_id
            self.monitor = GPUEigenvalueMonitor({
                'use_gpu': True,
                'cache_size': 1000
            })
            self.processed_count = 0
            
        async def analyze_matrix(self, matrix: List[List[float]]) -> Dict[str, Any]:
            """Analyze matrix eigenvalues"""
            matrix_np = np.array(matrix)
            
            try:
                analysis = await self.monitor.analyze_matrix(matrix_np)
                self.processed_count += 1
                
                return {
                    'success': True,
                    'eigenvalues': analysis.eigenvalues.tolist(),
                    'max_eigenvalue': analysis.max_eigenvalue,
                    'is_stable': analysis.is_stable,
                    'condition_number': analysis.condition_number,
                    'worker_id': self.worker_id,
                    'gpu_used': self.monitor.gpu_backend is not None
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'worker_id': self.worker_id
                }
        
        async def batch_analyze(self, matrices: List[List[List[float]]]) -> List[Dict[str, Any]]:
            """Batch analyze multiple matrices"""
            matrices_np = [np.array(m) for m in matrices]
            results = await self.monitor.batch_analyze_matrices(matrices_np)
            
            self.processed_count += len(matrices)
            
            return [
                {
                    'eigenvalues': r.eigenvalues.tolist(),
                    'max_eigenvalue': r.max_eigenvalue,
                    'is_stable': r.is_stable
                }
                for r in results
            ]

# ========== Distributed TORI System ==========

class DistributedTORI:
    """
    Distributed TORI system using Ray or file-based coordination
    """
    
    def __init__(self, 
                 num_chaos_workers: int = 4,
                 num_eigen_workers: int = 2,
                 use_ray: bool = True,
                 coordination_path: Path = Path("data/distributed")):
        
        self.use_ray = use_ray and RAY_AVAILABLE
        self.coordination_path = coordination_path
        
        if self.use_ray:
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(
                    address='auto',  # Connect to existing cluster or start local
                    namespace='tori',
                    runtime_env={
                        "working_dir": ".",
                        "pip": ["numpy", "scipy", "torch"]
                    }
                )
            
            # Create Ray actors
            self.chaos_workers = [
                RayChaosWorker.remote(f"chaos_worker_{i}")
                for i in range(num_chaos_workers)
            ]
            
            self.eigen_workers = [
                RayEigenvalueWorker.remote(f"eigen_worker_{i}")
                for i in range(num_eigen_workers)
            ]
            
            # Task queues
            self.chaos_queue = RayQueue(maxsize=1000)
            self.eigen_queue = RayQueue(maxsize=1000)
            
            logger.info(f"Initialized Ray distributed system with {num_chaos_workers} chaos workers and {num_eigen_workers} eigen workers")
            
        else:
            # Use file-based coordination
            self.coordinator = FileBasedCoordinator(coordination_path)
            
            # Local worker pools
            self.chaos_executor = ProcessPoolExecutor(max_workers=num_chaos_workers)
            self.eigen_executor = ProcessPoolExecutor(max_workers=num_eigen_workers)
            
            # Start worker loops
            self.worker_threads = []
            for i in range(num_chaos_workers):
                thread = threading.Thread(
                    target=self._file_based_worker_loop,
                    args=("chaos", self._process_chaos_local),
                    daemon=True
                )
                thread.start()
                self.worker_threads.append(thread)
            
            logger.info(f"Initialized file-based distributed system with {num_chaos_workers} chaos workers")
    
    async def submit_chaos_task(self, 
                              mode: ChaosMode,
                              input_data: np.ndarray,
                              parameters: Dict[str, Any] = None) -> ChaosResult:
        """Submit chaos task for distributed processing"""
        
        task = {
            'mode': mode.value,
            'input_data': input_data.tolist(),
            'parameters': parameters or {}
        }
        
        if self.use_ray:
            # Submit to Ray
            await self.chaos_queue.put(task)
            
            # Get available worker
            worker = self.chaos_workers[hash(str(task)) % len(self.chaos_workers)]
            
            # Process
            result = await worker.process_chaos_task.remote(task)
            
            if result['success']:
                return ChaosResult(
                    task_id=str(hash(str(task))),
                    success=True,
                    output_data=np.array(result['output_data']),
                    energy_used=100,  # Simplified
                    computation_time=result['processing_time'],
                    efficiency_gain=3.0  # Simplified
                )
            else:
                raise Exception(result['error'])
                
        else:
            # Use file-based coordination
            task_obj = DistributedTask(
                task_id=str(uuid.uuid4()),
                task_type="chaos",
                payload=task
            )
            
            task_id = self.coordinator.submit_task(task_obj)
            result = self.coordinator.get_task_result(task_id)
            
            return ChaosResult(**result)
    
    async def analyze_eigenvalues_distributed(self, 
                                            matrices: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze eigenvalues of multiple matrices in parallel"""
        
        if self.use_ray:
            # Distribute across workers
            batch_size = max(1, len(matrices) // len(self.eigen_workers))
            futures = []
            
            for i in range(0, len(matrices), batch_size):
                batch = matrices[i:i+batch_size]
                batch_list = [m.tolist() for m in batch]
                
                worker = self.eigen_workers[i // batch_size % len(self.eigen_workers)]
                future = worker.batch_analyze.remote(batch_list)
                futures.append(future)
            
            # Gather results
            results = []
            for future in futures:
                batch_results = await future
                results.extend(batch_results)
            
            return results
            
        else:
            # Use file-based coordination
            results = []
            
            for matrix in matrices:
                task = DistributedTask(
                    task_id=str(uuid.uuid4()),
                    task_type="eigenvalue",
                    payload={'matrix': matrix.tolist()}
                )
                
                task_id = self.coordinator.submit_task(task)
                result = self.coordinator.get_task_result(task_id)
                results.append(result)
            
            return results
    
    def _file_based_worker_loop(self, task_type: str, processor: Callable):
        """Worker loop for file-based coordination"""
        while True:
            try:
                # Claim a task
                task = self.coordinator.claim_task()
                
                if task and task.task_type == task_type:
                    try:
                        # Process task
                        result = processor(task.payload)
                        self.coordinator.complete_task(task.task_id, result)
                    except Exception as e:
                        self.coordinator.fail_task(task.task_id, str(e))
                else:
                    time.sleep(1)  # No task available
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(5)
    
    def _process_chaos_local(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process chaos task locally"""
        # This would be similar to RayChaosWorker.process_chaos_task
        # but running in local process
        processor = DarkSolitonProcessor()  # Simplified
        
        mode = ChaosMode[payload['mode']]
        input_data = np.array(payload['input_data'])
        
        # Process based on mode (simplified)
        if mode == ChaosMode.DARK_SOLITON:
            encoded = processor.encode_memory(input_data)
            trajectory = processor.propagate(encoded, 100)
            output = processor.decode_memory(trajectory[-1])
        else:
            output = input_data  # Simplified
        
        return {
            'task_id': str(uuid.uuid4()),
            'success': True,
            'output_data': output,
            'energy_used': 100,
            'computation_time': 0.1,
            'efficiency_gain': 3.0
        }
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of distributed cluster"""
        
        if self.use_ray:
            # Get Ray cluster info
            nodes = list_nodes()
            
            return {
                'backend': 'ray',
                'nodes': len(nodes),
                'chaos_workers': len(self.chaos_workers),
                'eigen_workers': len(self.eigen_workers),
                'ray_version': ray.__version__,
                'cluster_resources': ray.cluster_resources()
            }
        else:
            # Get file-based cluster info
            workers = self.coordinator.list_workers()
            
            return {
                'backend': 'file_based',
                'active_workers': len(workers),
                'coordination_path': str(self.coordination_path),
                'pending_tasks': len(self.coordinator.pending_tasks),
                'active_tasks': len(self.coordinator.active_tasks)
            }
    
    def scale_workers(self, chaos_workers: Optional[int] = None, 
                     eigen_workers: Optional[int] = None):
        """Scale the number of workers"""
        
        if self.use_ray:
            # Scale Ray actors
            if chaos_workers is not None:
                current = len(self.chaos_workers)
                if chaos_workers > current:
                    # Add workers
                    for i in range(current, chaos_workers):
                        self.chaos_workers.append(
                            RayChaosWorker.remote(f"chaos_worker_{i}")
                        )
                elif chaos_workers < current:
                    # Remove workers
                    for i in range(chaos_workers, current):
                        ray.kill(self.chaos_workers[i])
                    self.chaos_workers = self.chaos_workers[:chaos_workers]
            
            logger.info(f"Scaled to {len(self.chaos_workers)} chaos workers")
            
        else:
            logger.warning("Dynamic scaling not supported for file-based backend")

# ========== Integration with TORI ==========

class DistributedTORIIntegration:
    """Integrate distributed processing with TORI system"""
    
    def __init__(self, tori_system, distributed_tori: DistributedTORI):
        self.tori = tori_system
        self.distributed = distributed_tori
        
        # Replace local processors with distributed versions
        self._patch_processors()
    
    def _patch_processors(self):
        """Replace local processors with distributed versions"""
        
        # Patch CCL chaos processing
        original_process = self.tori.ccl._process_inline
        
        async def distributed_chaos_process(task):
            # Use distributed processing
            result = await self.distributed.submit_chaos_task(
                task.mode,
                task.input_data,
                task.parameters
            )
            return result
        
        self.tori.ccl._process_inline = distributed_chaos_process
        
        # Patch eigenvalue analysis
        if hasattr(self.tori.eigen_sentry, 'monitor'):
            original_analyze = self.tori.eigen_sentry.monitor.analyze_matrix
            
            async def distributed_eigen_analyze(matrix):
                results = await self.distributed.analyze_eigenvalues_distributed([matrix])
                return results[0] if results else None
            
            self.tori.eigen_sentry.monitor.analyze_matrix = distributed_eigen_analyze
        
        logger.info("TORI system patched for distributed processing")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_distributed():
        """Test distributed processing"""
        
        # Initialize distributed system
        distributed = DistributedTORI(
            num_chaos_workers=4,
            num_eigen_workers=2,
            use_ray=RAY_AVAILABLE
        )
        
        print(f"Cluster status: {distributed.get_cluster_status()}")
        
        # Test chaos processing
        print("\nTesting distributed chaos processing...")
        test_data = np.random.randn(100)
        
        result = await distributed.submit_chaos_task(
            ChaosMode.DARK_SOLITON,
            test_data,
            {'time_steps': 50}
        )
        
        print(f"Chaos result: success={result.success}, time={result.computation_time:.3f}s")
        
        # Test eigenvalue analysis
        print("\nTesting distributed eigenvalue analysis...")
        test_matrices = [np.random.randn(10, 10) for _ in range(5)]
        
        results = await distributed.analyze_eigenvalues_distributed(test_matrices)
        
        for i, result in enumerate(results):
            print(f"Matrix {i}: max_eigenvalue={result['max_eigenvalue']:.3f}, stable={result['is_stable']}")
        
        # Scale workers
        print("\nScaling workers...")
        distributed.scale_workers(chaos_workers=8)
        
        print(f"Updated status: {distributed.get_cluster_status()}")
    
    asyncio.run(test_distributed())
