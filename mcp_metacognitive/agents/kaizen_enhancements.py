"""
Circuit Breaker and Additional Enhancements for Kaizen
======================================================

This module provides circuit breaker functionality and additional helpers.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Simple circuit breaker for external service calls.
    
    States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
    """
    
    def __init__(self, 
                 failure_threshold: int = 3,
                 recovery_timeout: int = 300,  # 5 minutes
                 success_threshold: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.state = "CLOSED"
        
    def call_succeeded(self):
        """Record successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                logger.info("Circuit breaker closed (recovered)")
                
    def call_failed(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def can_execute(self) -> bool:
        """Check if calls are allowed"""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker half-open (testing recovery)")
                    return True
            return False
            
        # HALF_OPEN state
        return True
        
    def is_open(self) -> bool:
        """Check if circuit is open (failing)"""
        return self.state == "OPEN"


async def with_circuit_breaker(circuit_breaker: CircuitBreaker,
                              func: Callable,
                              *args,
                              **kwargs):
    """
    Execute function with circuit breaker protection.
    
    Example:
        result = await with_circuit_breaker(
            gap_fill_breaker,
            trigger_gap_fill_search,
            query="some query"
        )
    """
    if not circuit_breaker.can_execute():
        raise Exception(f"Circuit breaker is OPEN - service unavailable for {circuit_breaker.recovery_timeout}s")
        
    try:
        result = await func(*args, **kwargs)
        circuit_breaker.call_succeeded()
        return result
    except Exception as e:
        circuit_breaker.call_failed()
        raise


class KnowledgeBaseRotator:
    """
    Handles knowledge base rotation and compression.
    """
    
    def __init__(self, kb_path: str, keep_days: int = 30):
        self.kb_path = Path(kb_path)
        self.keep_days = keep_days
        
    async def rotate_if_needed(self):
        """Rotate KB file if it's a new day"""
        if not self.kb_path.exists():
            return
            
        # Check file modification time
        mtime = datetime.fromtimestamp(self.kb_path.stat().st_mtime)
        if mtime.date() < datetime.now().date():
            await self._rotate_kb_file()
            
    async def _rotate_kb_file(self):
        """Rotate and compress old KB file"""
        import gzip
        import shutil
        
        try:
            # Create dated filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            rotated_name = f"{self.kb_path.stem}.{date_str}.json"
            rotated_path = self.kb_path.parent / rotated_name
            
            # Copy current KB to dated file
            shutil.copy2(self.kb_path, rotated_path)
            
            # Compress the rotated file
            compressed_path = rotated_path.with_suffix('.json.gz')
            with open(rotated_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Remove uncompressed rotated file
            rotated_path.unlink()
            
            logger.info(f"Rotated KB to {compressed_path}")
            
            # Clean up old files
            await self._cleanup_old_rotations()
            
        except Exception as e:
            logger.error(f"Failed to rotate KB file: {e}")
            
    async def _cleanup_old_rotations(self):
        """Remove rotated files older than keep_days"""
        cutoff_date = datetime.now() - timedelta(days=self.keep_days)
        
        for gz_file in self.kb_path.parent.glob(f"{self.kb_path.stem}.*.json.gz"):
            try:
                # Extract date from filename
                date_str = gz_file.stem.split('.')[-1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    gz_file.unlink()
                    logger.info(f"Removed old KB rotation: {gz_file}")
                    
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue


# Enhanced gap-fill trigger with circuit breaker
async def trigger_gap_fill_with_breaker(breaker: CircuitBreaker,
                                       query: str,
                                       mcp_bridge=None) -> bool:
    """
    Trigger gap-fill search with circuit breaker protection.
    """
    async def _do_gap_fill():
        if mcp_bridge is None:
            from ..core.mcp_bridge import mcp_bridge as bridge
            mcp_bridge_instance = bridge
        else:
            mcp_bridge_instance = mcp_bridge
            
        mcp_bridge_instance.dispatch("paper_search", {"query": query})
        logger.info(f"Triggered paper search for gap: {query}")
        return True
        
    try:
        return await with_circuit_breaker(breaker, _do_gap_fill)
    except Exception as e:
        logger.error(f"Gap-fill trigger failed: {e}")
        return False


# Celery task wrapper (if Celery is available)
try:
    from celery import Celery, Task
    
    class KaizenCeleryTasks:
        """Celery tasks for heavy Kaizen operations"""
        
        def __init__(self, app: Celery):
            self.app = app
            
        def create_tasks(self):
            """Create Celery tasks for Kaizen"""
            
            @self.app.task(name='kaizen.analyze_query_patterns')
            def analyze_query_patterns_task(patterns: Dict[str, int]) -> List[Dict]:
                """Heavy clustering analysis as Celery task"""
                # Implementation would go here
                pass
                
            @self.app.task(name='kaizen.deep_error_analysis')  
            def deep_error_analysis_task(error_data: Dict) -> List[Dict]:
                """Deep error pattern analysis as Celery task"""
                # Implementation would go here
                pass
                
            return {
                'analyze_query_patterns': analyze_query_patterns_task,
                'deep_error_analysis': deep_error_analysis_task
            }
            
except ImportError:
    KaizenCeleryTasks = None


__all__ = [
    'CircuitBreaker',
    'with_circuit_breaker',
    'KnowledgeBaseRotator',
    'trigger_gap_fill_with_breaker',
    'KaizenCeleryTasks'
]
