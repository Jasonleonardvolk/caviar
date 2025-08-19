"""
ingest_common/progress.py

Progress tracking for ingestion pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressTracker:
    """Track progress of ingestion operations"""
    current: int = 0
    total: int = 0
    stage: str = "init"
    message: str = ""
    callback: Optional[Callable] = None
    
    async def update(self, current: int, total: int, stage: str = "", message: str = ""):
        """Update progress state"""
        self.current = current
        self.total = total
        if stage:
            self.stage = stage
        if message:
            self.message = message
            
        if self.callback:
            try:
                await self.callback({
                    "current": self.current,
                    "total": self.total,
                    "stage": self.stage,
                    "message": self.message,
                    "progress": self.current / self.total if self.total > 0 else 0
                })
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    async def send_progress(self, stage: str, progress: float, message: str):
        """Send progress update (compatible with legacy API)"""
        await self.update(
            current=int(progress),
            total=100,
            stage=stage,
            message=message
        )
    
    def __init__(self, callback: Optional[Callable] = None):
        """Initialize with optional callback"""
        self.callback = callback
        self.current = 0
        self.total = 0
        self.stage = "init"
        self.message = ""
