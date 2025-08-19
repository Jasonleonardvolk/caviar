"""
TORI/KHA Unified Memory Vault - Production-Ready Implementation
Fully async, atomic operations, optimized for scale
"""

import json
import pickle
import hashlib
import time
import logging
import asyncio
import os
import signal
import atexit
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import gzip
import shutil
import aiofiles
import aiofiles.os
from contextlib import asynccontextmanager
import msgpack
import sys
import heapq
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    GHOST = "ghost"
    SOLITON = "soliton"

@dataclass
class MemoryEntry:
    """Single memory entry with optimized serialization"""
    id: str
    type: MemoryType
    content: Any
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: Optional[float] = None
    decay_rate: float = 0.0
    importance: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['type'] = self.type.value
        # Don't include embedding in main dict - store separately
        data.pop('embedding', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str,