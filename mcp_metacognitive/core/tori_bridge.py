"""
TORI MCP Bridge Integration Module
==================================

This module provides the bridge functionality from mcp_server_arch
migrated to work with the mcp_metacognitive architecture.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import TORI filtering if available
try:
    from ingest_pdf.pipeline import (
        analyze_concept_purity, 
        is_rogue_concept_contextual,
        boost_known_concepts
    )
    from ingest_pdf.source_validator import validate_source, analyze_content_quality
    from ingest_pdf.pipeline_validator import validate_concepts
    TORI_FILTERING_AVAILABLE = True
except ImportError:
    TORI_FILTERING_AVAILABLE = False
    logger.warning("TORI filtering not available - using basic filtering")

class BasicTORIFilter:
    """Basic TORI filter implementation when full pipeline not available"""
    
    def __init__(self):
        self.logger = logging.getLogger("tori_filter")
        self.logger.info("Basic TORI Filter initialized")
    
    async def filter_input(self, content: Any) -> Any:
        """Basic input filtering"""
        if isinstance(content, str):
            # Basic content checks
            dangerous_patterns = ['<script', 'DROP TABLE', 'eval(', 'exec(']
            for pattern in dangerous_patterns:
                if pattern in content:
                    return "[CONTENT BLOCKED - DANGEROUS PATTERN]"
            return content
        elif isinstance(content, dict):
            filtered_dict = {}
            for key, value in content.items():
                filtered_dict[key] = await self.filter_input(value)
            return filtered_dict
        elif isinstance(content, list):
            return [await self.filter_input(item) for item in content]
        return content
    
    async def filter_output(self, content: Any) -> Any:
        """Basic output filtering"""
        return await self.filter_input(content)  # Same as input for basic
    
    async def filter_error(self, error: str) -> str:
        """Basic error filtering"""
        # Remove paths
        import re
        filtered = re.sub(r'[A-Z]:\\[^\s]+', '[PATH]', error)
        filtered = re.sub(r'/[^\s]+/', '[PATH]/', filtered)
        return filtered

@dataclass
class FilteredContent:
    """Content wrapper for filtering audit trail"""
    id: str
    original: Any
    filtered: Any
    tori_flags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    filtering_history: List[Dict] = field(default_factory=list)
    
    def add_filter_step(self, filter_name: str, result: str, details: Optional[Dict] = None):
        entry = {
            'filter': filter_name,
            'result': result,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        self.filtering_history.append(entry)

class TORIBridge:
    """Bridge for TORI filtering in metacognitive server"""
    
    def __init__(self):
        if TORI_FILTERING_AVAILABLE:
            # Import real filter if available
            from mcp_server_arch.solidownload.mcp_bridge_real_tori import RealTORIFilter
            self.filter = RealTORIFilter()
            logger.info("Using real TORI filter")
        else:
            self.filter = BasicTORIFilter()
            logger.info("Using basic TORI filter")
        
        self.metrics = {
            'requests_filtered': 0,
            'requests_blocked': 0
        }
    
    async def filter_content(self, content: Any, direction: str = "input") -> FilteredContent:
        """Filter content with audit trail"""
        import hashlib
        
        # Generate content ID
        content_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}-{id(content)}".encode()
        ).hexdigest()[:16]
        
        wrapped = FilteredContent(
            id=content_id,
            original=content,
            filtered=content
        )
        
        try:
            if direction == "input":
                filtered = await self.filter.filter_input(content)
            else:
                filtered = await self.filter.filter_output(content)
            
            wrapped.filtered = filtered
            wrapped.add_filter_step(f'tori.{direction}', 'passed')
            self.metrics['requests_filtered'] += 1
            
            # Check if blocked
            if isinstance(filtered, str) and "[CONTENT BLOCKED" in filtered:
                self.metrics['requests_blocked'] += 1
                wrapped.tori_flags.append('blocked')
            
        except Exception as e:
            logger.error(f"Filtering error: {e}")
            wrapped.filtered = "[FILTER ERROR]"
            wrapped.add_filter_step(f'tori.{direction}', 'error', {'error': str(e)})
        
        return wrapped

# Global bridge instance
tori_bridge = TORIBridge()
