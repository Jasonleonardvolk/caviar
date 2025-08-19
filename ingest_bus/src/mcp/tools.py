"""
MCP tools for ingest-bus service.

This module provides the MCP tools for the ingest-bus service:
- ingest.queue: Queue a document for ingestion
- ingest.status: Get the status of an ingest job
- ingest.metrics: Get metrics about the ingest process
"""

import json
import os
import re
import logging
from typing import Dict, Any, List, Optional, Callable

from ..services.ingest_service import IngestService
from ..services.metrics_service import get_metrics_service

# Configure logging
logger = logging.getLogger(__name__)


class MCPToolContext:
    """
    Context for MCP tools
    
    Provides access to services and configuration for MCP tools.
    """
    
    def __init__(self, ingest_service: IngestService, build_hash: str = 'dev'):
        """
        Initialize the MCP tool context
        
        Args:
            ingest_service: Ingest service
            build_hash: Git hash or build identifier
        """
        self.ingest_service = ingest_service
        self.build_hash = build_hash
        self.metrics_service = get_metrics_service(build_hash)


def get_queue_tool(context: MCPToolContext) -> Callable:
    """
    Create the ingest.queue tool
    
    Args:
        context: MCP tool context
        
    Returns:
        The tool function
    """
    
    async def queue_tool(file_url: str, track: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Queue a document for ingestion
        
        Args:
            file_url: URL to the file to process
            track: Track to assign the document to (optional, will be detected from URL if not provided)
            metadata: Additional metadata (optional)
            
        Returns:
            Job information
        """
        # Validate URL
        if not file_url or not isinstance(file_url, str):
            return {
                "success": False,
                "error": "Missing or invalid file_url parameter"
            }
        
        # Normalize URL
        file_url = file_url.strip()
        
        # Validate track if provided
        valid_tracks = {'programming', 'math_physics', 'ai_ml', 'domain', 'ops_sre'}
        if track and track not in valid_tracks:
            return {
                "success": False,
                "error": f"Invalid track. Must be one of: {', '.join(valid_tracks)}"
            }
        
        # Queue the job
        try:
            job = await context.ingest_service.queue_job(
                file_url=file_url,
                track=track,
                metadata=metadata or {}
            )
            
            # Record metric
            context.metrics_service.record_job_queued(job.job_id, job.track)
            
            return {
                "success": True,
                "job_id": job.job_id,
                "file_url": job.file_url,
                "track": job.track,
                "status": job.status.value,
                "created_at": job.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error queueing job: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error queueing job: {str(e)}"
            }
    
    return queue_tool


def get_status_tool(context: MCPToolContext) -> Callable:
    """
    Create the ingest.status tool
    
    Args:
        context: MCP tool context
        
    Returns:
        The tool function
    """
    
    async def status_tool(job_id: Optional[str] = None, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Get the status of an ingest job or list of jobs
        
        Args:
            job_id: Job ID (optional, if not provided, returns a list of jobs)
            limit: Maximum number of jobs to return (optional, default: 10)
            offset: Number of jobs to skip (optional, default: 0)
            
        Returns:
            Job status information
        """
        # If job_id is provided, get that specific job
        if job_id:
            try:
                job = await context.ingest_service.get_job(job_id)
                if not job:
                    return {
                        "success": False,
                        "error": f"Job {job_id} not found"
                    }
                
                # Build detailed job status response
                chunks_info = []
                # Only include up to 5 chunks to avoid large responses
                for chunk in job.chunks[:5]:
                    chunks_info.append({
                        'id': chunk.id,
                        'start_offset': chunk.start_offset,
                        'end_offset': chunk.end_offset,
                        'metadata': chunk.metadata
                    })
                
                return {
                    "success": True,
                    "job": {
                        "job_id": job.job_id,
                        "file_url": job.file_url,
                        "file_name": job.file_name,
                        "file_size": job.file_size,
                        "file_sha256": job.file_sha256,
                        "track": job.track,
                        "status": job.status.value,
                        "created_at": job.created_at.isoformat(),
                        "updated_at": job.updated_at.isoformat(),
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "processing_stage": job.processing_stage.value,
                        "progress": job.progress,
                        "error": job.error,
                        "chunk_count": job.chunk_count,
                        "chunks_preview": chunks_info,
                        "concept_ids": job.concept_ids
                    }
                }
            except Exception as e:
                logger.error(f"Error getting job status: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Error getting job status: {str(e)}"
                }
        
        # Otherwise, get a list of jobs
        try:
            # Validate limit and offset
            limit = max(1, min(100, limit))  # Between 1 and 100
            offset = max(0, offset)
            
            # Get jobs
            jobs = await context.ingest_service.get_jobs(limit=limit, offset=offset)
            
            # Build response
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "job_id": job.job_id,
                    "file_url": job.file_url,
                    "file_name": job.file_name,
                    "track": job.track,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.updated_at.isoformat(),
                    "progress": job.progress,
                    "chunk_count": job.chunk_count
                })
            
            return {
                "success": True,
                "jobs": jobs_data,
                "count": len(jobs_data),
                "offset": offset,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error listing jobs: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error listing jobs: {str(e)}"
            }
    
    return status_tool


def get_metrics_tool(context: MCPToolContext) -> Callable:
    """
    Create the ingest.metrics tool
    
    Args:
        context: MCP tool context
        
    Returns:
        The tool function
    """
    
    async def metrics_tool() -> Dict[str, Any]:
        """
        Get metrics about the ingest process
        
        Returns:
            Metrics data
        """
        try:
            # Get stats from ingest service
            stats = context.ingest_service.get_stats()
            
            # Add build info
            stats['build_hash'] = context.build_hash
            
            return {
                "success": True,
                "metrics": stats
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error getting metrics: {str(e)}"
            }
    
    return metrics_tool


def create_mcp_tools(ingest_service: IngestService, build_hash: str = 'dev') -> Dict[str, Callable]:
    """
    Create MCP tools
    
    Args:
        ingest_service: Ingest service
        build_hash: Git hash or build identifier
        
    Returns:
        Dictionary of MCP tools
    """
    # Create context
    context = MCPToolContext(ingest_service, build_hash)
    
    # Create tools
    return {
        'ingest.queue': get_queue_tool(context),
        'ingest.status': get_status_tool(context),
        'ingest.metrics': get_metrics_tool(context)
    }
