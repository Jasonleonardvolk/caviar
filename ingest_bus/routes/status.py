"""
Status route module for TORI Ingest Bus.

This module provides API endpoints for checking the status of ingest jobs
and processing statistics.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from models.schemas import IngestStatus, IngestJob, IngestJobUpdate

# Import the job store from queue module
from routes.queue import jobs

# Set up logging
logger = logging.getLogger("ingest-bus.status")

# Initialize router
router = APIRouter()

@router.get("/", tags=["status"])
async def get_service_status():
    """
    Get the overall status of the ingest service.
    
    Returns statistics about current jobs and processing rates.
    """
    # Count jobs by status
    status_counts = {}
    for status in IngestStatus:
        status_counts[status] = 0
    
    # Calculate statistics
    total_jobs = 0
    active_jobs = 0
    recent_completed = 0
    recent_failed = 0
    total_chunks_processed = 0
    total_concepts_mapped = 0
    
    now = datetime.now()
    
    for job in jobs.values():
        total_jobs += 1
        status_counts[job.status] += 1
        
        if job.status in [IngestStatus.PROCESSING, IngestStatus.EXTRACTING, 
                          IngestStatus.CHUNKING, IngestStatus.VECTORIZING,
                          IngestStatus.CONCEPT_MAPPING, IngestStatus.STORING]:
            active_jobs += 1
        
        if job.completed_at and (now - job.completed_at).total_seconds() < 3600:  # Last hour
            if job.status == IngestStatus.COMPLETED:
                recent_completed += 1
            elif job.status == IngestStatus.FAILED:
                recent_failed += 1
        
        total_chunks_processed += job.chunks_processed
        total_concepts_mapped += job.concepts_mapped
    
    # Calculate processing rate (jobs per hour)
    processing_rate = recent_completed
    
    return {
        "total_jobs": total_jobs,
        "active_jobs": active_jobs,
        "queued_jobs": status_counts[IngestStatus.QUEUED],
        "completed_jobs": status_counts[IngestStatus.COMPLETED],
        "failed_jobs": status_counts[IngestStatus.FAILED],
        "status_counts": status_counts,
        "processing_rate_per_hour": processing_rate,
        "recent_completed": recent_completed,
        "recent_failed": recent_failed,
        "total_chunks_processed": total_chunks_processed,
        "total_concepts_mapped": total_concepts_mapped,
        "service_uptime_seconds": 0,  # Replace with actual uptime tracking
        "timestamp": now.isoformat()
    }

@router.get("/job/{job_id}", response_model=IngestJob, tags=["status"])
async def get_job_status(job_id: str):
    """
    Get the status of a specific job.
    
    Args:
        job_id: The ID of the job to get
        
    Returns:
        IngestJob: The job status
        
    Raises:
        HTTPException: If the job is not found
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return jobs[job_id]

@router.put("/job/{job_id}", response_model=IngestJob, tags=["status"])
async def update_job_status(job_id: str, update: IngestJobUpdate):
    """
    Update the status of a job.
    
    This endpoint is primarily used by workers to update job status.
    
    Args:
        job_id: The ID of the job to update
        update: The updates to apply to the job
        
    Returns:
        IngestJob: The updated job
        
    Raises:
        HTTPException: If the job is not found or cannot be updated
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    
    # Prevent updates to completed or failed jobs
    if job.status in [IngestStatus.COMPLETED, IngestStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update job {job_id} with status {job.status}"
        )
    
    # Apply updates
    updated = False
    
    if update.status is not None:
        job.status = update.status
        updated = True
    
    if update.percent_complete is not None:
        job.percent_complete = update.percent_complete
        updated = True
    
    if update.chunks_processed is not None:
        job.chunks_processed = update.chunks_processed
        updated = True
    
    if update.chunks_total is not None:
        job.chunks_total = update.chunks_total
        updated = True
    
    if update.concepts_mapped is not None:
        job.concepts_mapped = update.concepts_mapped
        updated = True
    
    if update.failure_code is not None:
        job.failure_code = update.failure_code
        updated = True
    
    if update.failure_message is not None:
        job.failure_message = update.failure_message
        updated = True
    
    if update.chunk_ids is not None:
        job.chunk_ids = update.chunk_ids
        updated = True
    
    if update.concept_ids is not None:
        job.concept_ids = update.concept_ids
        updated = True
    
    if updated:
        job.updated_at = datetime.now()
        
        # If job is marked as completed or failed, set completed_at
        if job.status in [IngestStatus.COMPLETED, IngestStatus.FAILED] and not job.completed_at:
            job.completed_at = datetime.now()
    
    return job

@router.get("/stats", tags=["status"])
async def get_processing_stats(
    timeframe: str = Query("all", enum=["hour", "day", "week", "month", "all"]),
):
    """
    Get processing statistics for jobs.
    
    Args:
        timeframe: The timeframe to get statistics for
        
    Returns:
        Dict: Processing statistics
    """
    # Determine cutoff time based on timeframe
    now = datetime.now()
    cutoff = None
    
    if timeframe == "hour":
        cutoff = now.replace(minute=0, second=0, microsecond=0)
    elif timeframe == "day":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeframe == "week":
        # Get the start of the current week (Monday)
        days_since_monday = now.weekday()
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=days_since_monday)
    elif timeframe == "month":
        cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # For "all", cutoff remains None
    
    # Initialize stats
    stats = {
        "total_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "avg_processing_time_seconds": 0,
        "total_chunks_processed": 0,
        "total_concepts_mapped": 0,
        "avg_chunks_per_job": 0,
        "avg_concepts_per_job": 0,
        "success_rate": 0,
        "timeframe": timeframe,
        "timestamp": now.isoformat()
    }
    
    # Process jobs
    completed_jobs = []
    processing_times = []
    
    for job in jobs.values():
        # Skip jobs outside the timeframe
        if cutoff and job.created_at < cutoff:
            continue
            
        stats["total_jobs"] += 1
        
        if job.status == IngestStatus.COMPLETED:
            stats["completed_jobs"] += 1
            completed_jobs.append(job)
            
            if job.completed_at:
                processing_time = (job.completed_at - job.created_at).total_seconds()
                processing_times.append(processing_time)
            
            stats["total_chunks_processed"] += job.chunks_processed
            stats["total_concepts_mapped"] += job.concepts_mapped
            
        elif job.status == IngestStatus.FAILED:
            stats["failed_jobs"] += 1
    
    # Calculate averages
    if processing_times:
        stats["avg_processing_time_seconds"] = sum(processing_times) / len(processing_times)
    
    if completed_jobs:
        stats["avg_chunks_per_job"] = stats["total_chunks_processed"] / len(completed_jobs)
        stats["avg_concepts_per_job"] = stats["total_concepts_mapped"] / len(completed_jobs)
    
    if stats["total_jobs"] > 0:
        stats["success_rate"] = stats["completed_jobs"] / stats["total_jobs"]
    
    return stats
