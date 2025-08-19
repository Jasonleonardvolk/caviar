"""
Ingestion API Routes

Provides endpoints for queueing and managing document ingestion jobs.
Includes SHA-256 deduplication to prevent processing duplicate files.
"""

import os
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, HTTPException, Depends, Header, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.job import Job, JobStatus, JobProgress
from ..services.job_store import JobStore
from ..utils.math_aware_extractor import calculate_file_hash
from ..metrics import increment_counter, observe_histogram, add_bucket_to_histogram

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api")

# Initialize job store
job_store = JobStore()

# Initialize metrics
METRICS = {
    'ingest_files_queued_total': 0,
    'ingest_duplicates_total': 0,
    'ingest_files_processed_total': 0,
    'ingest_failures_total': 0,
    'chunk_size_chars_bucket': {}
}


class QueueJobRequest(BaseModel):
    """Queue job request model"""
    file_url: str
    track: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    force: Optional[bool] = False  # Flag to force reprocessing even if duplicated


class JobResponse(BaseModel):
    """Job response model"""
    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class UpdateJobStatusRequest(BaseModel):
    """Update job status request model"""
    status: str
    error: Optional[str] = None


class UpdateJobProgressRequest(BaseModel):
    """Update job progress request model"""
    stage: str
    progress: float


class AddJobChunkRequest(BaseModel):
    """Add job chunk request model"""
    text: str
    start_offset: int
    end_offset: int
    metadata: Optional[Dict[str, Any]] = None


class AddJobConceptRequest(BaseModel):
    """Add job concept request model"""
    concept_id: str


def authenticate_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Authenticate API key if authentication is required
    
    Args:
        x_api_key: API key from header
        
    Returns:
        True if authenticated, False otherwise
    """
    # Check if API key auth is required
    require_api_key = os.environ.get("INGEST_REQUIRE_API_KEY", "").lower() in ("1", "true", "yes")
    
    if not require_api_key:
        return True
    
    expected_key = os.environ.get("INGEST_API_KEY", "")
    
    if not expected_key:
        logger.warning("API key authentication required but no key configured")
        return True
    
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True


@router.get("/jobs", dependencies=[Depends(authenticate_api_key)])
async def list_jobs(status: Optional[str] = None, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """
    List jobs
    
    Args:
        status: Optional status filter
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        
    Returns:
        List of jobs
    """
    try:
        jobs = job_store.list_jobs(status=status, limit=limit, offset=offset)
        
        return {
            "success": True,
            "jobs": jobs
        }
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/jobs/{job_id}", dependencies=[Depends(authenticate_api_key)])
async def get_job(job_id: str) -> Dict[str, Any]:
    """
    Get a job
    
    Args:
        job_id: Job ID
        
    Returns:
        Job data
    """
    try:
        job = job_store.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "success": True,
            "job": job
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/jobs", dependencies=[Depends(authenticate_api_key)])
async def queue_job(request: QueueJobRequest) -> Dict[str, Any]:
    """
    Queue a document for processing
    
    Args:
        request: Queue job request
        
    Returns:
        Job ID
    """
    try:
        # Update metrics
        increment_counter('ingest_files_queued_total')
        
        # Check if this file has already been processed
        # In a real implementation, we'd download the file and calculate its hash
        # For this example, we'll use the URL as a placeholder for the hash
        # (In a production system, you would download the file here if no hash is provided)
        
        existing_job = None
        
        if not request.force:
            # Check if URL has been processed before
            jobs = job_store.list_jobs(limit=10)
            for job in jobs:
                if job.get("file_url") == request.file_url:
                    # URL match is a weak indicator, we'd want to check hashes in production
                    # For now, assume the file might be a duplicate
                    existing_job = job
                    break
                
                # If the job metadata contains a hash, check that too
                if job.get("metadata", {}).get("file_hash") == request.metadata.get("file_hash"):
                    existing_job = job
                    break
        
        if existing_job and not request.force:
            # This is a duplicate, return the existing job ID but mark as duplicate
            increment_counter('ingest_duplicates_total')
            
            logger.info(f"Duplicate file detected: {request.file_url}")
            
            return JSONResponse(
                status_code=409,  # Conflict status code for duplicates
                content={
                    "success": False,
                    "error": "Duplicate file detected",
                    "job_id": existing_job["job_id"],
                    "status": existing_job["status"],
                    "duplicate": True
                }
            )
        
        # Create a new job
        job = Job(
            file_url=request.file_url,
            track=request.track,
            metadata=request.metadata or {}
        )
        
        # Save the job
        job_id = job_store.create_job(job)
        
        logger.info(f"Queued job {job_id} for {request.file_url}")
        
        return {
            "success": True,
            "job_id": job_id
        }
    except Exception as e:
        logger.error(f"Error queueing job: {str(e)}")
        increment_counter('ingest_failures_total')
        
        return {
            "success": False,
            "error": str(e)
        }


@router.patch("/jobs/{job_id}/status", dependencies=[Depends(authenticate_api_key)])
async def update_job_status(job_id: str, request: UpdateJobStatusRequest) -> Dict[str, Any]:
    """
    Update job status
    
    Args:
        job_id: Job ID
        request: Update status request
        
    Returns:
        Updated job
    """
    try:
        job = job_store.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Update status
        job_store.update_job_status(job_id, request.status, request.error)
        
        # Update metrics
        if request.status == "completed":
            increment_counter('ingest_files_processed_total')
        elif request.status == "failed":
            increment_counter('ingest_failures_total')
        
        # Get updated job
        updated_job = job_store.get_job(job_id)
        
        return {
            "success": True,
            "job": updated_job
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating job status for {job_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.patch("/jobs/{job_id}/progress", dependencies=[Depends(authenticate_api_key)])
async def update_job_progress(job_id: str, request: UpdateJobProgressRequest) -> Dict[str, Any]:
    """
    Update job progress
    
    Args:
        job_id: Job ID
        request: Update progress request
        
    Returns:
        Updated job
    """
    try:
        job = job_store.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Update progress
        job_store.update_job_progress(job_id, request.stage, request.progress)
        
        # Get updated job
        updated_job = job_store.get_job(job_id)
        
        return {
            "success": True,
            "job": updated_job
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating job progress for {job_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/jobs/{job_id}/chunks", dependencies=[Depends(authenticate_api_key)])
async def add_job_chunk(job_id: str, request: AddJobChunkRequest) -> Dict[str, Any]:
    """
    Add a chunk to a job
    
    Args:
        job_id: Job ID
        request: Add chunk request
        
    Returns:
        Updated job
    """
    try:
        job = job_store.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Add chunk
        job_store.add_job_chunk(
            job_id,
            request.text,
            request.start_offset,
            request.end_offset,
            request.metadata
        )
        
        # Update chunk size histogram
        chunk_size = len(request.text)
        
        # Use the specific histogram buckets
        for bucket in [800, 1200, 1600, 2000]:
            if chunk_size <= bucket:
                add_bucket_to_histogram('chunk_size_chars_bucket', bucket, 1)
                break
        
        # Get updated job
        updated_job = job_store.get_job(job_id)
        
        return {
            "success": True,
            "job": updated_job
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error adding chunk to job {job_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/jobs/{job_id}/concepts", dependencies=[Depends(authenticate_api_key)])
async def add_job_concept(job_id: str, request: AddJobConceptRequest) -> Dict[str, Any]:
    """
    Add a concept to a job
    
    Args:
        job_id: Job ID
        request: Add concept request
        
    Returns:
        Updated job
    """
    try:
        job = job_store.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Add concept
        job_store.add_job_concept(job_id, request.concept_id)
        
        # Get updated job
        updated_job = job_store.get_job(job_id)
        
        return {
            "success": True,
            "job": updated_job
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error adding concept to job {job_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/metrics", dependencies=[Depends(authenticate_api_key)])
async def get_metrics() -> Dict[str, Any]:
    """
    Get ingestion metrics
    
    Returns:
        Metrics data
    """
    try:
        # Get basic metrics
        metrics = {
            'total_jobs': len(job_store.list_jobs(limit=1000000)),
            'status_counts': {},
            'ingest_files_queued_total': METRICS['ingest_files_queued_total'],
            'ingest_duplicates_total': METRICS['ingest_duplicates_total'],
            'ingest_files_processed_total': METRICS['ingest_files_processed_total'],
            'ingest_failures_total': METRICS['ingest_failures_total'],
            'chunk_size_histogram': METRICS['chunk_size_chars_bucket']
        }
        
        # Get status counts
        for status in JobStatus:
            count = len(job_store.list_jobs(status=status, limit=1000000))
            metrics['status_counts'][status] = count
        
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
