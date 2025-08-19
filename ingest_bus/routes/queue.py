"""
Ingest queue route module for TORI Ingest Bus.

This module provides the API endpoints for adding documents to the ingest queue,
listing current jobs, and managing the queue.
"""

import os
import uuid
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Query, Body, Form, Depends
from fastapi.responses import JSONResponse

from models.schemas import (
    IngestStatus, DocumentType, FailureCode, Chunk,
    ConceptVectorLink, IngestRequest, IngestJob, IngestJobUpdate
)
from workers import extract

# Set up logging
logger = logging.getLogger("ingest-bus.queue")

# Initialize router
router = APIRouter()

# In-memory job store (would be replaced with a file_storage in production)
jobs: Dict[str, IngestJob] = {}

# Queue processing
async def process_ingest_job(job_id: str, file_path: Optional[str] = None, file_content: Optional[bytes] = None):
    """
    Process an ingest job in the background.
    
    Args:
        job_id: The ID of the job to process
        file_path: Optional path to the file to process
        file_content: Optional raw content of the file to process
    """
    try:
        if job_id not in jobs:
            logger.error(f"Job {job_id} not found in job store")
            return
        
        job = jobs[job_id]
        job.status = IngestStatus.PROCESSING
        job.updated_at = datetime.now()
        job.percent_complete = 5.0
        
        logger.info(f"Processing job {job_id} of type {job.request.document_type}")
        
        # Extraction phase
        job.status = IngestStatus.EXTRACTING
        job.updated_at = datetime.now()
        job.percent_complete = 10.0
        
        # Call the appropriate extractor based on document type
        if job.request.document_type == DocumentType.PDF:
            result = await extract.extract_pdf(file_path, file_content, job)
        elif job.request.document_type == DocumentType.CONVERSATION:
            result = await extract.extract_conversation(file_path, file_content, job)
        else:
            # Default extraction for other document types
            result = await extract.extract_text(file_path, file_content, job)
        
        if not result:
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.EXTRACTION_ERROR
            job.failure_message = "Failed to extract content from document"
            job.updated_at = datetime.now()
            logger.error(f"Extraction failed for job {job_id}")
            return
        
        # Chunking phase
        job.status = IngestStatus.CHUNKING
        job.updated_at = datetime.now()
        job.percent_complete = 30.0
        
        chunks = await extract.chunk_content(result, job)
        if not chunks:
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.CHUNKING_ERROR
            job.failure_message = "Failed to chunk content"
            job.updated_at = datetime.now()
            logger.error(f"Chunking failed for job {job_id}")
            return
        
        job.chunks_total = len(chunks)
        job.chunks_processed = 0
        
        # Vectorization phase
        job.status = IngestStatus.VECTORIZING
        job.updated_at = datetime.now()
        job.percent_complete = 50.0
        
        vectors = await extract.vectorize_chunks(chunks, job)
        if not vectors:
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.VECTORIZATION_ERROR
            job.failure_message = "Failed to vectorize chunks"
            job.updated_at = datetime.now()
            logger.error(f"Vectorization failed for job {job_id}")
            return
        
        # Concept mapping phase
        job.status = IngestStatus.CONCEPT_MAPPING
        job.updated_at = datetime.now()
        job.percent_complete = 70.0
        
        concept_links = await extract.map_to_concepts(vectors, job)
        if not concept_links:
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.CONCEPT_MAPPING_ERROR
            job.failure_message = "Failed to map to concepts"
            job.updated_at = datetime.now()
            logger.error(f"Concept mapping failed for job {job_id}")
            return
        
        job.concepts_mapped = len(concept_links)
        
        # Store the results
        job.status = IngestStatus.STORING
        job.updated_at = datetime.now()
        job.percent_complete = 90.0
        
        store_result = await extract.store_results(chunks, vectors, concept_links, job)
        if not store_result:
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.STORAGE_ERROR
            job.failure_message = "Failed to store results"
            job.updated_at = datetime.now()
            logger.error(f"Storage failed for job {job_id}")
            return
        
        # Update job with concept IDs
        job.chunk_ids = [chunk.id for chunk in chunks]
        job.concept_ids = list(set(link.concept_id for link in concept_links))
        
        # Job completed successfully
        job.status = IngestStatus.COMPLETED
        job.updated_at = datetime.now()
        job.completed_at = datetime.now()
        job.percent_complete = 100.0
        job.chunks_processed = len(chunks)
        
        logger.info(f"Job {job_id} completed successfully. Processed {len(chunks)} chunks and mapped {len(concept_links)} concepts.")
        
        # Trigger callback if provided
        if job.request.callback_url:
            await extract.send_callback(job)
            
    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {str(e)}")
        if job_id in jobs:
            job = jobs[job_id]
            job.status = IngestStatus.FAILED
            job.failure_code = FailureCode.UNKNOWN
            job.failure_message = f"Unexpected error: {str(e)}"
            job.updated_at = datetime.now()

@router.post("/", response_model=IngestJob, status_code=202, tags=["ingest"])
async def queue_ingest(
    background_tasks: BackgroundTasks,
    request: IngestRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Queue a document for ingestion.
    
    This endpoint accepts a document and adds it to the ingest queue for processing.
    The document can be provided as a file upload or as a reference to an existing file.
    
    Returns:
        IngestJob: The created ingest job
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create the job
    job = IngestJob(
        id=job_id,
        request=request,
        status=IngestStatus.QUEUED,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Store the job
    jobs[job_id] = job
    
    # Process the file if provided
    file_path = None
    file_content = None
    
    if file:
        # Create a temporary file to store the uploaded content
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{job_id}_{file.filename}"
        file_content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(file_content)
    
    # Start background processing
    background_tasks.add_task(
        process_ingest_job,
        job_id=job_id,
        file_path=str(file_path) if file_path else None,
        file_content=file_content
    )
    
    logger.info(f"Queued job {job_id} of type {request.document_type}")
    return job

@router.get("/{job_id}", response_model=IngestJob, tags=["ingest"])
async def get_job(job_id: str):
    """
    Get the status of an ingest job.
    
    Args:
        job_id: The ID of the job to get
        
    Returns:
        IngestJob: The requested job
        
    Raises:
        HTTPException: If the job is not found
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return jobs[job_id]

@router.get("/", response_model=List[IngestJob], tags=["ingest"])
async def list_jobs(
    status: Optional[IngestStatus] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List ingest jobs.
    
    Args:
        status: Filter by job status
        limit: Maximum number of jobs to return
        offset: Offset for pagination
        
    Returns:
        List[IngestJob]: List of jobs matching the criteria
    """
    result = []
    
    for job in jobs.values():
        if status is None or job.status == status:
            result.append(job)
    
    # Sort by creation time (newest first)
    result.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    return result[offset:offset + limit]

@router.delete("/{job_id}", status_code=204, tags=["ingest"])
async def cancel_job(job_id: str):
    """
    Cancel an ingest job.
    
    Args:
        job_id: The ID of the job to cancel
        
    Raises:
        HTTPException: If the job is not found or cannot be canceled
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    
    # Only queued or processing jobs can be canceled
    if job.status not in [IngestStatus.QUEUED, IngestStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be canceled (status: {job.status})"
        )
    
    # Mark the job as failed
    job.status = IngestStatus.FAILED
    job.failure_code = FailureCode.UNKNOWN
    job.failure_message = "Job canceled by user"
    job.updated_at = datetime.now()
    
    logger.info(f"Job {job_id} canceled by user")
    return None
