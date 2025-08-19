"""
Metrics route module for TORI Ingest Bus.

This module provides metrics endpoints for monitoring the ingest service.
It exposes Prometheus-compatible metrics for integration with monitoring systems
and Kaizen alerts.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Response
from prometheus_client import (
    Counter, Gauge, Histogram, REGISTRY,
    generate_latest, CONTENT_TYPE_LATEST
)

from models.schemas import IngestStatus, FailureCode, MetricsResponse
from routes.queue import jobs

# Set up logging
logger = logging.getLogger("ingest-bus.metrics")

# Initialize router
router = APIRouter()

# Define Prometheus metrics
ingest_files_queued_total = Counter(
    'ingest_files_queued_total',
    'Total number of files queued for ingestion',
    ['document_type']
)

ingest_files_processed_total = Counter(
    'ingest_files_processed_total',
    'Total number of files successfully processed',
    ['document_type']
)

ingest_failures_total = Counter(
    'ingest_failures_total',
    'Total number of ingest failures',
    ['document_type', 'failure_code']
)

chunk_count_total = Counter(
    'chunk_count_total',
    'Total number of chunks created'
)

concept_mapping_total = Counter(
    'concept_mapping_total',
    'Total number of concept mappings created'
)

# Gauge metrics
chunk_avg_len_chars = Gauge(
    'chunk_avg_len_chars',
    'Average length of chunks in characters'
)

concept_recall_accuracy = Gauge(
    'concept_recall_accuracy',
    'Accuracy of concept recall (0-1)'
)

active_jobs = Gauge(
    'active_jobs',
    'Number of active ingest jobs'
)

queue_depth = Gauge(
    'queue_depth',
    'Number of jobs waiting in the queue'
)

# Histogram for processing time
processing_time_seconds = Histogram(
    'processing_time_seconds',
    'Time to process an ingest job',
    ['document_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
)

@router.get("/prometheus", tags=["metrics"])
async def get_prometheus_metrics():
    """
    Get Prometheus metrics.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    # Update gauge metrics before returning
    update_gauge_metrics()
    
    # Generate and return metrics
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/", response_model=MetricsResponse, tags=["metrics"])
async def get_ingest_metrics():
    """
    Get ingest metrics in JSON format.
    
    Returns:
        MetricsResponse: Current ingest metrics
    """
    # Update gauge metrics
    update_gauge_metrics()
    
    # Calculate metrics for JSON response
    now = datetime.now()
    
    # Count jobs by status
    active_job_count = 0
    queued_job_count = 0
    completed_job_count = 0
    failed_job_count = 0
    chunks_processed = 0
    concepts_mapped = 0
    failure_by_code = {}
    
    for job in jobs.values():
        if job.status in [IngestStatus.PROCESSING, IngestStatus.EXTRACTING, 
                          IngestStatus.CHUNKING, IngestStatus.VECTORIZING,
                          IngestStatus.CONCEPT_MAPPING, IngestStatus.STORING]:
            active_job_count += 1
        elif job.status == IngestStatus.QUEUED:
            queued_job_count += 1
        elif job.status == IngestStatus.COMPLETED:
            completed_job_count += 1
            chunks_processed += job.chunks_processed
            concepts_mapped += job.concepts_mapped
        elif job.status == IngestStatus.FAILED:
            failed_job_count += 1
            
            if job.failure_code:
                code_str = job.failure_code.value
                failure_by_code[code_str] = failure_by_code.get(code_str, 0) + 1
    
    # Calculate average processing time
    processing_times = []
    for job in jobs.values():
        if job.status == IngestStatus.COMPLETED and job.completed_at:
            processing_time = (job.completed_at - job.created_at).total_seconds() * 1000  # Convert to ms
            processing_times.append(processing_time)
    
    avg_processing_time = 0
    if processing_times:
        avg_processing_time = sum(processing_times) / len(processing_times)
    
    # Calculate average chunk length (from prometheus gauge)
    avg_chunk_len = chunk_avg_len_chars._value.get()
    
    # Create and return metrics response
    return MetricsResponse(
        timestamp=now,
        ingest_files_queued_total=queued_job_count,
        ingest_files_processed_total=completed_job_count,
        ingest_failures_total=failed_job_count,
        chunk_avg_len_chars=avg_chunk_len,
        concept_recall_accuracy=concept_recall_accuracy._value.get(),
        active_jobs=active_job_count,
        queue_depth=queued_job_count,
        processing_time_avg_ms=avg_processing_time,
        failure_by_code=failure_by_code
    )

def update_gauge_metrics():
    """Update values for gauge metrics based on current system state."""
    # Count active and queued jobs
    active_job_count = 0
    queued_job_count = 0
    completed_chunks = 0
    total_characters = 0
    
    for job in jobs.values():
        if job.status in [IngestStatus.PROCESSING, IngestStatus.EXTRACTING, 
                          IngestStatus.CHUNKING, IngestStatus.VECTORIZING,
                          IngestStatus.CONCEPT_MAPPING, IngestStatus.STORING]:
            active_job_count += 1
        elif job.status == IngestStatus.QUEUED:
            queued_job_count += 1
        
        # For completed jobs, collect chunk statistics
        if job.status == IngestStatus.COMPLETED:
            completed_chunks += job.chunks_processed
            
            # We don't have actual chunk lengths in this simple example
            # In a real implementation, we would use actual chunk data
            # This is a placeholder calculation
            total_characters += job.chunks_processed * 500  # Assuming average 500 chars per chunk
    
    # Update gauges
    active_jobs.set(active_job_count)
    queue_depth.set(queued_job_count)
    
    # Update average chunk length if there are chunks
    if completed_chunks > 0:
        chunk_avg_len_chars.set(total_characters / completed_chunks)
    
    # For concept recall accuracy, we would need a way to measure this
    # This is a placeholder that sets it to a fixed value
    # In a real implementation, this would be measured from actual recall tests
    concept_recall_accuracy.set(0.85)  # Example fixed value

@router.post("/increment/{metric_name}", tags=["metrics"])
async def increment_metric(
    metric_name: str,
    value: float = 1.0,
    labels: Dict[str, str] = None
):
    """
    Increment a specific metric.
    
    This endpoint is useful for external services to increment metrics.
    
    Args:
        metric_name: Name of the metric to increment
        value: Value to increment by
        labels: Dictionary of label values to use
        
    Returns:
        Dict: Confirmation of increment
    """
    if labels is None:
        labels = {}
    
    # Map metric name to actual metric
    metric_map = {
        "ingest_files_queued": ingest_files_queued_total,
        "ingest_files_processed": ingest_files_processed_total,
        "ingest_failures": ingest_failures_total,
        "chunk_count": chunk_count_total,
        "concept_mapping": concept_mapping_total
    }
    
    if metric_name not in metric_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric: {metric_name}"
        )
    
    metric = metric_map[metric_name]
    
    try:
        if hasattr(metric, 'labels'):
            # Only use labels that are valid for this metric
            valid_labels = {}
            for label_name in labels:
                if label_name in metric._labelnames:
                    valid_labels[label_name] = labels[label_name]
            
            # Fill in missing labels with defaults
            for label_name in metric._labelnames:
                if label_name not in valid_labels:
                    valid_labels[label_name] = "unknown"
            
            # Increment with labels
            metric.labels(**valid_labels).inc(value)
        else:
            # Increment without labels
            metric.inc(value)
        
        logger.info(f"Incremented metric {metric_name} by {value} with labels {labels}")
        return {
            "status": "success",
            "metric": metric_name,
            "value": value,
            "labels": labels
        }
        
    except Exception as e:
        logger.exception(f"Error incrementing metric {metric_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error incrementing metric: {str(e)}"
        )

@router.post("/set/{metric_name}", tags=["metrics"])
async def set_metric(
    metric_name: str,
    value: float,
    labels: Dict[str, str] = None
):
    """
    Set a specific gauge metric to a value.
    
    This endpoint is useful for external services to set gauge metrics.
    
    Args:
        metric_name: Name of the metric to set
        value: Value to set
        labels: Dictionary of label values to use
        
    Returns:
        Dict: Confirmation of setting
    """
    if labels is None:
        labels = {}
    
    # Map metric name to actual metric
    metric_map = {
        "chunk_avg_len": chunk_avg_len_chars,
        "concept_recall_accuracy": concept_recall_accuracy,
        "active_jobs": active_jobs,
        "queue_depth": queue_depth
    }
    
    if metric_name not in metric_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric: {metric_name}"
        )
    
    metric = metric_map[metric_name]
    
    try:
        if hasattr(metric, 'labels'):
            # Only use labels that are valid for this metric
            valid_labels = {}
            for label_name in labels:
                if label_name in metric._labelnames:
                    valid_labels[label_name] = labels[label_name]
            
            # Fill in missing labels with defaults
            for label_name in metric._labelnames:
                if label_name not in valid_labels:
                    valid_labels[label_name] = "unknown"
            
            # Set with labels
            metric.labels(**valid_labels).set(value)
        else:
            # Set without labels
            metric.set(value)
        
        logger.info(f"Set metric {metric_name} to {value} with labels {labels}")
        return {
            "status": "success",
            "metric": metric_name,
            "value": value,
            "labels": labels
        }
        
    except Exception as e:
        logger.exception(f"Error setting metric {metric_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error setting metric: {str(e)}"
        )
