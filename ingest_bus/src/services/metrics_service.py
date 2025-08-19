"""
Prometheus metrics service for ingest-bus.

This module provides a service for exposing Prometheus metrics related to
document ingestion and processing.
"""

import time
from typing import Dict, Optional, Set, Any
import logging
from prometheus_client import Counter, Gauge, Histogram, Info

# Configure logging
logger = logging.getLogger(__name__)

# Define metrics
INGEST_FILES_QUEUED = Counter(
    'ingest_files_queued_total',
    'Total number of files queued for ingestion',
    ['track']
)

INGEST_FILES_PROCESSED = Counter(
    'ingest_files_processed_total',
    'Total number of files processed',
    ['status', 'track']
)

INGEST_CHUNKS_CREATED = Counter(
    'ingest_chunks_created_total',
    'Total number of chunks created from documents',
    ['track']
)

INGEST_CONCEPTS_CREATED = Counter(
    'ingest_concepts_created_total',
    'Total number of concepts created or linked',
    ['track']
)

INGEST_FAILURES = Counter(
    'ingest_failures_total',
    'Total number of ingest failures',
    ['stage', 'track']
)

INGEST_ACTIVE_JOBS = Gauge(
    'ingest_active_jobs',
    'Current number of active ingest jobs',
    ['status', 'track']
)

INGEST_JOB_DURATION = Histogram(
    'ingest_job_duration_seconds',
    'Duration of ingest jobs',
    ['status', 'track'],
    buckets=[30, 60, 120, 300, 600, 1800, 3600, 7200, 14400, 28800]
)

INGEST_FILE_SIZE = Histogram(
    'ingest_file_size_bytes',
    'Size of ingested files in bytes',
    ['track'],
    buckets=[1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024]
)

CHUNK_AVG_LENGTH = Gauge(
    'chunk_avg_len_chars',
    'Average length of chunks in characters'
)

DELTA_BYTES_SAVED = Counter(
    'delta_bytes_saved_total',
    'Total bytes saved by delta encoding'
)

DELTA_COMPRESSION_RATIO = Gauge(
    'delta_compression_ratio',
    'Compression ratio achieved by delta encoding'
)

INGEST_BUILD_INFO = Info(
    'ingest_build_info',
    'Build information for ingest-bus service'
)


class MetricsService:
    """
    Service for exposing Prometheus metrics

    Provides functionality to record and expose metrics related to the
    ingest process.
    """
    
    def __init__(self, build_hash: str = 'dev'):
        """
        Initialize the metrics service
        
        Args:
            build_hash: Git hash or build identifier
        """
        self.build_hash = build_hash
        self._set_build_info()
        
        # Track active jobs to update gauges properly
        self.active_jobs_by_status_track: Dict[str, Set[str]] = {}
        
        # Track job start times to calculate duration
        self.job_start_times: Dict[str, float] = {}
        
        logger.info("Initialized metrics service")
    
    def _set_build_info(self) -> None:
        """Set build information metrics"""
        INGEST_BUILD_INFO.info({
            'build_hash': self.build_hash,
            'version': '1.0.0',
            'timestamp': str(int(time.time()))
        })
    
    def record_job_queued(self, job_id: str, track: str) -> None:
        """
        Record a job being queued
        
        Args:
            job_id: Job ID
            track: Track the document belongs to
        """
        # Normalize track
        track = track or 'unknown'
        
        # Increment counter
        INGEST_FILES_QUEUED.labels(track=track).inc()
        
        # Update active jobs gauge
        self._update_active_jobs('queued', track, job_id, 1)
        
        # Record start time for duration calculation
        self.job_start_times[job_id] = time.time()
    
    def record_job_started(self, job_id: str, track: str) -> None:
        """
        Record a job being started
        
        Args:
            job_id: Job ID
            track: Track the document belongs to
        """
        # Normalize track
        track = track or 'unknown'
        
        # Update active jobs gauge (decrement queued, increment processing)
        self._update_active_jobs('queued', track, job_id, -1)
        self._update_active_jobs('processing', track, job_id, 1)
    
    def record_job_completed(self, 
                             job_id: str, 
                             track: str, 
                             status: str, 
                             file_size: Optional[int] = None,
                             chunk_count: int = 0,
                             concept_count: int = 0) -> None:
        """
        Record a job being completed
        
        Args:
            job_id: Job ID
            track: Track the document belongs to
            status: Completion status ('completed' or 'failed')
            file_size: Size of the file in bytes
            chunk_count: Number of chunks created
            concept_count: Number of concepts created or linked
        """
        # Normalize track and status
        track = track or 'unknown'
        status = status.lower()
        
        # Increment processed counter
        INGEST_FILES_PROCESSED.labels(status=status, track=track).inc()
        
        # Update active jobs gauge (decrement processing)
        self._update_active_jobs('processing', track, job_id, -1)
        
        # Record chunks and concepts
        if chunk_count > 0:
            INGEST_CHUNKS_CREATED.labels(track=track).inc(chunk_count)
        
        if concept_count > 0:
            INGEST_CONCEPTS_CREATED.labels(track=track).inc(concept_count)
        
        # Record file size if available
        if file_size and file_size > 0:
            INGEST_FILE_SIZE.labels(track=track).observe(file_size)
        
        # Record job duration
        if job_id in self.job_start_times:
            duration = time.time() - self.job_start_times[job_id]
            INGEST_JOB_DURATION.labels(status=status, track=track).observe(duration)
            del self.job_start_times[job_id]
    
    def record_failure(self, stage: str, track: str) -> None:
        """
        Record a processing failure
        
        Args:
            stage: Processing stage where failure occurred
            track: Track the document belongs to
        """
        # Normalize track
        track = track or 'unknown'
        
        # Increment failures counter
        INGEST_FAILURES.labels(stage=stage, track=track).inc()
    
    def update_chunk_avg_length(self, avg_length: float) -> None:
        """
        Update average chunk length metric
        
        Args:
            avg_length: Average length of chunks in characters
        """
        CHUNK_AVG_LENGTH.set(avg_length)
    
    def record_delta_bytes_saved(self, bytes_saved: int, compression_ratio: float) -> None:
        """
        Record bytes saved by delta encoding
        
        Args:
            bytes_saved: Number of bytes saved
            compression_ratio: Compression ratio achieved
        """
        DELTA_BYTES_SAVED.inc(bytes_saved)
        DELTA_COMPRESSION_RATIO.set(compression_ratio)
    
    def _update_active_jobs(self, status: str, track: str, job_id: str, change: int) -> None:
        """
        Update active jobs gauge
        
        Args:
            status: Job status
            track: Document track
            job_id: Job ID
            change: Amount to change by (+1 or -1)
        """
        key = f"{status}:{track}"
        
        if key not in self.active_jobs_by_status_track:
            self.active_jobs_by_status_track[key] = set()
        
        if change > 0:
            self.active_jobs_by_status_track[key].add(job_id)
        elif change < 0 and job_id in self.active_jobs_by_status_track[key]:
            self.active_jobs_by_status_track[key].remove(job_id)
        
        # Update gauge value
        INGEST_ACTIVE_JOBS.labels(status=status, track=track).set(
            len(self.active_jobs_by_status_track[key])
        )


# Create singleton instance
_metrics_service = None


def get_metrics_service(build_hash: str = 'dev') -> MetricsService:
    """
    Get or create the metrics service
    
    Args:
        build_hash: Git hash or build identifier
        
    Returns:
        The metrics service singleton
    """
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService(build_hash)
    return _metrics_service


def on_delta_metrics(metrics: Dict[str, Any]) -> None:
    """
    Callback for delta encoder metrics
    
    Args:
        metrics: Metrics data from delta encoder
    """
    ratio = metrics.get('deltaFullRatio', 1.0)
    if ratio < 1.0:
        full_size = metrics.get('fullStateSize', 0)
        delta_size = metrics.get('deltaSize', 0)
        bytes_saved = full_size - delta_size
        
        if bytes_saved > 0:
            metrics_service = get_metrics_service()
            metrics_service.record_delta_bytes_saved(bytes_saved, ratio)
