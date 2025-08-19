"""
Worker Extract Example

This script demonstrates how to implement a worker for processing
documents queued in the ingest-bus service.

Requirements:
- requests
- magic (python-magic)
- tika-python
- mathpix-markdown-it
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
import pathlib
import asyncio
import magic
import subprocess
from typing import Dict, Any, List, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ingest Bus client configuration
INGEST_BUS_URL = os.environ.get("INGEST_BUS_URL", "http://localhost:8000")
INGEST_API_KEY = os.environ.get("INGEST_API_KEY", "")
WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")
MAX_CHUNK_SIZE = int(os.environ.get("MAX_CHUNK_SIZE", "1800"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))


class IngestClient:
    """Client for the Ingest Bus API"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the Ingest Bus API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.headers = {}
        
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def get_queued_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get queued jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of queued jobs
        """
        url = f"{self.base_url}/api/jobs?limit={limit}&status=queued"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get("jobs", [])
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get a job by ID
        
        Args:
            job_id: Job ID
            
        Returns:
            Job data
        """
        url = f"{self.base_url}/api/jobs/{job_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get("job", {})
    
    def update_job_status(self, 
                         job_id: str, 
                         status: str, 
                         error: Optional[str] = None) -> Dict[str, Any]:
        """
        Update job status
        
        Args:
            job_id: Job ID
            status: New status (processing, completed, failed)
            error: Optional error message
            
        Returns:
            Updated job data
        """
        url = f"{self.base_url}/api/jobs/{job_id}/status"
        
        data = {
            "status": status
        }
        
        if error:
            data["error"] = error
        
        response = requests.patch(url, json=data, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def update_job_progress(self, 
                          job_id: str, 
                          stage: str, 
                          progress: float) -> Dict[str, Any]:
        """
        Update job progress
        
        Args:
            job_id: Job ID
            stage: Processing stage
            progress: Progress percentage (0-100)
            
        Returns:
            Updated job data
        """
        url = f"{self.base_url}/api/jobs/{job_id}/progress"
        
        data = {
            "stage": stage,
            "progress": progress
        }
        
        response = requests.patch(url, json=data, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def add_job_chunk(self, 
                    job_id: str, 
                    chunk_text: str, 
                    start_offset: int, 
                    end_offset: int,
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a chunk to a job
        
        Args:
            job_id: Job ID
            chunk_text: Chunk text
            start_offset: Start offset in the document
            end_offset: End offset in the document
            metadata: Optional metadata
            
        Returns:
            Updated job data
        """
        url = f"{self.base_url}/api/jobs/{job_id}/chunks"
        
        data = {
            "text": chunk_text,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "metadata": metadata or {}
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def add_job_concept(self, 
                      job_id: str, 
                      concept_id: str) -> Dict[str, Any]:
        """
        Add a concept to a job
        
        Args:
            job_id: Job ID
            concept_id: Concept ID
            
        Returns:
            Updated job data
        """
        url = f"{self.base_url}/api/jobs/{job_id}/concepts"
        
        data = {
            "concept_id": concept_id
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def queue_job(self, 
                 file_url: str, 
                 track: Optional[str] = None, 
                 metadata: Dict[str, Any] = None) -> str:
        """
        Queue a job
        
        Args:
            file_url: URL of the file to process
            track: Optional track to assign
            metadata: Optional metadata
            
        Returns:
            Job ID
        """
        url = f"{self.base_url}/api/jobs"
        
        data = {
            "file_url": file_url
        }
        
        if track:
            data["track"] = track
        
        if metadata:
            data["metadata"] = metadata
        
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        return result.get("job_id")


def download_file(url: str, output_dir: str) -> pathlib.Path:
    """
    Download a file from URL
    
    Args:
        url: URL to download
        output_dir: Directory to save to
        
    Returns:
        Path to downloaded file
    """
    # Create output directory if it doesn't exist
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse URL to get filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    if not filename:
        filename = f"document_{int(time.time())}.bin"
    
    # Create output file path
    file_path = output_path / filename
    
    # Download file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {url} to {file_path}")
    return file_path


def pdf_to_markdown(pdf_path: pathlib.Path) -> pathlib.Path:
    """
    Convert PDF to Markdown with math support
    
    This is a placeholder implementation. In a real system, you would use
    something like Apache Tika + MathPix to extract text and formulas.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Path to Markdown file
    """
    # This is a placeholder for a real implementation
    # In reality, this would call Tika and MathPix for text and formula extraction
    
    md_path = pdf_path.with_suffix('.md')
    
    # Simulate conversion with a simple command
    # In production, replace with actual conversion logic:
    # - Use Apache Tika for text extraction
    # - Use MathPix for formula recognition
    # - Combine results into markdown
    logger.info(f"Converting {pdf_path} to {md_path}")
    
    with open(md_path, 'w') as f:
        f.write(f"# Extracted from {pdf_path.name}\n\n")
        f.write("This is a placeholder for the extracted text.\n\n")
        f.write("Example math formula: $E = mc^2$\n\n")
    
    return md_path


def chunk_file(md_path: pathlib.Path, max_len: int = 1800) -> Iterator[Dict[str, Any]]:
    """
    Split file into semantic chunks
    
    This is a simple implementation that splits by paragraphs and tries to
    keep chunks within the maximum length. A production implementation should
    use more sophisticated techniques to split on semantic boundaries.
    
    Args:
        md_path: Path to Markdown file
        max_len: Maximum chunk length in characters
        
    Yields:
        Dictionaries with chunk information
    """
    with open(md_path, 'r') as f:
        content = f.read()
    
    # Split by double newlines (paragraphs)
    paragraphs = content.split('\n\n')
    
    chunk_text = ""
    start_offset = 0
    end_offset = 0
    
    # Track position in the source text
    pos = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            pos += len(paragraph) + 2  # +2 for the newlines
            continue
        
        paragraph_len = len(paragraph)
        
        # If adding this paragraph would exceed max_len, yield the current chunk
        if len(chunk_text) > 0 and len(chunk_text) + paragraph_len > max_len:
            yield {
                "text": chunk_text,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "metadata": {
                    "source_file": str(md_path),
                    "chunk_size": len(chunk_text)
                }
            }
            
            # Start a new chunk
            chunk_text = paragraph
            start_offset = pos
            end_offset = pos + paragraph_len
        else:
            # Add to the current chunk
            if chunk_text:
                chunk_text += "\n\n" + paragraph
                end_offset = pos + paragraph_len
            else:
                chunk_text = paragraph
                start_offset = pos
                end_offset = pos + paragraph_len
        
        pos += paragraph_len + 2  # +2 for the newlines
    
    # Yield the last chunk if there is one
    if chunk_text:
        yield {
            "text": chunk_text,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "metadata": {
                "source_file": str(md_path),
                "chunk_size": len(chunk_text)
            }
        }


def embed(text: str) -> tuple:
    """
    Create embeddings for text
    
    This is a placeholder. In a real implementation, you would use a
    vector embedding model like OpenAI, Sentence Transformers, etc.
    
    Args:
        text: Text to embed
        
    Returns:
        Tuple of (concept_id, phase_vector)
    """
    # This is a placeholder for a real embedding implementation
    # In reality, this would:
    # 1. Call an embedding service/model
    # 2. Generate a concept ID
    # 3. Return the ID and embedding vector
    
    # Generate a simple hash-based ID
    concept_id = f"concept-{hash(text) % 10000000:07d}"
    
    # Simulate a phase vector (in production, this would be from an embedding model)
    import hashlib
    import struct
    
    # Create a deterministic but fake embedding vector
    md5 = hashlib.md5(text.encode()).digest()
    phase_vec = [struct.unpack('f', md5[i:i+4])[0] for i in range(0, len(md5), 4)]
    
    logger.debug(f"Generated embedding for text ({len(text)} chars)")
    
    return concept_id, phase_vec


def push_concepts(chunks: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> None:
    """
    Push concepts to the knowledge graph
    
    This is a placeholder. In a real implementation, you would:
    1. Connect to your knowledge graph system
    2. Insert the concepts with their embeddings
    3. Link concepts according to their relationships
    
    Args:
        chunks: List of chunks with concept IDs and embeddings
        metadata: Optional metadata for all concepts
    """
    # This is a placeholder for a real knowledge graph integration
    # In reality, this would:
    # 1. Connect to a graph file_storage or vector store
    # 2. Insert the concepts with their embeddings
    # 3. Create relationships between concepts
    
    logger.info(f"Pushing {len(chunks)} concepts to knowledge graph")
    
    # Log some details about the concepts
    for i, chunk in enumerate(chunks[:3]):  # Just show the first 3
        logger.debug(f"Concept {i}: {chunk['concept_id'][:10]}... ({len(chunk['text'])} chars)")
    
    if len(chunks) > 3:
        logger.debug(f"... and {len(chunks) - 3} more")


def process_file(file_path: pathlib.Path, track: str, client: IngestClient) -> Optional[str]:
    """
    Process a file
    
    Args:
        file_path: Path to the file
        track: Track to assign
        client: Ingest client
        
    Returns:
        Job ID if successful, None otherwise
    """
    # Convert file path to URI
    file_url = file_path.as_uri()
    
    # Queue the job
    job_id = None
    try:
        job_id = client.queue_job(
            file_url=file_url,
            track=track,
            metadata={
                "worker_id": WORKER_ID,
                "processed_at": int(time.time())
            }
        )
        
        logger.info(f"Queued job {job_id} for {file_path}")
        
        # Update job status to processing
        client.update_job_status(job_id, "processing")
        client.update_job_progress(job_id, "downloading", 10.0)
        
        # Validate MIME type
        mime_type = magic.from_file(file_path, mime=True)
        if mime_type != "application/pdf":
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        
        client.update_job_progress(job_id, "mime_check", 20.0)
        
        # Convert to markdown with math support
        md_path = pdf_to_markdown(file_path)
        client.update_job_progress(job_id, "text_extraction", 40.0)
        
        # Split into chunks
        chunks = list(chunk_file(md_path, max_len=MAX_CHUNK_SIZE))
        client.update_job_progress(job_id, "chunking", 60.0)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create embeddings
            concept_id, phase_vec = embed(chunk["text"])
            chunk["concept_id"] = concept_id
            chunk["phase_vec"] = phase_vec
            
            # Add chunk to job
            client.add_job_chunk(
                job_id,
                chunk["text"],
                chunk["start_offset"],
                chunk["end_offset"],
                chunk["metadata"]
            )
            
            # Add concept ID to job
            client.add_job_concept(job_id, concept_id)
            
            # Update progress
            progress = 60.0 + (30.0 * (i + 1) / len(chunks))
            client.update_job_progress(job_id, "embedding", progress)
        
        # Push concepts to knowledge graph
        push_concepts(chunks, metadata={"source": file_path.name})
        client.update_job_progress(job_id, "graph_insertion", 95.0)
        
        # Mark job as completed
        client.update_job_status(job_id, "completed")
        client.update_job_progress(job_id, "completed", 100.0)
        
        logger.info(f"Successfully processed job {job_id} with {len(chunks)} chunks")
        return job_id
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        
        if job_id:
            client.update_job_status(job_id, "failed", str(e))
        
        return None


def process_queued_jobs(client: IngestClient, max_jobs: int = 10) -> None:
    """
    Process queued jobs
    
    Args:
        client: Ingest client
        max_jobs: Maximum number of jobs to process
    """
    # Get queued jobs
    try:
        jobs = client.get_queued_jobs(limit=max_jobs)
        
        if not jobs:
            logger.info("No queued jobs found")
            return
        
        logger.info(f"Found {len(jobs)} queued jobs")
        
        # Process each job
        for job in jobs:
            job_id = job.get("job_id")
            file_url = job.get("file_url")
            track = job.get("track")
            
            if not file_url:
                logger.error(f"Job {job_id} has no file URL")
                continue
            
            logger.info(f"Processing job {job_id} from {file_url}")
            
            try:
                # Download the file
                client.update_job_status(job_id, "processing")
                client.update_job_progress(job_id, "downloading", 5.0)
                
                file_path = download_file(file_url, "downloads")
                
                # Process the file
                process_file(file_path, track, client)
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                client.update_job_status(job_id, "failed", str(e))
    
    except Exception as e:
        logger.error(f"Error getting queued jobs: {str(e)}")


def worker_loop(client: IngestClient, poll_interval: int = 5) -> None:
    """
    Worker loop
    
    Args:
        client: Ingest client
        poll_interval: Polling interval in seconds
    """
    logger.info(f"Starting worker loop for {WORKER_ID}")
    
    while True:
        try:
            # Process queued jobs
            process_queued_jobs(client)
            
            # Wait for the next poll
            time.sleep(poll_interval)
        
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        
        except Exception as e:
            logger.error(f"Error in worker loop: {str(e)}")
            time.sleep(poll_interval)


def process_file_list(file_list: List[str], track: str, client: IngestClient) -> None:
    """
    Process a list of files
    
    Args:
        file_list: List of file paths
        track: Track to assign
        client: Ingest client
    """
    logger.info(f"Processing {len(file_list)} files on track '{track}'")
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each file for processing
        future_to_file = {
            executor.submit(process_file, pathlib.Path(file), track, client): file
            for file in file_list
        }
        
        # Wait for all tasks to complete
        for future in future_to_file:
            file = future_to_file[future]
            try:
                job_id = future.result()
                if job_id:
                    logger.info(f"Successfully processed {file} as job {job_id}")
                else:
                    logger.error(f"Failed to process {file}")
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ingest worker')
    parser.add_argument('--mode', choices=['worker', 'file', 'dir'], default='worker',
                       help='Mode of operation')
    parser.add_argument('--file', help='File to process')
    parser.add_argument('--dir', help='Directory to process')
    parser.add_argument('--track', default='programming',
                       help='Track to assign (programming, math_physics, ai_ml, domain, ops_sre)')
    parser.add_argument('--url', default=INGEST_BUS_URL,
                       help='Ingest Bus URL')
    parser.add_argument('--key', default=INGEST_API_KEY,
                       help='API key')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help='Maximum number of workers')
    parser.add_argument('--interval', type=int, default=POLL_INTERVAL,
                       help='Polling interval in seconds')
    
    args = parser.parse_args()
    
    # Create client
    client = IngestClient(args.url, args.key)
    
    # Set globals
    global MAX_WORKERS, POLL_INTERVAL
    MAX_WORKERS = args.workers
    POLL_INTERVAL = args.interval
    
    # Run in the specified mode
    if args.mode == 'worker':
        worker_loop(client, args.interval)
    
    elif args.mode == 'file':
        if not args.file:
            logger.error("Missing required argument: --file")
            sys.exit(1)
        
        process_file(pathlib.Path(args.file), args.track, client)
    
    elif args.mode == 'dir':
        if not args.dir:
            logger.error("Missing required argument: --dir")
            sys.exit(1)
        
        dir_path = pathlib.Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"{args.dir} is not a directory")
            sys.exit(1)
        
        # Get all PDF files in the directory
        pdf_files = [str(f) for f in dir_path.glob("*.pdf")]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {args.dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(pdf_files)} PDF files in {args.dir}")
        
        # Process each file
        process_file_list(pdf_files, args.track, client)


if __name__ == "__main__":
    main()
