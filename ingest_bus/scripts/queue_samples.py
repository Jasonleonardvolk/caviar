#!/usr/bin/env python
"""
Queue Sample PDFs Script

This script queues sample PDFs for processing, with deduplication
and status tracking. It's designed to be run from VS Code tasks.

Usage:
    python queue_samples.py [--force] PATH [PATH...]
    
Arguments:
    PATH        Path(s) to PDF file(s) or directory containing PDFs
    
Options:
    --force     Force reprocessing, even if the file is a duplicate
"""

import os
import sys
import glob
import time
import hashlib
import argparse
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ingest Bus client configuration
INGEST_BUS_URL = os.environ.get("INGEST_BUS_URL", "http://localhost:8000")
INGEST_API_KEY = os.environ.get("INGEST_API_KEY", "")


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to efficiently handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def queue_pdf(file_path: Path, force: bool = False, track: Optional[str] = None) -> Dict[str, Any]:
    """
    Queue a PDF for processing
    
    Args:
        file_path: Path to the PDF file
        force: Whether to force reprocessing even if the file is a duplicate
        track: Optional track to assign the job to
        
    Returns:
        Response from the API
    """
    logger.info(f"Queueing {file_path}...")
    
    # Calculate file hash
    file_hash = calculate_file_hash(file_path)
    short_hash = file_hash[:8]
    
    # Get default track from file path if not provided
    if not track:
        # Try to infer from directory name
        parent_dir = file_path.parent.name.lower()
        if 'programming' in parent_dir or 'prog' in parent_dir:
            track = 'programming'
        elif 'math' in parent_dir or 'physics' in parent_dir:
            track = 'math_physics'
        elif 'ai' in parent_dir or 'ml' in parent_dir:
            track = 'ai_ml'
        else:
            track = 'general'

    # Prepare request
    url = f"{INGEST_BUS_URL}/api/jobs"
    
    data = {
        "file_url": file_path.as_uri(),
        "track": track,
        "metadata": {
            "file_hash": file_hash,
            "file_name": file_path.name,
            "queued_at": int(time.time()),
            "source": "queue_samples.py"
        },
        "force": force
    }
    
    headers = {}
    if INGEST_API_KEY:
        headers["X-API-Key"] = INGEST_API_KEY
    
    # Send request
    response = requests.post(url, json=data, headers=headers)
    
    # Check for duplicates (HTTP 409)
    if response.status_code == 409:
        result = response.json()
        job_id = result.get('job_id')
        status = result.get('status')
        
        logger.info(f"Duplicate file detected: {file_path.name} [{short_hash}]")
        logger.info(f"  Existing job: {job_id} (status: {status})")
        
        if force:
            logger.info("  Forcing reprocessing...")
            data["force"] = True
            response = requests.post(url, json=data, headers=headers)
            result = response.json()
        
        return result
    
    # Check for other errors
    if not response.ok:
        logger.error(f"Error queueing {file_path.name}: {response.status_code} {response.text}")
        return {"success": False, "error": response.text}
    
    # Return response
    result = response.json()
    job_id = result.get('job_id')
    
    logger.info(f"Successfully queued {file_path.name} [{short_hash}]")
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Track: {track}")
    
    return result


def process_files(file_paths: List[Path], force: bool) -> None:
    """
    Process a list of files
    
    Args:
        file_paths: List of file paths
        force: Whether to force reprocessing
    """
    logger.info(f"Processing {len(file_paths)} file(s)")
    
    # Track statistics
    stats = {
        "queued": 0,
        "skipped_duplicates": 0,
        "errors": 0
    }
    
    # Process each file
    for file_path in file_paths:
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                stats["errors"] += 1
                continue
            
            if not file_path.is_file():
                logger.error(f"Not a file: {file_path}")
                stats["errors"] += 1
                continue
            
            result = queue_pdf(file_path, force)
            
            if result.get("success", False):
                stats["queued"] += 1
            elif result.get("duplicate", False):
                stats["skipped_duplicates"] += 1
            else:
                stats["errors"] += 1
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            stats["errors"] += 1
    
    # Print summary
    logger.info("\n== Summary ==")
    logger.info(f"Files queued: {stats['queued']}")
    logger.info(f"Duplicates skipped: {stats['skipped_duplicates']}")
    logger.info(f"Errors: {stats['errors']}")


def find_pdfs(paths: List[str]) -> List[Path]:
    """
    Find PDFs from the given paths
    
    Args:
        paths: List of file or directory paths
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    
    for path in paths:
        path_obj = Path(path)
        
        if path_obj.is_file() and path_obj.suffix.lower() == '.pdf':
            # Single file
            pdf_files.append(path_obj)
        elif path_obj.is_dir():
            # Directory - find PDFs
            for pdf_path in path_obj.glob('**/*.pdf'):
                pdf_files.append(pdf_path)
        elif '*' in path:
            # Glob pattern
            for pdf_path in glob.glob(path, recursive=True):
                if pdf_path.lower().endswith('.pdf'):
                    pdf_files.append(Path(pdf_path))
        else:
            logger.warning(f"Invalid path (not a PDF or directory): {path}")
    
    return pdf_files


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Queue PDFs for processing')
    parser.add_argument('paths', nargs='+', help='Path(s) to PDF file(s) or directory containing PDFs')
    parser.add_argument('--force', action='store_true', help='Force reprocessing, even if the file is a duplicate')
    
    args = parser.parse_args()
    
    # Find PDFs
    pdf_files = find_pdfs(args.paths)
    
    if not pdf_files:
        logger.error("No PDF files found")
        sys.exit(1)
    
    # Process files
    process_files(pdf_files, args.force)


if __name__ == "__main__":
    main()
