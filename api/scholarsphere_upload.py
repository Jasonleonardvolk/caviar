"""
ScholarSphere Upload Integration
Handles uploading concept diffs to ScholarSphere after processing
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Configuration
SCHOLARSPHERE_API_URL = os.getenv("SCHOLARSPHERE_API_URL", "https://api.scholarsphere.org")
SCHOLARSPHERE_API_KEY = os.getenv("SCHOLARSPHERE_API_KEY", "")
SCHOLARSPHERE_BUCKET = os.getenv("SCHOLARSPHERE_BUCKET", "concept-diffs")
UPLOAD_TIMEOUT = int(os.getenv("SCHOLARSPHERE_UPLOAD_TIMEOUT", "300"))  # 5 minutes

class ScholarSphereUploader:
    """Handles uploading concept diffs to ScholarSphere"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SCHOLARSPHERE_API_KEY or "local_no_auth"
        self.api_url = SCHOLARSPHERE_API_URL
        self.bucket = SCHOLARSPHERE_BUCKET
        
    async def get_presigned_url(self, filename: str, content_type: str = "application/x-ndjson") -> Dict[str, str]:
        """Get a presigned URL for uploading to ScholarSphere (local filesystem version)"""
        import uuid
        
        # Local filesystem implementation - no external API needed
        diff_id = uuid.uuid4().hex
        upload_path = Path("data/scholarsphere/uploaded") / f"{diff_id}_{filename}"
        
        # Ensure directory exists
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "url": upload_path.as_uri(),
            "diff_id": diff_id,
            "local_path": str(upload_path)
        }
    
    async def upload_file(self, file_path: str, presigned_data: Dict[str, str]) -> bool:
        """Upload a file using local filesystem move"""
        try:
            import shutil
            
            # Extract local path from presigned data
            local_path = presigned_data.get('local_path')
            if not local_path:
                logger.error("No local path in presigned data")
                return False
            
            # Move file to "uploaded" directory
            shutil.move(file_path, local_path)
            logger.info(f"Successfully moved {file_path} to ScholarSphere (local): {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    async def upload_concepts(self, concepts: List[Dict[str, Any]], source: str = "unknown") -> Optional[str]:
        """
        Upload concepts to ScholarSphere
        
        Args:
            concepts: List of concept dictionaries
            source: Source identifier (e.g., "pdf_upload", "manual_entry")
            
        Returns:
            Diff ID if successful, None otherwise
        """
        # Generate diff file
        timestamp = datetime.utcnow()
        diff_id = f"{source}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        filename = f"diff_{diff_id}.jsonl"
        
        # Ensure output directory exists
        output_dir = Path("data/scholarsphere/pending")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / filename
        
        # Write JSONL file
        with open(file_path, 'w') as f:
            for concept in concepts:
                entry = {
                    "diff_id": diff_id,
                    "timestamp": timestamp.isoformat(),
                    "source": source,
                    "concept": concept
                }
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Created diff file: {file_path} with {len(concepts)} concepts")
        
        # Get presigned URL
        presigned_data = await self.get_presigned_url(filename)
        if not presigned_data or 'local_path' not in presigned_data:
            logger.warning("Could not get presigned URL, keeping file in pending")
            return diff_id
        
        # Upload file
        success = await self.upload_file(str(file_path), presigned_data)
        
        if success:
            # File already moved by upload_file, just update the diff_id
            
            # Notify ScholarSphere of completion (local version - just log)
            await self.notify_upload_complete(diff_id, filename)
            
            return diff_id
        else:
            logger.error(f"Failed to upload {filename}, keeping in pending")
            return None
    
    async def notify_upload_complete(self, diff_id: str, filename: str):
        """Notify ScholarSphere that an upload is complete (local version)"""
        # Local implementation - just log the completion
        logger.info(f"ScholarSphere upload completed: diff_id={diff_id}, filename={filename}")
        logger.info(f"File stored in: data/scholarsphere/uploaded/")
    
    async def check_status(self, diff_id: str) -> Dict[str, Any]:
        """Check the status of a diff in ScholarSphere (local version)"""
        # Local implementation - check if file exists in uploaded directory
        uploaded_dir = Path("data/scholarsphere/uploaded")
        
        # Look for files with this diff_id
        matching_files = list(uploaded_dir.glob(f"{diff_id}_*.jsonl"))
        
        if matching_files:
            return {
                "status": "uploaded",
                "message": "File found in local ScholarSphere",
                "files": [str(f) for f in matching_files]
            }
        else:
            return {
                "status": "not_found",
                "message": f"No files found for diff_id: {diff_id}"
            }

# Global uploader instance
_uploader = None

def get_uploader() -> ScholarSphereUploader:
    """Get or create the global uploader instance"""
    global _uploader
    if _uploader is None:
        _uploader = ScholarSphereUploader()
    return _uploader

async def upload_concepts_to_scholarsphere(concepts: List[Dict[str, Any]], source: str = "unknown") -> Optional[str]:
    """
    Convenience function to upload concepts to ScholarSphere
    
    Args:
        concepts: List of concept dictionaries
        source: Source identifier
        
    Returns:
        Diff ID if successful, None otherwise
    """
    uploader = get_uploader()
    return await uploader.upload_concepts(concepts, source)

def upload_concepts_sync(concepts: List[Dict[str, Any]], source: str = "unknown") -> Optional[str]:
    """Synchronous wrapper for uploading concepts"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(upload_concepts_to_scholarsphere(concepts, source))
    finally:
        loop.close()
