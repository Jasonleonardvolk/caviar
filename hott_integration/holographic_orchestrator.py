"""
Holographic Ingestion Orchestrator
Coordinates the complete pipeline from upload to mesh integration
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import json
import time

from hott_integration.psi_morphon import HolographicMemory
from hott_integration.ingest_handlers.image_handler import ImageIngestHandler
from hott_integration.ingest_handlers.audio_handler import AudioIngestHandler
from hott_integration.ingest_handlers.video_handler import VideoIngestHandler
from hott_integration.concept_synthesizer import ConceptSynthesizer
from python.core.scoped_wal import WALManager

logger = logging.getLogger(__name__)

class HolographicOrchestrator:
    """
    Main orchestrator for holographic memory ingestion
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize handlers
        self.handlers = {
            'image': ImageIngestHandler(config),
            'audio': AudioIngestHandler(config),
            'video': VideoIngestHandler(config)
        }
        
        # Initialize synthesizer
        self.synthesizer = ConceptSynthesizer(config)
        
        # Storage settings
        self.storage_dir = Path(self.config.get('storage_dir', 'data/holograms'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing queue
        self.processing_queue = asyncio.Queue()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🎭 Holographic Orchestrator initialized")
    
    def get_handler_for_file(self, file_path: Path) -> Optional[Any]:
        """Get the appropriate handler for a file"""
        for handler_type, handler in self.handlers.items():
            if handler.can_handle(file_path):
                return handler
        return None
    
    async def ingest_file(self, file_path: Path, tenant_scope: str, tenant_id: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ingest a media file and return job ID
        
        Args:
            file_path: Path to the media file
            tenant_scope: "user" or "group"
            tenant_id: ID of the tenant
            metadata: Additional metadata
            
        Returns:
            Job ID for tracking
        """
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get handler
        handler = self.get_handler_for_file(file_path)
        if not handler:
            raise ValueError(f"No handler for file type: {file_path.suffix}")
        
        # Create job
        job_id = f"holo_job_{int(time.time() * 1000)}"
        job = {
            "id": job_id,
            "file_path": str(file_path),
            "tenant_scope": tenant_scope,
            "tenant_id": tenant_id,
            "metadata": metadata or {},
            "status": "queued",
            "created_at": time.time(),
            "handler_type": type(handler).__name__
        }
        
        self.active_jobs[job_id] = job
        
        # Queue for processing
        await self.processing_queue.put(job)
        
        # Start processor if not running
        asyncio.create_task(self._process_queue())
        
        logger.info(f"📥 Queued ingestion job {job_id} for {file_path}")
        return job_id
    
    async def _process_queue(self):
        """Process jobs from the queue"""
        while True:
            try:
                # Get next job
                job = await asyncio.wait_for(self.processing_queue.get(), timeout=5.0)
                
                # Update status
                job["status"] = "processing"
                job["started_at"] = time.time()
                
                # Process the job
                await self._process_job(job)
                
            except asyncio.TimeoutError:
                # No jobs in queue
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single ingestion job"""
        job_id = job["id"]
        file_path = Path(job["file_path"])
        
        try:
            logger.info(f"🔄 Processing job {job_id}")
            
            # Get handler
            handler = self.get_handler_for_file(file_path)
            if not handler:
                raise ValueError("Handler not found")
            
            # Get WAL for tenant
            wal = WALManager.get_wal(job["tenant_scope"], job["tenant_id"])
            
            # Log start in WAL
            wal.write("hologram_ingest_start", {
                "job_id": job_id,
                "file": str(file_path),
                "handler": type(handler).__name__
            })
            
            # Ingest file
            memory = await handler.ingest(
                file_path,
                job["tenant_scope"],
                job["tenant_id"],
                job["metadata"]
            )
            
            # Synthesize connections
            connections = await self.synthesizer.synthesize(memory)
            memory.metadata["connections_created"] = connections
            
            # Save holographic memory
            memory_path = await self._save_memory(memory, job)
            
            # Log completion in WAL
            wal.write("hologram_ingest_complete", {
                "job_id": job_id,
                "morphons": len(memory.morphons),
                "strands": len(memory.strands),
                "connections": connections,
                "memory_path": str(memory_path)
            })
            
            # Update job status
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["processing_time"] = job["completed_at"] - job["started_at"]
            job["result"] = {
                "memory_id": memory.id,
                "morphons": len(memory.morphons),
                "strands": len(memory.strands),
                "connections": connections,
                "memory_path": str(memory_path)
            }
            
            logger.info(f"✅ Job {job_id} completed: {len(memory.morphons)} morphons, "
                       f"{len(memory.strands)} strands, {connections} connections")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Log failure in WAL
            wal.write("hologram_ingest_failed", {
                "job_id": job_id,
                "error": str(e)
            })
            
            # Update job status
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = str(e)
    
    async def _save_memory(self, memory: HolographicMemory, job: Dict[str, Any]) -> Path:
        """Save holographic memory to disk"""
        # Create tenant directory
        tenant_dir = self.storage_dir / job["tenant_scope"] / job["tenant_id"]
        tenant_dir.mkdir(parents=True, exist_ok=True)
        
        # Save memory
        memory_file = tenant_dir / f"{memory.id}.json"
        with open(memory_file, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
        
        logger.info(f"💾 Saved holographic memory to {memory_file}")
        return memory_file
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job"""
        return self.active_jobs.get(job_id)
    
    def get_all_jobs(self, tenant_scope: Optional[str] = None, 
                    tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all jobs, optionally filtered by tenant"""
        jobs = list(self.active_jobs.values())
        
        if tenant_scope:
            jobs = [j for j in jobs if j["tenant_scope"] == tenant_scope]
        if tenant_id:
            jobs = [j for j in jobs if j["tenant_id"] == tenant_id]
        
        return jobs
    
    async def get_memory(self, memory_id: str, tenant_scope: str, 
                        tenant_id: str) -> Optional[HolographicMemory]:
        """Load a holographic memory from disk"""
        memory_file = self.storage_dir / tenant_scope / tenant_id / f"{memory_id}.json"
        
        if not memory_file.exists():
            return None
        
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
            
            return HolographicMemory.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load memory {memory_id}: {e}")
            return None
    
    async def list_memories(self, tenant_scope: str, tenant_id: str) -> List[Dict[str, Any]]:
        """List all memories for a tenant"""
        tenant_dir = self.storage_dir / tenant_scope / tenant_id
        
        if not tenant_dir.exists():
            return []
        
        memories = []
        for memory_file in tenant_dir.glob("*.json"):
            try:
                # Load basic info without full parsing
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                
                memories.append({
                    "id": data["id"],
                    "title": data.get("title"),
                    "source_file": data.get("source_file"),
                    "created_at": data.get("created_at"),
                    "num_morphons": len(data.get("morphons", [])),
                    "num_strands": len(data.get("strands", []))
                })
                
            except Exception as e:
                logger.error(f"Failed to read memory file {memory_file}: {e}")
        
        return memories


# Singleton instance
_orchestrator = None

def get_orchestrator(config: Optional[Dict[str, Any]] = None) -> HolographicOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HolographicOrchestrator(config)
    return _orchestrator
