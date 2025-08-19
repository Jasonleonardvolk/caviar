# core/psi_archive_extended.py - Fixed Psi Archive with safe JSON encoding
import logging
import time
import os
from typing import List, Dict, Any
from .json_encoder_fix import safe_json_dump

logger = logging.getLogger(__name__)

class PsiArchiveExtended:
    """Extended Psi Archive for concept provenance tracking"""
    
    def __init__(self):
        # Use local directory for Windows
        self.archive_dir = os.getenv("PSI_ARCHIVE_DIR", "./psi_archive")
        os.makedirs(self.archive_dir, exist_ok=True)
        logger.info(f"Psi Archive initialized at: {self.archive_dir}")
    
    def log_concept_ingestion(
        self,
        source_path: str,
        concepts: List[str],
        embeddings: List[List[float]],
        mesh_delta: Dict[str, Any],
        penrose_stats: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Log concept ingestion with full provenance"""
        archive_id = f"archive_{int(time.time() * 1000)}_{abs(hash(source_path)) % 1000000}"
        
        # Convert penrose_stats if it's an object
        if hasattr(penrose_stats, '__dict__'):
            penrose_dict = {}
            for key, value in penrose_stats.__dict__.items():
                if not key.startswith('_'):
                    if hasattr(value, '__dict__'):
                        # Convert nested objects
                        penrose_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                    else:
                        penrose_dict[key] = value
            penrose_stats = penrose_dict
        
        archive_entry = {
            "id": archive_id,
            "timestamp": time.time(),
            "source_path": source_path,
            "concepts_count": len(concepts),
            "mesh_delta": mesh_delta,
            "penrose_stats": penrose_stats,
            "metadata": metadata or {}
        }
        
        # Save with safe JSON encoding
        archive_path = os.path.join(self.archive_dir, f"{archive_id}.json")
        with open(archive_path, 'w') as f:
            safe_json_dump(archive_entry, f, indent=2)
        
        logger.info(f"Archived ingestion: {archive_id}")
        return archive_id
    
    def log_error(
        self,
        error_type: str,
        source: str,
        error: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Log error for analysis"""
        error_id = f"error_{int(time.time() * 1000)}_{abs(hash(error)) % 1000000}"
        
        error_entry = {
            "id": error_id,
            "timestamp": time.time(),
            "error_type": error_type,
            "source": source,
            "error": error,
            "metadata": metadata or {}
        }
        
        error_path = os.path.join(self.archive_dir, f"{error_id}.json")
        with open(error_path, 'w') as f:
            safe_json_dump(error_entry, f, indent=2)
        
        logger.error(f"Archived error: {error_id}")
        return error_id

# Global instance
PSI_ARCHIVER = PsiArchiveExtended()
