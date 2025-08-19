"""
Base AV Ingest Handler
Abstract base class for processing audiovisual content
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import json

from hott_integration.psi_morphon import (
    PsiMorphon, PsiStrand, HolographicMemory,
    ModalityType, StrandType
)

logger = logging.getLogger(__name__)

class BaseIngestHandler(ABC):
    """
    Abstract base class for ingesting different media types
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_extensions = []
        self.modality_type = ModalityType.TEXT
        
        # Processing settings
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.temp_dir = Path(self.config.get('temp_dir', 'data/temp'))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the file"""
        return file_path.suffix.lower() in self.supported_extensions
    
    async def ingest(self, file_path: Path, tenant_scope: str, tenant_id: str,
                    metadata: Optional[Dict[str, Any]] = None) -> HolographicMemory:
        """
        Main ingestion pipeline
        
        Args:
            file_path: Path to the media file
            tenant_scope: "user" or "group"
            tenant_id: ID of the tenant
            metadata: Additional metadata
            
        Returns:
            HolographicMemory containing morphons and strands
        """
        logger.info(f"Starting ingestion of {file_path} for {tenant_scope}:{tenant_id}")
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {file_path.stat().st_size} bytes")
        
        # Create holographic memory container
        memory = HolographicMemory(
            source_file=str(file_path),
            tenant_scope=tenant_scope,
            tenant_id=tenant_id,
            title=metadata.get('title') if metadata else file_path.stem
        )
        
        try:
            # Extract content-specific morphons
            morphons = await self.extract_morphons(file_path, metadata)
            for morphon in morphons:
                morphon.source_file = str(file_path)
                morphon.source_type = self.modality_type.value
                memory.add_morphon(morphon)
            
            # Generate embeddings
            await self.generate_embeddings(memory)
            
            # Create cross-modal strands
            strands = await self.create_strands(memory)
            for strand in strands:
                memory.add_strand(strand)
            
            # Link to existing concepts
            await self.link_to_concepts(memory)
            
            memory.processed = True
            logger.info(f"Successfully ingested {file_path}: "
                       f"{len(memory.morphons)} morphons, {len(memory.strands)} strands")
            
        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            memory.processed = False
            memory.error = str(e)
            raise
        
        return memory
    
    @abstractmethod
    async def extract_morphons(self, file_path: Path, 
                             metadata: Optional[Dict[str, Any]]) -> List[PsiMorphon]:
        """Extract morphons from the media file"""
        pass
    
    async def generate_embeddings(self, memory: HolographicMemory) -> None:
        """Generate embeddings for morphons that don't have them"""
        for morphon in memory.morphons:
            if morphon.embedding is None:
                morphon.embedding = await self._generate_embedding(morphon)
    
    async def _generate_embedding(self, morphon: PsiMorphon) -> Optional[np.ndarray]:
        """Generate embedding for a single morphon"""
        # Default implementation - override in subclasses
        import numpy as np
        
        if morphon.modality == ModalityType.TEXT and morphon.content:
            # Simple hash-based embedding for text
            text_hash = hashlib.sha256(str(morphon.content).encode()).hexdigest()
            # Convert to 512-dim vector
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.randn(512).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        return None
    
    async def create_strands(self, memory: HolographicMemory) -> List[PsiStrand]:
        """Create strands between morphons"""
        strands = []
        
        # Default: connect temporally adjacent morphons
        morphons = sorted(memory.morphons, 
                         key=lambda m: m.temporal_index or 0)
        
        for i in range(len(morphons) - 1):
            if morphons[i].temporal_index is not None:
                strand = PsiStrand(
                    source_morphon_id=morphons[i].id,
                    target_morphon_id=morphons[i + 1].id,
                    strand_type=StrandType.TEMPORAL,
                    strength=0.8,
                    temporal_offset=morphons[i + 1].temporal_index - morphons[i].temporal_index
                )
                strands.append(strand)
        
        return strands
    
    async def link_to_concepts(self, memory: HolographicMemory) -> None:
        """Link morphons to existing concepts in the mesh"""
        # This will be implemented to connect to the tenant's concept mesh
        # For now, just log the intent
        logger.info(f"Would link {len(memory.morphons)} morphons to concept mesh")
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file for deduplication"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file metadata"""
        stat = file_path.stat()
        return {
            "filename": file_path.name,
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "hash": self.compute_file_hash(file_path)
        }


# Import numpy for embeddings
import numpy as np
