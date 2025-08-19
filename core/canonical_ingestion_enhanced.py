"""
Integration wrapper for canonical_ingestion.py to automatically capture mesh deltas
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import numpy as np
import os
import sys

# Add penrose_projector to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the original functions
from ingest_pdf.canonical_ingestion import (
    ActiveIngestionManager,
    get_active_ingestion_manager,
    ingest_pdf_file as original_ingest_pdf_file
)
from core.psi_archive_extended import PSI_ARCHIVER
from core.delta_tracking_mesh import DeltaTrackingConceptMesh
from penrose_projector import project_sparse

logger = logging.getLogger(__name__)


class EnhancedActiveIngestionManager(ActiveIngestionManager):
    """Enhanced ingestion manager that automatically tracks mesh deltas"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace the mesh with delta-tracking version if not already
        if not hasattr(self.koopman_graph, 'get_last_delta'):
            # Wrap existing mesh
            config = {'storage_path': self.koopman_graph.storage_path}
            new_mesh = DeltaTrackingConceptMesh(config)
            
            # Copy existing data
            new_mesh.concepts = self.koopman_graph.concepts
            new_mesh.relations = self.koopman_graph.relations
            new_mesh.name_index = self.koopman_graph.name_index
            new_mesh.adjacency = self.koopman_graph.adjacency
            
            # Replace reference
            self.koopman_graph = new_mesh
    
    def get_last_delta(self) -> Optional[Dict[str, Any]]:
        """Get mesh delta from the delta-tracking mesh"""
        if hasattr(self.koopman_graph, 'get_last_delta'):
            return self.koopman_graph.get_last_delta()
        else:
            logger.warning("Mesh does not support delta tracking")
            return None
    
    def ingest_concepts_from_source_with_delta(
        self,
        source: Any,
        content: str,
        session_id: Optional[str] = None,
        enable_penrose: bool = True
    ) -> Dict[str, Any]:
        """Enhanced ingestion that captures mesh delta"""
        # Clear any pending delta before starting
        if hasattr(self.koopman_graph, 'clear_delta'):
            self.koopman_graph.clear_delta()
        
        # Perform normal ingestion
        result = self.ingest_concepts_from_source(source, content)
        
        # Get mesh delta (automatically captured during ingestion)
        mesh_delta = self.get_last_delta()
        
        # Log to PsiArchive with delta
        if result.get('concepts_ingested', 0) > 0:
            ingested_concept_ids = result.get('ingested_concepts', [])
            penrose_stats = None
            
            # Run Penrose similarity if enabled and we have embeddings
            if enable_penrose and hasattr(self, 'embedder') and len(ingested_concept_ids) > 1:
                try:
                    # Collect embeddings for ingested concepts
                    embeddings = []
                    concept_id_list = []
                    
                    for concept_id in ingested_concept_ids:
                        concept = self.koopman_graph.concepts.get(concept_id)
                        if concept and hasattr(concept, 'embedding') and concept.embedding is not None:
                            embeddings.append(concept.embedding)
                            concept_id_list.append(concept_id)
                    
                    if len(embeddings) >= 2:
                        # Stack embeddings into matrix
                        embedding_matrix = np.vstack(embeddings)
                        
                        logger.info(f"🎯 Running Penrose projection on {len(embeddings)} concepts")
                        
                        # Compute sparse similarity
                        sparse_sim, penrose_stats = project_sparse(
                            embeddings=embedding_matrix,
                            rank=32,
                            threshold=0.7
                        )
                        
                        # Add relations to mesh
                        if hasattr(self.koopman_graph, 'add_relations_from_penrose'):
                            relation_stats = self.koopman_graph.add_relations_from_penrose(
                                sparse_matrix=sparse_sim,
                                concept_ids=concept_id_list
                            )
                            penrose_stats['relations'] = relation_stats
                            
                            # Save CSR matrix for replay
                            event_id = self._generate_event_id() if hasattr(self, '_generate_event_id') else f"evt_{session_id}"
                            csr_path = Path(f"data/concept_mesh/deltas/{event_id}.csr.zst")
                            
                            from penrose_projector.core import PenroseProjector
                            projector = PenroseProjector()
                            compression_stats = projector.save_sparse_compressed(sparse_sim, csr_path)
                            penrose_stats['csr_file'] = str(csr_path)
                            penrose_stats['compression'] = compression_stats
                            
                            logger.info(f"  ✅ Penrose: {relation_stats['relations_added']} relations added")
                            
                            # Update mesh delta to include Penrose relations
                            mesh_delta = self.get_last_delta()
                        
                except Exception as e:
                    logger.warning(f"Penrose projection failed: {e}")
                    penrose_stats = None
            
            # Extract tags from domain/category
            tags = []
            if hasattr(source, 'domain') and source.domain:
                tags.append(source.domain)
            if hasattr(source, 'category') and source.category:
                tags.append(source.category)
            
            event_id = PSI_ARCHIVER.log_concept_ingestion(
                concept_ids=ingested_concept_ids,
                source_doc_path=source.file_path if hasattr(source, 'file_path') else str(source),
                session_id=session_id or result.get('task_id'),
                mesh_delta=mesh_delta,
                tags=tags,
                penrose_stats=penrose_stats
            )
            
            result['psi_event_id'] = event_id
            logger.info(f"📝 Logged ingestion to PsiArchive: {event_id}")
        
        return result


# Enhanced singleton getter
_enhanced_manager = None

def get_enhanced_ingestion_manager(embedding_dim: int = 768) -> EnhancedActiveIngestionManager:
    """Get or create the enhanced singleton ingestion manager"""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedActiveIngestionManager(embedding_dim=embedding_dim)
    return _enhanced_manager


def ingest_pdf_file(
    file_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    source_type: str = "paper",
    domain: str = "general",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    enable_penrose: bool = True
) -> Dict[str, Any]:
    """Enhanced PDF ingestion with automatic delta tracking"""
    manager = get_enhanced_ingestion_manager()
    
    # Original ingestion
    result = manager.process_pdf_file(
        file_path=file_path,
        title=title,
        author=author,
        source_type=source_type,
        domain=domain,
        metadata=metadata
    )
    
    # Ensure PsiArchive logging happened
    if 'psi_event_id' not in result and result.get('status') == 'success':
        # Manual logging if needed
        source = manager.source_curator.get_source_by_id(result.get('source_id'))
        if source:
            mesh_delta = manager.get_last_delta()
            event_id = PSI_ARCHIVER.log_concept_ingestion(
                concept_ids=result.get('ingestion_result', {}).get('ingested_concepts', []),
                source_doc_path=file_path,
                session_id=session_id,
                mesh_delta=mesh_delta
            )
            result['psi_event_id'] = event_id
    
    return result


# Export enhanced functions
__all__ = [
    'get_enhanced_ingestion_manager',
    'ingest_pdf_file',
    'EnhancedActiveIngestionManager'
]
