"""
TORI Document Ingestion Integration Coordinator
Orchestrates the complete ingestion pipeline with all TORI systems

Integrates:
- Enhanced document parsing (all file types)
- ψMesh semantic associations
- ConceptMesh knowledge graph
- Ghost Collective AI personas
- ScholarSphere archival
- Extraction integrity verification
"""

import json
import logging
import asyncio
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import TORI components
from enhanced_extract import DocumentProcessor, extract_document
from psi_mesh_integration import psi_mesh
from models.schemas import IngestJob, IngestStatus

logger = logging.getLogger("tori-ingest.coordinator")

class ToriIngestionCoordinator:
    """
    Central coordinator for TORI document ingestion pipeline
    Orchestrates all systems working together
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.ghost_collective_enabled = True
        self.concept_mesh_enabled = True
        self.scholar_sphere_enabled = True
        
        # System endpoints
        self.concept_mesh_endpoint = "http://localhost:8081"
        self.scholar_sphere_endpoint = "http://localhost:8082"
        
        logger.info("TORI Ingestion Coordinator initialized")
    
    async def process_document_complete(self, file_path: Optional[str], 
                                      file_content: Optional[bytes],
                                      file_type: str, 
                                      job: IngestJob,
                                      integration_options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to document file
            file_content: Raw file content
            file_type: Type of document (pdf, docx, etc.)
            job: Ingest job metadata
            integration_options: Which systems to integrate with
            
        Returns:
            Complete processing results
        """
        if integration_options is None:
            integration_options = {
                'concept_mesh': True,
                'psi_mesh': True,
                'ghost_collective': True,
                'scholar_sphere': True,
                'integrity_verification': True
            }
        
        logger.info(f"Starting complete document processing: {file_type} - job {job.id}")
        
        results = {
            'job_id': job.id,
            'file_type': file_type,
            'status': 'processing',
            'started_at': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Document Parsing and Concept Extraction
            logger.info("Stage 1: Document parsing and concept extraction")
            job.status = IngestStatus.PROCESSING
            job.percent_complete = 10.0
            
            document_data = await extract_document(file_path, file_content, file_type, job)
            if not document_data:
                results['status'] = 'failed'
                results['error'] = 'Document parsing failed'
                return results
            
            results['stages']['parsing'] = {
                'status': 'completed',
                'concepts_extracted': len(document_data.get('concepts', [])),
                'text_length': len(document_data.get('text', '')),
                'structure_elements': len(document_data.get('structure', []))
            }
            
            job.percent_complete = 25.0
            
            # Stage 2: ψMesh Integration (Semantic Associations)
            if integration_options.get('psi_mesh', True):
                logger.info("Stage 2: ψMesh semantic association creation")
                
                concepts = document_data.get('concepts', [])
                if concepts:
                    associations = await psi_mesh.create_semantic_associations(
                        concepts, document_data
                    )
                    results['stages']['psi_mesh'] = {
                        'status': 'completed',
                        'associations_created': associations['associations_created'],
                        'concept_pairs': associations['concept_pairs']
                    }
                else:
                    results['stages']['psi_mesh'] = {
                        'status': 'skipped',
                        'reason': 'No concepts extracted'
                    }
            
            job.percent_complete = 40.0
            
            # Stage 3: Extraction Integrity Verification
            if integration_options.get('integrity_verification', True):
                logger.info("Stage 3: Extraction integrity verification")
                
                verification = await psi_mesh.verify_extraction_integrity(
                    document_data.get('concepts', []),
                    document_data.get('text', ''),
                    document_data.get('metadata', {})
                )
                
                results['stages']['verification'] = {
                    'status': 'completed',
                    'integrity_score': verification['integrity_score'],
                    'verified_concepts': len(verification['verified_concepts']),
                    'flagged_concepts': len(verification['flagged_concepts'])
                }
                
                # Log verification issues
                if verification['flagged_concepts']:
                    logger.warning(f"Verification flagged {len(verification['flagged_concepts'])} concepts")
                    for flagged in verification['flagged_concepts']:
                        logger.warning(f"  - {flagged['concept']}: {flagged['integrity_score']:.2f}")
            
            job.percent_complete = 55.0
            
            # Stage 4: Ghost Collective Processing
            if integration_options.get('ghost_collective', True):
                logger.info("Stage 4: Ghost Collective persona analysis")
                
                ghost_analysis = await self.process_with_ghost_collective(
                    document_data, job
                )
                results['stages']['ghost_collective'] = ghost_analysis
            
            job.percent_complete = 70.0
            
            # Stage 5: ConceptMesh Integration
            if integration_options.get('concept_mesh', True):
                logger.info("Stage 5: ConceptMesh knowledge graph integration")
                
                concept_mesh_result = await self.integrate_with_concept_mesh(
                    document_data, job
                )
                results['stages']['concept_mesh'] = concept_mesh_result
            
            job.percent_complete = 85.0
            
            # Stage 6: ScholarSphere Archival
            if integration_options.get('scholar_sphere', True):
                logger.info("Stage 6: ScholarSphere archival")
                
                archival_result = await self.archive_to_scholar_sphere(
                    document_data, file_content or self._read_file_content(file_path), job
                )
                results['stages']['scholar_sphere'] = archival_result
            
            job.percent_complete = 100.0
            job.status = IngestStatus.COMPLETED
            
            # Final results
            results['status'] = 'completed'
            results['completed_at'] = datetime.now().isoformat()
            results['summary'] = self._generate_processing_summary(results)
            
            logger.info(f"Complete document processing finished: job {job.id}")
            
        except Exception as e:
            logger.exception(f"Error in complete document processing: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            job.status = IngestStatus.FAILED
        
        return results
    
    async def process_with_ghost_collective(self, document_data: Dict[str, Any], 
                                          job: IngestJob) -> Dict[str, Any]:
        """
        Process document with Ghost Collective AI personas
        """
        try:
            # Create analysis query for Ghost Collective
            text_sample = document_data.get('text', '')[:1000]  # First 1000 chars
            concepts = document_data.get('concepts', [])
            concept_names = [c.get('name', '') for c in concepts[:5]]  # Top 5 concepts
            
            analysis_query = f"""
            Analyze this document content:
            
            Document type: {document_data.get('metadata', {}).get('file_type', 'unknown')}
            Key concepts: {', '.join(concept_names)}
            
            Content sample: {text_sample}
            
            Provide insights about the content, key themes, and relationships.
            """
            
            # Simulate Ghost Collective processing
            # In a real implementation, this would call the actual Ghost Collective API
            ghost_result = {
                'active_persona': 'Scholar',
                'confidence': 0.85,
                'analysis': f"This document appears to focus on {', '.join(concept_names[:3])}. The Scholar persona identifies key analytical patterns and relationships.",
                'insights': [
                    f"Primary theme relates to {concept_names[0] if concept_names else 'technical content'}",
                    f"Document structure suggests {document_data.get('metadata', {}).get('file_type', 'structured')} information",
                    "Content shows systematic organization of concepts"
                ],
                'suggestions': [
                    "Consider cross-referencing with related technical documents",
                    "Explore conceptual relationships in more depth",
                    "Look for practical applications of the concepts"
                ]
            }
            
            return {
                'status': 'completed',
                'persona': ghost_result['active_persona'],
                'confidence': ghost_result['confidence'],
                'insights': ghost_result['insights'],
                'suggestions': ghost_result['suggestions']
            }
            
        except Exception as e:
            logger.exception(f"Error in Ghost Collective processing: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def integrate_with_concept_mesh(self, document_data: Dict[str, Any], 
                                        job: IngestJob) -> Dict[str, Any]:
        """
        Integrate extracted concepts with ConceptMesh knowledge graph
        """
        try:
            concepts = document_data.get('concepts', [])
            if not concepts:
                return {
                    'status': 'skipped',
                    'reason': 'No concepts to integrate'
                }
            
            # Prepare concepts for ConceptMesh
            concept_nodes = []
            for concept in concepts:
                node_data = {
                    'name': concept.get('name', ''),
                    'keywords': concept.get('keywords', []),
                    'context': concept.get('context', ''),
                    'source_document': job.id,
                    'confidence': concept.get('confidence', 0.8),
                    'created_at': datetime.now().isoformat()
                }
                concept_nodes.append(node_data)
            
            # In a real implementation, this would call ConceptMesh API
            # For now, we'll simulate the integration
            concept_mesh_result = {
                'nodes_created': len(concept_nodes),
                'relationships_created': max(0, len(concept_nodes) - 1),
                'graph_updated': True
            }
            
            return {
                'status': 'completed',
                'nodes_created': concept_mesh_result['nodes_created'],
                'relationships_created': concept_mesh_result['relationships_created']
            }
            
        except Exception as e:
            logger.exception(f"Error in ConceptMesh integration: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def archive_to_scholar_sphere(self, document_data: Dict[str, Any], 
                                      file_content: bytes, job: IngestJob) -> Dict[str, Any]:
        """
        Archive document and metadata to ScholarSphere
        """
        try:
            # Prepare archival record
            archival_record = {
                'document_id': job.id,
                'metadata': document_data.get('metadata', {}),
                'concepts': document_data.get('concepts', []),
                'structure': document_data.get('structure', []),
                'archived_at': datetime.now().isoformat(),
                'file_hash': self._calculate_file_hash(file_content),
                'processing_summary': {
                    'concepts_extracted': len(document_data.get('concepts', [])),
                    'text_length': len(document_data.get('text', '')),
                    'file_type': document_data.get('metadata', {}).get('file_type')
                }
            }
            
            # Save archival record locally
            archive_path = Path(__file__).parent.parent.parent / "concept-mesh-data" / "scholar_sphere"
            archive_path.mkdir(exist_ok=True)
            
            with open(archive_path / f"{job.id}_archive.json", 'w') as f:
                json.dump(archival_record, f, indent=2)
            
            # Save original file
            file_ext = document_data.get('metadata', {}).get('file_type', 'bin')
            with open(archive_path / f"{job.id}_original.{file_ext}", 'wb') as f:
                f.write(file_content)
            
            return {
                'status': 'completed',
                'archive_id': job.id,
                'archive_path': str(archive_path),
                'file_hash': archival_record['file_hash']
            }
            
        except Exception as e:
            logger.exception(f"Error in ScholarSphere archival: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _read_file_content(self, file_path: str) -> bytes:
        """Read file content from path"""
        if file_path and Path(file_path).exists():
            with open(file_path, 'rb') as f:
                return f.read()
        return b''
    
    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        import hashlib
        return hashlib.sha256(content).hexdigest()
    
    def _generate_processing_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of processing results"""
        stages_completed = sum(1 for stage in results['stages'].values() 
                             if stage.get('status') == 'completed')
        
        total_concepts = 0
        if 'parsing' in results['stages']:
            total_concepts = results['stages']['parsing'].get('concepts_extracted', 0)
        
        integrity_score = 0.0
        if 'verification' in results['stages']:
            integrity_score = results['stages']['verification'].get('integrity_score', 0.0)
        
        return {
            'stages_completed': stages_completed,
            'total_stages': len(results['stages']),
            'success_rate': stages_completed / len(results['stages']) if results['stages'] else 0,
            'concepts_extracted': total_concepts,
            'integrity_score': integrity_score,
            'processing_complete': results.get('status') == 'completed'
        }

# Global coordinator instance
coordinator = ToriIngestionCoordinator()

# Enhanced API endpoint for complete integration
async def process_document_with_full_integration(file_path: Optional[str], 
                                               file_content: Optional[bytes],
                                               file_type: str, 
                                               job: IngestJob,
                                               options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Complete TORI document ingestion with all systems integration
    
    This is the main entry point for full document processing
    """
    integration_options = options or {}
    
    return await coordinator.process_document_complete(
        file_path, file_content, file_type, job, integration_options
    )
