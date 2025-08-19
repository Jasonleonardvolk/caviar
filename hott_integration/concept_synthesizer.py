"""
Concept Synthesizer
Links PsiMorphons to existing concepts in the tenant's mesh
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hott_integration.psi_morphon import (
    PsiMorphon, PsiStrand, HolographicMemory,
    ModalityType, StrandType
)
from python.core.scoped_concept_mesh import ScopedConceptMesh
from hott_integration.tenant_proof_queue import create_tenant_proof

logger = logging.getLogger(__name__)

class ConceptSynthesizer:
    """
    Synthesizes connections between morphons and existing concepts
    Creates cross-modal strands and updates the concept mesh
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Similarity thresholds
        self.semantic_threshold = self.config.get('semantic_threshold', 0.7)
        self.visual_threshold = self.config.get('visual_threshold', 0.8)
        self.temporal_window = self.config.get('temporal_window', 5.0)  # seconds
        
        # Concept creation settings
        self.auto_create_concepts = self.config.get('auto_create_concepts', True)
        self.min_salience_for_concept = self.config.get('min_salience_for_concept', 0.5)
    
    async def synthesize(self, memory: HolographicMemory) -> int:
        """
        Synthesize connections between morphons and mesh concepts
        
        Returns:
            Number of connections created
        """
        # Get the tenant's concept mesh
        mesh = ScopedConceptMesh.get_instance(
            memory.tenant_scope,
            memory.tenant_id
        )
        
        connections_created = 0
        
        # Process each morphon
        for morphon in memory.morphons:
            # Skip low salience morphons
            if morphon.salience < self.min_salience_for_concept:
                continue
            
            # Find or create matching concepts
            if morphon.modality == ModalityType.TEXT:
                connections = await self._link_text_morphon(morphon, mesh, memory)
                connections_created += connections
            
            elif morphon.modality in [ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO]:
                connections = await self._link_media_morphon(morphon, mesh, memory)
                connections_created += connections
        
        # Create cross-modal connections
        cross_modal = await self._create_cross_modal_strands(memory, mesh)
        connections_created += cross_modal
        
        logger.info(f"Synthesized {connections_created} connections for {len(memory.morphons)} morphons")
        return connections_created
    
    async def _link_text_morphon(self, morphon: PsiMorphon, mesh, memory: HolographicMemory) -> int:
        """Link text morphon to concepts"""
        connections = 0
        
        # Extract key terms from text
        terms = self._extract_terms(morphon.content)
        
        for term in terms:
            # Check if concept exists
            concept_id = mesh.name_index.get(term)
            
            if concept_id:
                # Link to existing concept
                strand = PsiStrand(
                    source_morphon_id=morphon.id,
                    target_morphon_id=f"concept_{concept_id}",
                    strand_type=StrandType.SEMANTIC,
                    strength=0.8,
                    evidence=f"Text contains term '{term}'"
                )
                memory.add_strand(strand)
                connections += 1
                
                # Update concept access count
                mesh.concepts[concept_id].access_count += 1
                
            elif self.auto_create_concepts:
                # Create new concept
                concept_id = mesh.add_concept(
                    name=term,
                    description=f"Extracted from: {morphon.content[:100]}...",
                    category="extracted",
                    importance=morphon.salience,
                    metadata={
                        "source_morphon": morphon.id,
                        "extraction_method": "text_analysis"
                    }
                )
                
                # Create bidirectional link
                strand = PsiStrand(
                    source_morphon_id=morphon.id,
                    target_morphon_id=f"concept_{concept_id}",
                    strand_type=StrandType.DESCRIBES,
                    strength=0.9,
                    bidirectional=True,
                    evidence=f"Concept '{term}' extracted from morphon"
                )
                memory.add_strand(strand)
                connections += 1
                
                # Generate proof for new concept
                await create_tenant_proof(
                    memory.tenant_scope,
                    memory.tenant_id,
                    concept_id,
                    {"term": term, "source": "morphon_extraction"}
                )
        
        return connections
    
    async def _link_media_morphon(self, morphon: PsiMorphon, mesh, memory: HolographicMemory) -> int:
        """Link media morphon to concepts using embeddings"""
        connections = 0
        
        if morphon.embedding is None:
            return connections
        
        # Find similar concepts by embedding
        similar_concepts = self._find_similar_concepts(morphon.embedding, mesh)
        
        for concept_id, similarity in similar_concepts:
            if similarity > self.visual_threshold:
                # Determine strand type based on modality
                if morphon.modality == ModalityType.IMAGE:
                    strand_type = StrandType.VISUAL_INSTANCE
                elif morphon.modality == ModalityType.AUDIO:
                    strand_type = StrandType.AUDIO_INSTANCE
                else:
                    strand_type = StrandType.SEMANTIC
                
                strand = PsiStrand(
                    source_morphon_id=f"concept_{concept_id}",
                    target_morphon_id=morphon.id,
                    strand_type=strand_type,
                    strength=similarity,
                    evidence=f"{morphon.modality.value} similarity: {similarity:.2f}"
                )
                memory.add_strand(strand)
                connections += 1
        
        # If no similar concepts and auto-create enabled
        if not similar_concepts and self.auto_create_concepts:
            # Create concept from media
            concept_name = self._generate_concept_name(morphon)
            
            concept_id = mesh.add_concept(
                name=concept_name,
                description=f"Media concept from {morphon.modality.value}",
                category=morphon.modality.value,
                importance=morphon.salience,
                embedding=morphon.embedding,
                metadata={
                    "source_morphon": morphon.id,
                    "modality": morphon.modality.value
                }
            )
            
            strand = PsiStrand(
                source_morphon_id=f"concept_{concept_id}",
                target_morphon_id=morphon.id,
                strand_type=StrandType.SEMANTIC,
                strength=1.0,
                bidirectional=True,
                evidence=f"Concept created from {morphon.modality.value} morphon"
            )
            memory.add_strand(strand)
            connections += 1
        
        return connections
    
    async def _create_cross_modal_strands(self, memory: HolographicMemory, mesh) -> int:
        """Create strands between morphons of different modalities"""
        connections = 0
        
        # Group morphons by modality
        by_modality = {}
        for morphon in memory.morphons:
            if morphon.modality not in by_modality:
                by_modality[morphon.modality] = []
            by_modality[morphon.modality].append(morphon)
        
        # Connect text to visual/audio based on temporal proximity
        if ModalityType.TEXT in by_modality:
            for text_morphon in by_modality[ModalityType.TEXT]:
                if text_morphon.temporal_index is None:
                    continue
                
                # Check images
                if ModalityType.IMAGE in by_modality:
                    for image_morphon in by_modality[ModalityType.IMAGE]:
                        if image_morphon.temporal_index is None:
                            continue
                        
                        time_diff = abs(text_morphon.temporal_index - image_morphon.temporal_index)
                        if time_diff < self.temporal_window:
                            strand = PsiStrand(
                                source_morphon_id=text_morphon.id,
                                target_morphon_id=image_morphon.id,
                                strand_type=StrandType.DESCRIBES,
                                strength=max(0.3, 1.0 - time_diff / self.temporal_window),
                                evidence=f"Temporal alignment: {time_diff:.1f}s",
                                temporal_offset=time_diff
                            )
                            memory.add_strand(strand)
                            connections += 1
        
        # Connect similar embeddings across modalities
        morphons_with_embeddings = [m for m in memory.morphons if m.embedding is not None]
        
        for i, morphon1 in enumerate(morphons_with_embeddings):
            for morphon2 in morphons_with_embeddings[i+1:]:
                if morphon1.modality != morphon2.modality:
                    similarity = self._compute_similarity(morphon1.embedding, morphon2.embedding)
                    
                    if similarity > self.semantic_threshold:
                        strand = PsiStrand(
                            source_morphon_id=morphon1.id,
                            target_morphon_id=morphon2.id,
                            strand_type=StrandType.SYNESTHETIC,
                            strength=similarity,
                            bidirectional=True,
                            evidence=f"Cross-modal similarity: {similarity:.2f}"
                        )
                        memory.add_strand(strand)
                        connections += 1
        
        return connections
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple extraction - in production, use NLP
        import re
        
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        terms = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Return unique terms
        return list(set(terms))[:10]  # Limit to 10 terms
    
    def _find_similar_concepts(self, embedding: np.ndarray, mesh) -> List[Tuple[str, float]]:
        """Find concepts with similar embeddings"""
        similar = []
        
        for concept_id, concept in mesh.concepts.items():
            if concept.embedding is not None:
                similarity = self._compute_similarity(embedding, concept.embedding)
                if similarity > self.semantic_threshold:
                    similar.append((concept_id, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:5]  # Top 5
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        # Reshape for sklearn
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        
        # Compute similarity
        similarity = cosine_similarity(e1, e2)[0, 0]
        return float(similarity)
    
    def _generate_concept_name(self, morphon: PsiMorphon) -> str:
        """Generate a concept name from morphon"""
        if morphon.modality == ModalityType.IMAGE:
            return f"visual_{morphon.id[:8]}"
        elif morphon.modality == ModalityType.AUDIO:
            return f"audio_{morphon.id[:8]}"
        elif morphon.modality == ModalityType.VIDEO:
            return f"video_{morphon.id[:8]}"
        else:
            return f"concept_{morphon.id[:8]}"
