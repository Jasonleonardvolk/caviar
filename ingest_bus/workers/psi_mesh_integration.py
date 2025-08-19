"""
ψMesh Integration Module for TORI Document Ingestion
Provides semantic association mesh and extraction verification

Integrates with ConceptMesh and Ghost Collective for deep semantic understanding
"""

import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("tori-ingest.psi_mesh")

class PsiMeshIntegrator:
    """
    ψMesh (PsiMesh) - Semantic Association Mesh
    
    Creates and maintains semantic associations between concepts
    Provides extraction integrity verification
    """
    
    def __init__(self):
        self.associations = {}  # concept_id -> list of associated concepts
        self.semantic_vectors = {}  # concept_id -> vector embedding
        self.confidence_scores = {}  # concept_id -> confidence score
        self.verification_threshold = 0.85
        
        # Load existing associations if available
        self._load_associations()
        
        logger.info("ψMesh Integrator initialized")
    
    def _load_associations(self):
        """Load existing ψMesh associations from storage"""
        try:
            psi_mesh_path = Path(__file__).parent.parent.parent / "concept-mesh-data" / "psi_mesh.json"
            if psi_mesh_path.exists():
                with open(psi_mesh_path, 'r') as f:
                    data = json.load(f)
                    self.associations = data.get('associations', {})
                    self.confidence_scores = data.get('confidence_scores', {})
                logger.info(f"Loaded {len(self.associations)} ψMesh associations")
        except Exception as e:
            logger.warning(f"Could not load existing ψMesh: {e}")
    
    def _save_associations(self):
        """Save ψMesh associations to storage"""
        try:
            psi_mesh_path = Path(__file__).parent.parent.parent / "concept-mesh-data" / "psi_mesh.json"
            psi_mesh_path.parent.mkdir(exist_ok=True)
            
            data = {
                'associations': self.associations,
                'confidence_scores': self.confidence_scores,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(psi_mesh_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("ψMesh associations saved")
        except Exception as e:
            logger.error(f"Error saving ψMesh: {e}")
    
    async def create_semantic_associations(self, concepts: List[Dict[str, Any]], 
                                         document_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create semantic associations between extracted concepts
        
        Args:
            concepts: List of extracted concepts with embeddings
            document_context: Context from the source document
            
        Returns:
            Dict containing association map and metrics
        """
        logger.info(f"Creating semantic associations for {len(concepts)} concepts")
        
        associations_created = 0
        concept_pairs = []
        
        # Create associations between concepts
        for i, concept_a in enumerate(concepts):
            concept_id_a = self._generate_concept_id(concept_a)
            
            # Store concept in ψMesh
            self.confidence_scores[concept_id_a] = concept_a.get('confidence', 0.8)
            
            for j, concept_b in enumerate(concepts[i+1:], i+1):
                concept_id_b = self._generate_concept_id(concept_b)
                
                # Calculate semantic similarity
                similarity = self._calculate_semantic_similarity(concept_a, concept_b)
                
                if similarity > 0.6:  # Threshold for creating association
                    # Create bidirectional association
                    if concept_id_a not in self.associations:
                        self.associations[concept_id_a] = []
                    if concept_id_b not in self.associations:
                        self.associations[concept_id_b] = []
                    
                    association_data = {
                        'target': concept_id_b,
                        'similarity': similarity,
                        'context': document_context.get('metadata', {}),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    self.associations[concept_id_a].append(association_data)
                    
                    # Reverse association
                    reverse_association = association_data.copy()
                    reverse_association['target'] = concept_id_a
                    self.associations[concept_id_b].append(reverse_association)
                    
                    concept_pairs.append((concept_a['name'], concept_b['name'], similarity))
                    associations_created += 1
        
        # Save updated associations
        self._save_associations()
        
        logger.info(f"Created {associations_created} semantic associations")
        
        return {
            'associations_created': associations_created,
            'concept_pairs': concept_pairs,
            'total_concepts': len(concepts),
            'average_connectivity': associations_created / len(concepts) if concepts else 0
        }
    
    async def verify_extraction_integrity(self, extracted_concepts: List[Dict[str, Any]], 
                                        original_content: str, 
                                        document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that extracted concepts are faithful to the original content
        
        Args:
            extracted_concepts: Concepts extracted from document
            original_content: Original document text
            document_metadata: Document metadata for context
            
        Returns:
            Verification results with integrity scores
        """
        logger.info(f"Verifying extraction integrity for {len(extracted_concepts)} concepts")
        
        verification_results = {
            'verified_concepts': [],
            'flagged_concepts': [],
            'integrity_score': 0.0,
            'verification_details': {}
        }
        
        content_lower = original_content.lower()
        total_score = 0.0
        
        for concept in extracted_concepts:
            concept_name = concept.get('name', '')
            keywords = concept.get('keywords', [])
            context = concept.get('context', '')
            
            # Verification checks
            checks = {
                'name_in_content': self._verify_concept_in_text(concept_name, content_lower),
                'keywords_in_content': self._verify_keywords_in_text(keywords, content_lower),
                'context_matches': self._verify_context_match(context, content_lower),
                'semantic_coherence': self._verify_semantic_coherence(concept, original_content)
            }
            
            # Calculate concept integrity score
            concept_score = sum(checks.values()) / len(checks)
            total_score += concept_score
            
            concept_verification = {
                'concept': concept_name,
                'integrity_score': concept_score,
                'checks': checks,
                'verified': concept_score >= self.verification_threshold
            }
            
            if concept_score >= self.verification_threshold:
                verification_results['verified_concepts'].append(concept_verification)
            else:
                verification_results['flagged_concepts'].append(concept_verification)
                logger.warning(f"Concept '{concept_name}' failed verification (score: {concept_score:.2f})")
        
        # Overall integrity score
        verification_results['integrity_score'] = total_score / len(extracted_concepts) if extracted_concepts else 0.0
        
        # Additional verification metrics
        verification_results['verification_details'] = {
            'total_concepts': len(extracted_concepts),
            'verified_count': len(verification_results['verified_concepts']),
            'flagged_count': len(verification_results['flagged_concepts']),
            'verification_threshold': self.verification_threshold,
            'document_type': document_metadata.get('file_type', 'unknown'),
            'verified_at': datetime.now().isoformat()
        }
        
        logger.info(f"Verification complete: {verification_results['integrity_score']:.2f} integrity score")
        
        return verification_results
    
    def _generate_concept_id(self, concept: Dict[str, Any]) -> str:
        """Generate unique ID for concept"""
        import hashlib
        concept_str = f"{concept.get('name', '')}{concept.get('keywords', [])}"
        return hashlib.md5(concept_str.encode()).hexdigest()[:12]
    
    def _calculate_semantic_similarity(self, concept_a: Dict[str, Any], 
                                     concept_b: Dict[str, Any]) -> float:
        """Calculate semantic similarity between two concepts"""
        # Simple keyword overlap for now - can be enhanced with embeddings
        keywords_a = set(concept_a.get('keywords', []))
        keywords_b = set(concept_b.get('keywords', []))
        
        if not keywords_a or not keywords_b:
            return 0.0
        
        intersection = len(keywords_a.intersection(keywords_b))
        union = len(keywords_a.union(keywords_b))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity if concepts appear in similar contexts
        context_a = concept_a.get('context', '').lower()
        context_b = concept_b.get('context', '').lower()
        
        if context_a and context_b:
            # Simple context similarity based on common words
            words_a = set(context_a.split())
            words_b = set(context_b.split())
            context_overlap = len(words_a.intersection(words_b)) / max(len(words_a), len(words_b))
            jaccard_similarity = (jaccard_similarity + context_overlap) / 2
        
        return jaccard_similarity
    
    def _verify_concept_in_text(self, concept_name: str, content: str) -> float:
        """Verify that concept name appears in original text"""
        if not concept_name:
            return 0.0
        
        # Check exact match
        if concept_name.lower() in content:
            return 1.0
        
        # Check partial matches (individual words)
        words = concept_name.lower().split()
        found_words = sum(1 for word in words if word in content)
        
        return found_words / len(words) if words else 0.0
    
    def _verify_keywords_in_text(self, keywords: List[str], content: str) -> float:
        """Verify that keywords appear in original text"""
        if not keywords:
            return 0.5  # Neutral score if no keywords
        
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in content)
        return found_keywords / len(keywords)
    
    def _verify_context_match(self, context: str, content: str) -> float:
        """Verify that context matches original content"""
        if not context:
            return 0.5  # Neutral score if no context
        
        # Check if context appears in content
        context_lower = context.lower()
        if context_lower in content:
            return 1.0
        
        # Check word overlap
        context_words = set(context_lower.split())
        content_words = set(content.split())
        
        if not context_words:
            return 0.5
        
        overlap = len(context_words.intersection(content_words))
        return overlap / len(context_words)
    
    def _verify_semantic_coherence(self, concept: Dict[str, Any], content: str) -> float:
        """Verify semantic coherence of concept with document"""
        # This is a simplified coherence check
        # In a full implementation, this would use advanced NLP
        
        concept_name = concept.get('name', '').lower()
        keywords = [k.lower() for k in concept.get('keywords', [])]
        
        # Check if concept and keywords form a coherent topic
        if not concept_name or not keywords:
            return 0.5
        
        # Simple coherence: are concept terms distributed throughout content?
        content_sections = content.lower().split('\n\n')
        sections_with_concept = 0
        
        for section in content_sections:
            if any(term in section for term in [concept_name] + keywords):
                sections_with_concept += 1
        
        # Coherence score based on distribution
        if len(content_sections) > 0:
            distribution_score = min(1.0, sections_with_concept / len(content_sections) * 2)
            return distribution_score
        
        return 0.5
    
    async def get_concept_associations(self, concept_id: str) -> List[Dict[str, Any]]:
        """Get all associations for a concept"""
        return self.associations.get(concept_id, [])
    
    async def find_similar_concepts(self, concept: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar concepts in ψMesh"""
        concept_id = self._generate_concept_id(concept)
        associations = self.associations.get(concept_id, [])
        
        # Sort by similarity score
        similar = [(assoc['target'], assoc['similarity']) for assoc in associations]
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar[:top_k]
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get ψMesh statistics"""
        total_concepts = len(self.associations)
        total_associations = sum(len(assocs) for assocs in self.associations.values())
        
        avg_connectivity = total_associations / total_concepts if total_concepts > 0 else 0
        
        return {
            'total_concepts': total_concepts,
            'total_associations': total_associations,
            'average_connectivity': avg_connectivity,
            'confidence_scores': len(self.confidence_scores)
        }

# Global ψMesh instance
psi_mesh = PsiMeshIntegrator()
