"""
Concept Mesh Stub - Fallback implementation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

logger = logging.getLogger(__name__)

class ConceptMeshStub:
    """Minimal concept mesh implementation for fallback"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config: Dict[str, Any] = None):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config or {})
        return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'data/concept_mesh'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.concepts = {}
        self.concept_file = self.storage_path / 'concepts.json'
        
        # Load existing concepts
        if self.concept_file.exists():
            try:
                with open(self.concept_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.concepts = data.get('concepts', {})
                    elif isinstance(data, list):
                        # Convert list to dict
                        self.concepts = {f"concept_{i}": c for i, c in enumerate(data)}
            except Exception as e:
                logger.error(f"Failed to load concepts: {e}")
        
        logger.info(f"✅ ConceptMeshStub initialized with {len(self.concepts)} concepts")
    
    def add_concept(self, concept_id: str, name: str, **kwargs):
        """Add a concept to the mesh"""
        self.concepts[concept_id] = {
            'id': concept_id,
            'name': name,
            **kwargs
        }
        self._save()
        return concept_id
    
    def get_concept(self, concept_id: str):
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def search_concepts(self, query: str, limit: int = 10):
        """Simple search by name"""
        results = []
        query_lower = query.lower()
        
        for concept_id, concept in self.concepts.items():
            if query_lower in concept.get('name', '').lower():
                results.append(concept)
                if len(results) >= limit:
                    break
        
        return results
    
    def _save(self):
        """Save concepts to disk"""
        try:
            data = {
                'concepts': self.concepts,
                'version': '1.0'
            }
            with open(self.concept_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save concepts: {e}")

# Global instance getter
def get_mesh_instance(config: Dict[str, Any] = None):
    """Get the concept mesh instance"""
    return ConceptMeshStub.get_instance(config)
