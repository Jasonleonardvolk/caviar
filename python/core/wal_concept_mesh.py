"""
Enhanced ConceptMesh with WAL Integration
Extends the base ConceptMesh with Write-Ahead Logging for durability
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import base classes
from python.core.concept_mesh import ConceptMesh, Concept, ConceptRelation, ConceptDiff
from python.core.scoped_wal import WALManager
from python.core.scoped_concept_mesh import ScopedConceptMesh

logger = logging.getLogger(__name__)

class WALEnabledConceptMesh(ConceptMesh):
    """
    ConceptMesh with Write-Ahead Logging for crash recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Extract scope information
        self.scope = config.get('scope', 'user')
        self.scope_id = config.get('scope_id', 'default')
        
        # Initialize WAL
        self.wal = WALManager.get_wal(self.scope, self.scope_id)
        
        # Replay WAL to restore state
        self._replay_wal()
        
        logger.info(f"✅ WAL-enabled ConceptMesh initialized for {self.scope}:{self.scope_id}")
    
    def _replay_wal(self):
        """Replay WAL entries to restore mesh state"""
        def replay_operation(operation: str, data: Dict[str, Any]):
            try:
                if operation == "add_concept":
                    # Bypass WAL write during replay
                    self._add_concept_internal(
                        name=data['name'],
                        description=data.get('description', ''),
                        category=data.get('category', 'general'),
                        importance=data.get('importance', 1.0),
                        embedding=data.get('embedding'),
                        metadata=data.get('metadata', {})
                    )
                elif operation == "remove_concept":
                    self._remove_concept_internal(data['concept_id'])
                elif operation == "add_relation":
                    self._add_relation_internal(
                        source_id=data['source_id'],
                        target_id=data['target_id'],
                        relation_type=data['relation_type'],
                        strength=data.get('strength', 1.0),
                        bidirectional=data.get('bidirectional', False),
                        metadata=data.get('metadata', {})
                    )
                elif operation == "remove_relation":
                    self._remove_relation_internal(
                        source_id=data['source_id'],
                        target_id=data['target_id']
                    )
                elif operation == "update_concept":
                    self._update_concept_internal(
                        concept_id=data['concept_id'],
                        updates=data['updates']
                    )
            except Exception as e:
                logger.error(f"Failed to replay operation {operation}: {e}")
        
        # Replay all WAL entries
        replayed = self.wal.replay(replay_operation)
        if replayed > 0:
            logger.info(f"Replayed {replayed} operations from WAL")
    
    # Override public methods to add WAL logging
    
    def add_concept(self, name: str, description: str = "", category: str = "general",
                   importance: float = 1.0, embedding: Optional[Any] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add concept with WAL logging"""
        # Write to WAL first
        wal_data = {
            'name': name,
            'description': description,
            'category': category,
            'importance': importance,
            'embedding': embedding.tolist() if embedding is not None else None,
            'metadata': metadata or {}
        }
        self.wal.write("add_concept", wal_data)
        
        # Then perform operation
        return self._add_concept_internal(name, description, category, importance, embedding, metadata)
    
    def _add_concept_internal(self, name: str, description: str = "", category: str = "general",
                            importance: float = 1.0, embedding: Optional[Any] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Internal add without WAL logging (for replay)"""
        return super().add_concept(name, description, category, importance, embedding, metadata)
    
    def remove_concept(self, concept_id: str) -> bool:
        """Remove concept with WAL logging"""
        if concept_id not in self.concepts:
            return False
        
        # Write to WAL first
        self.wal.write("remove_concept", {'concept_id': concept_id})
        
        # Then perform operation
        return self._remove_concept_internal(concept_id)
    
    def _remove_concept_internal(self, concept_id: str) -> bool:
        """Internal remove without WAL logging"""
        return super().remove_concept(concept_id)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                    strength: float = 1.0, bidirectional: bool = False,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add relation with WAL logging"""
        # Write to WAL first
        wal_data = {
            'source_id': source_id,
            'target_id': target_id,
            'relation_type': relation_type,
            'strength': strength,
            'bidirectional': bidirectional,
            'metadata': metadata or {}
        }
        self.wal.write("add_relation", wal_data)
        
        # Then perform operation
        return self._add_relation_internal(source_id, target_id, relation_type,
                                         strength, bidirectional, metadata)
    
    def _add_relation_internal(self, source_id: str, target_id: str, relation_type: str,
                             strength: float = 1.0, bidirectional: bool = False,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Internal add relation without WAL logging"""
        # Check if concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            logger.warning(f"Cannot add relation: concepts not found")
            return False
        
        # Create relation
        relation = ConceptRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            metadata=metadata or {}
        )
        
        self.relations.append(relation)
        
        # Update graph
        self.graph.add_edge(source_id, target_id, 
                          relation_type=relation_type,
                          strength=strength,
                          **relation.metadata)
        
        if bidirectional:
            self.graph.add_edge(target_id, source_id,
                              relation_type=relation_type,
                              strength=strength,
                              **relation.metadata)
        
        # Record diff
        diff = ConceptDiff(
            id=f"diff_{datetime.now().timestamp()}",
            diff_type="relate",
            concepts=[source_id, target_id],
            new_value={'relation': relation_type, 'strength': strength}
        )
        self._record_diff(diff)
        
        return True
    
    def save_checkpoint(self):
        """Force a checkpoint of the current state"""
        # Save mesh to disk
        self._save_mesh()
        
        # Create WAL checkpoint
        self.wal._create_checkpoint()
        
        logger.info(f"Created checkpoint for {self.scope}:{self.scope_id}")
    
    def get_wal_stats(self) -> Dict[str, Any]:
        """Get WAL statistics for this mesh"""
        return self.wal.get_stats()


# Update the ScopedConceptMesh factory to use WAL-enabled meshes
def update_scoped_factory():
    """Monkey-patch the ScopedConceptMesh to use WAL-enabled meshes"""
    original_get_instance = ScopedConceptMesh.get_instance
    
    @classmethod
    def get_instance_with_wal(cls, scope: str, scope_id: str,
                            config: Optional[Dict[str, Any]] = None) -> ConceptMesh:
        """Get or create a WAL-enabled scoped ConceptMesh instance"""
        scope_key = f"{scope}_{scope_id}"
        
        if scope_key in cls._instances:
            return cls._instances[scope_key]
        
        with cls._lock:
            if scope_key in cls._instances:
                return cls._instances[scope_key]
            
            # Build scoped configuration
            base_config = config or {}
            scoped_config = {
                **base_config,
                'scope': scope,
                'scope_id': scope_id,
                'storage_path': f'data/concept_mesh/{scope}/{scope_id}',
                'storage_key': f'{scope}_mesh_{scope_id}'
            }
            
            # Create WAL-enabled instance
            instance = WALEnabledConceptMesh(scoped_config)
            cls._instances[scope_key] = instance
            
            logger.info(f"✅ Created WAL-enabled {scope}-scoped ConceptMesh for {scope_id}")
            return instance
    
    # Replace the method
    ScopedConceptMesh.get_instance = get_instance_with_wal
    logger.info("✅ Updated ScopedConceptMesh factory to use WAL-enabled meshes")


# Import for missing datetime
from datetime import datetime

# Apply the update when module is imported
update_scoped_factory()
