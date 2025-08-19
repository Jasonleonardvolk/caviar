"""
Group-Scoped ConceptMesh Extension
Adds multi-tenancy support for Phase 3 preparation
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading

from python.core.concept_mesh import ConceptMesh

logger = logging.getLogger(__name__)

class ScopedConceptMesh:
    """
    Factory and manager for scoped ConceptMesh instances
    Supports user-scoped and group-scoped mesh isolation
    """
    
    _instances: Dict[str, ConceptMesh] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, scope: str, scope_id: str, 
                    config: Optional[Dict[str, Any]] = None) -> ConceptMesh:
        """
        Get or create a scoped ConceptMesh instance
        
        Args:
            scope: Either "user" or "group"
            scope_id: The user_id or group_id
            config: Optional configuration overrides
            
        Returns:
            ConceptMesh instance for the given scope
        """
        # Create scope key
        scope_key = f"{scope}_{scope_id}"
        
        # Check if instance exists
        if scope_key in cls._instances:
            return cls._instances[scope_key]
        
        # Create new instance with lock
        with cls._lock:
            # Double-check pattern
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
            
            # Create instance
            instance = ConceptMesh(scoped_config)
            cls._instances[scope_key] = instance
            
            logger.info(f"✅ Created {scope}-scoped ConceptMesh for {scope_id}")
            return instance
    
    @classmethod
    def get_user_mesh(cls, user_id: str, config: Optional[Dict[str, Any]] = None) -> ConceptMesh:
        """Convenience method to get user-scoped mesh"""
        return cls.get_instance("user", user_id, config)
    
    @classmethod
    def get_group_mesh(cls, group_id: str, config: Optional[Dict[str, Any]] = None) -> ConceptMesh:
        """Convenience method to get group-scoped mesh"""
        return cls.get_instance("group", group_id, config)
    
    @classmethod
    def list_instances(cls) -> Dict[str, str]:
        """List all active mesh instances"""
        return {
            key: f"{mesh.config.get('scope', 'unknown')} ({len(mesh.concepts)} concepts)"
            for key, mesh in cls._instances.items()
        }
    
    @classmethod
    def unload_instance(cls, scope: str, scope_id: str) -> bool:
        """
        Unload a mesh instance from memory (saves to disk first)
        
        Args:
            scope: Either "user" or "group"
            scope_id: The user_id or group_id
            
        Returns:
            True if unloaded, False if not found
        """
        scope_key = f"{scope}_{scope_id}"
        
        with cls._lock:
            if scope_key in cls._instances:
                # Save before unloading
                mesh = cls._instances[scope_key]
                mesh._save_mesh()  # Assuming ConceptMesh has this method
                
                # Remove from instances
                del cls._instances[scope_key]
                logger.info(f"📤 Unloaded {scope}-scoped mesh for {scope_id}")
                return True
        
        return False
    
    @classmethod
    def merge_meshes(cls, source_scope: str, source_id: str,
                    target_scope: str, target_id: str,
                    merge_strategy: str = "union") -> bool:
        """
        Merge one mesh into another (for group joins, etc.)
        
        Args:
            source_scope: Source mesh scope
            source_id: Source mesh ID
            target_scope: Target mesh scope 
            target_id: Target mesh ID
            merge_strategy: How to handle conflicts ("union", "replace", "skip")
            
        Returns:
            True if successful
        """
        source_mesh = cls.get_instance(source_scope, source_id)
        target_mesh = cls.get_instance(target_scope, target_id)
        
        if not source_mesh or not target_mesh:
            logger.error("One or both meshes not found for merge")
            return False
        
        # Merge concepts
        merged_count = 0
        for concept_id, concept in source_mesh.concepts.items():
            if concept.name not in target_mesh.name_index:
                # Add new concept
                target_mesh.add_concept(
                    name=concept.name,
                    description=concept.description,
                    category=concept.category,
                    importance=concept.importance,
                    embedding=concept.embedding,
                    metadata={
                        **concept.metadata,
                        'merged_from': f"{source_scope}_{source_id}",
                        'merge_timestamp': datetime.now().isoformat()
                    }
                )
                merged_count += 1
            elif merge_strategy == "replace":
                # Replace existing concept
                existing_id = target_mesh.name_index[concept.name]
                target_mesh.concepts[existing_id] = concept
                merged_count += 1
            # "skip" strategy does nothing for existing concepts
        
        # Merge relations
        for relation in source_mesh.relations:
            # Only add if both concepts exist in target
            if (relation.source_id in target_mesh.concepts and 
                relation.target_id in target_mesh.concepts):
                target_mesh.add_relation(
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_type=relation.relation_type,
                    strength=relation.strength,
                    bidirectional=relation.bidirectional,
                    metadata={
                        **relation.metadata,
                        'merged_from': f"{source_scope}_{source_id}"
                    }
                )
        
        logger.info(f"✅ Merged {merged_count} concepts from {source_scope}_{source_id} "
                   f"to {target_scope}_{target_id}")
        return True


class GroupMemoryBridge:
    """
    Bridge between GroupManager and ConceptMesh for group memory operations
    """
    
    def __init__(self):
        self.scoped_mesh = ScopedConceptMesh
        
    def initialize_group_memory(self, group_id: str, owner_id: str) -> bool:
        """Initialize memory space for a new group"""
        try:
            # Get group mesh
            group_mesh = self.scoped_mesh.get_group_mesh(group_id)
            
            # Add initial metadata concept
            group_mesh.add_concept(
                name=f"_group_metadata_{group_id}",
                description=f"Metadata for group {group_id}",
                category="system",
                metadata={
                    "group_id": group_id,
                    "owner_id": owner_id,
                    "created_at": datetime.now().isoformat(),
                    "type": "group_metadata"
                }
            )
            
            logger.info(f"✅ Initialized group memory for {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize group memory: {e}")
            return False
    
    def add_member_concepts(self, group_id: str, user_id: str, 
                          merge_personal: bool = False) -> bool:
        """
        Add a member's concepts to group (optionally merge personal mesh)
        
        Args:
            group_id: Target group ID
            user_id: User joining the group
            merge_personal: If True, merge user's personal mesh into group
        """
        if not merge_personal:
            # Just record membership in group mesh
            group_mesh = self.scoped_mesh.get_group_mesh(group_id)
            group_mesh.add_concept(
                name=f"_member_{user_id}",
                description=f"Member {user_id} joined the group",
                category="system",
                metadata={
                    "user_id": user_id,
                    "joined_at": datetime.now().isoformat(),
                    "type": "membership"
                }
            )
            return True
        
        # Merge personal mesh into group
        return self.scoped_mesh.merge_meshes(
            source_scope="user",
            source_id=user_id,
            target_scope="group", 
            target_id=group_id,
            merge_strategy="union"
        )
    
    def remove_member_concepts(self, group_id: str, user_id: str,
                             remove_contributions: bool = False) -> bool:
        """
        Handle member removal from group memory
        
        Args:
            group_id: Group ID
            user_id: User leaving the group
            remove_contributions: If True, remove concepts added by this user
        """
        group_mesh = self.scoped_mesh.get_group_mesh(group_id)
        
        if not remove_contributions:
            # Just mark as left
            group_mesh.add_concept(
                name=f"_member_left_{user_id}_{datetime.now().timestamp()}",
                description=f"Member {user_id} left the group",
                category="system",
                metadata={
                    "user_id": user_id,
                    "left_at": datetime.now().isoformat(),
                    "type": "membership_end"
                }
            )
            return True
        
        # Remove concepts added by this user
        concepts_to_remove = []
        for concept_id, concept in group_mesh.concepts.items():
            if concept.metadata.get('added_by') == user_id:
                concepts_to_remove.append(concept_id)
        
        for concept_id in concepts_to_remove:
            group_mesh.remove_concept(concept_id)
        
        logger.info(f"Removed {len(concepts_to_remove)} concepts from {user_id}")
        return True
    
    def get_group_stats(self, group_id: str) -> Dict[str, Any]:
        """Get statistics about group memory"""
        try:
            group_mesh = self.scoped_mesh.get_group_mesh(group_id)
            
            # Category breakdown
            category_counts = {}
            for concept in group_mesh.concepts.values():
                category_counts[concept.category] = category_counts.get(concept.category, 0) + 1
            
            return {
                "total_concepts": len(group_mesh.concepts),
                "total_relations": len(group_mesh.relations),
                "categories": category_counts,
                "diff_history_size": len(group_mesh.diff_history),
                "most_accessed": self._get_most_accessed(group_mesh, limit=5)
            }
        except Exception as e:
            logger.error(f"Failed to get group stats: {e}")
            return {}
    
    def _get_most_accessed(self, mesh: ConceptMesh, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most accessed concepts in a mesh"""
        sorted_concepts = sorted(
            mesh.concepts.values(),
            key=lambda c: c.access_count,
            reverse=True
        )[:limit]
        
        return [
            {
                "name": c.name,
                "access_count": c.access_count,
                "category": c.category
            }
            for c in sorted_concepts
        ]


# Import for enhanced mesh methods
from datetime import datetime

# Make the bridge available as a singleton
_group_memory_bridge = None

def get_group_memory_bridge() -> GroupMemoryBridge:
    """Get singleton instance of GroupMemoryBridge"""
    global _group_memory_bridge
    if _group_memory_bridge is None:
        _group_memory_bridge = GroupMemoryBridge()
    return _group_memory_bridge
