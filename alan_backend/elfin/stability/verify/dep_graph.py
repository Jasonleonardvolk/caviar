"""
Dependency graph for tracking verification proof dependencies.

This module provides a dependency graph that tracks the relationships
between ELFIN entities (concepts, Lyapunov networks, files) and their
verification proofs. It allows for efficient tracking of which proofs
need to be reverified when entities change.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, Iterable, List


class DepGraph:
    """
    Maps proof hashes ⇆ ELFIN entity IDs (concepts, Lyap nets, files).
    
    The dependency graph allows for quick determination of which proofs
    are *dirty* (need to be reverified) after an entity changes.
    
    Attributes:
        parents: Mapping from proof hash to the set of entities it depends on
        children: Mapping from entity ID to the set of proofs that depend on it
        _dirty: Set of proof hashes that need to be reverified
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self.parents: Dict[str, Set[str]] = defaultdict(set)   # proof → {entity}
        self.children: Dict[str, Set[str]] = defaultdict(set)  # entity → {proof}
        self._dirty: Set[str] = set()

    # ---------- registration -------------------------------------------
    def add_edge(self, proof_hash: str, entities: Iterable[str]) -> None:
        """
        Register a dependency between a proof and one or more entities.
        
        Args:
            proof_hash: Hash of the proof
            entities: Iterable of entity IDs that the proof depends on
        """
        for e in entities:
            self.parents[proof_hash].add(e)
            self.children[e].add(proof_hash)

    # ---------- mark / query -------------------------------------------
    def mark_dirty(self, entity_id: str) -> None:
        """
        Mark all proofs that depend on an entity as dirty.
        
        Args:
            entity_id: ID of the entity that has changed
        """
        for proof in self.children.get(entity_id, ()):
            self._dirty.add(proof)
            
    def mark_dirty_many(self, entity_ids: Iterable[str]) -> None:
        """
        Mark all proofs that depend on any of the given entities as dirty.
        
        Args:
            entity_ids: Iterable of entity IDs that have changed
        """
        for entity_id in entity_ids:
            self.mark_dirty(entity_id)

    def mark_fresh(self, proof_hash: str) -> None:
        """
        Mark a proof as fresh (not dirty).
        
        Args:
            proof_hash: Hash of the proof that has been freshly verified
        """
        self._dirty.discard(proof_hash)
        
    def mark_fresh_many(self, proof_hashes: Iterable[str]) -> None:
        """
        Mark multiple proofs as fresh.
        
        Args:
            proof_hashes: Iterable of proof hashes that have been freshly verified
        """
        for proof_hash in proof_hashes:
            self.mark_fresh(proof_hash)

    def is_dirty(self, proof_hash: str) -> bool:
        """
        Check if a proof is dirty.
        
        Args:
            proof_hash: Hash of the proof to check
            
        Returns:
            True if the proof is dirty, False otherwise
        """
        return proof_hash in self._dirty

    def dirty_proofs(self) -> Set[str]:
        """
        Get all dirty proofs.
        
        Returns:
            Set of all dirty proof hashes
        """
        return set(self._dirty)
    
    def get_entity_dependencies(self, proof_hash: str) -> Set[str]:
        """
        Get all entities that a proof depends on.
        
        Args:
            proof_hash: Hash of the proof
            
        Returns:
            Set of entity IDs that the proof depends on
        """
        return set(self.parents.get(proof_hash, set()))
    
    def get_proof_dependencies(self, entity_id: str) -> Set[str]:
        """
        Get all proofs that depend on an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Set of proof hashes that depend on the entity
        """
        return set(self.children.get(entity_id, set()))
    
    def clear(self) -> None:
        """Clear all dependency information."""
        self.parents.clear()
        self.children.clear()
        self._dirty.clear()
