"""
Unified ID Generator for TORI/KHA Memory Systems
=================================================

Provides a single, consistent ID format across all memory systems:
- UnifiedMemoryVault
- SolitonMemory
- FractalSolitonMemory
- Ingest Service

ID Format: "tori_<type>_<timestamp>_<hash>"
Example: "tori_mem_1704067200_a1b2c3d4"
"""

import hashlib
import time
import uuid
from typing import Optional, Dict, Any
from enum import Enum

class MemorySystemType(Enum):
    """Types of memory systems"""
    MEMORY = "mem"      # UnifiedMemoryVault
    SOLITON = "sol"     # SolitonMemory  
    FRACTAL = "frc"     # FractalSolitonMemory
    INGEST = "ing"      # Ingest Service
    CONCEPT = "con"     # ConceptMesh
    GENERIC = "gen"     # Generic/Unknown

class UnifiedIDGenerator:
    """
    Singleton ID generator that ensures consistent IDs across all TORI systems.
    
    Format: tori_<type>_<timestamp>_<hash>
    - tori: System prefix for easy identification
    - type: 3-letter code for the originating system
    - timestamp: Unix timestamp (seconds)
    - hash: 8-character hash for uniqueness
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.counter = 0
        
    def generate_id(
        self, 
        system_type: MemorySystemType = MemorySystemType.GENERIC,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a unified ID for any memory system.
        
        Args:
            system_type: The type of system generating the ID
            content: Optional content to include in hash
            metadata: Optional metadata to include in hash
            
        Returns:
            Unified ID string
        """
        # Get current timestamp
        timestamp = int(time.time())
        
        # Create hash components
        hash_parts = []
        
        # Add content hash if provided
        if content is not None:
            hash_parts.append(str(content))
            
        # Add metadata hash if provided
        if metadata is not None:
            hash_parts.append(str(sorted(metadata.items())))
            
        # Add counter for uniqueness
        self.counter += 1
        hash_parts.append(str(self.counter))
        
        # Add random component
        hash_parts.append(str(uuid.uuid4()))
        
        # Create hash
        combined = "|".join(hash_parts)
        full_hash = hashlib.sha256(combined.encode()).hexdigest()
        short_hash = full_hash[:8]
        
        # Construct ID
        unified_id = f"tori_{system_type.value}_{timestamp}_{short_hash}"
        
        return unified_id
    
    def parse_id(self, unified_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse a unified ID into its components.
        
        Args:
            unified_id: The ID to parse
            
        Returns:
            Dictionary with parsed components or None if invalid
        """
        try:
            parts = unified_id.split("_")
            
            if len(parts) != 4 or parts[0] != "tori":
                return None
                
            return {
                "prefix": parts[0],
                "system_type": parts[1],
                "timestamp": int(parts[2]),
                "hash": parts[3],
                "full_id": unified_id
            }
        except:
            return None
    
    def is_valid_unified_id(self, id_string: str) -> bool:
        """Check if a string is a valid unified ID"""
        parsed = self.parse_id(id_string)
        return parsed is not None
    
    def convert_legacy_id(self, legacy_id: str, system_type: MemorySystemType) -> str:
        """
        Convert a legacy ID to unified format.
        
        Args:
            legacy_id: The old format ID
            system_type: The system that generated the legacy ID
            
        Returns:
            New unified ID
        """
        # Use the legacy ID as part of the hash to maintain some continuity
        timestamp = int(time.time())
        hash_source = f"{legacy_id}:{system_type.value}"
        hash_value = hashlib.sha256(hash_source.encode()).hexdigest()[:8]
        
        return f"tori_{system_type.value}_{timestamp}_{hash_value}"

# Global singleton instance
ID_GENERATOR = UnifiedIDGenerator.get_instance()

# Convenience functions
def generate_memory_id(content=None, metadata=None) -> str:
    """Generate ID for UnifiedMemoryVault"""
    return ID_GENERATOR.generate_id(MemorySystemType.MEMORY, content, metadata)

def generate_soliton_id(content=None, metadata=None) -> str:
    """Generate ID for SolitonMemory"""
    return ID_GENERATOR.generate_id(MemorySystemType.SOLITON, content, metadata)

def generate_fractal_id(content=None, metadata=None) -> str:
    """Generate ID for FractalSolitonMemory"""
    return ID_GENERATOR.generate_id(MemorySystemType.FRACTAL, content, metadata)

def generate_ingest_id(content=None, metadata=None) -> str:
    """Generate ID for Ingest Service"""
    return ID_GENERATOR.generate_id(MemorySystemType.INGEST, content, metadata)

def generate_concept_id(content=None, metadata=None) -> str:
    """Generate ID for ConceptMesh"""
    return ID_GENERATOR.generate_id(MemorySystemType.CONCEPT, content, metadata)

def generate_generic_id(content=None, metadata=None) -> str:
    """Generate generic ID"""
    return ID_GENERATOR.generate_id(MemorySystemType.GENERIC, content, metadata)

# Example usage
if __name__ == "__main__":
    # Test ID generation
    print("Testing Unified ID Generator:")
    print("-" * 50)
    
    # Generate different types of IDs
    mem_id = generate_memory_id("test content", {"user": "test"})
    sol_id = generate_soliton_id("soliton wave", {"amplitude": 0.8})
    frc_id = generate_fractal_id("fractal pattern", {"lattice_size": 100})
    ing_id = generate_ingest_id("document.pdf", {"session": "abc123"})
    con_id = generate_concept_id("consciousness", {"category": "cognitive"})
    
    print(f"Memory ID: {mem_id}")
    print(f"Soliton ID: {sol_id}")
    print(f"Fractal ID: {frc_id}")
    print(f"Ingest ID: {ing_id}")
    print(f"Concept ID: {con_id}")
    
    print("\nParsing IDs:")
    print("-" * 50)
    
    for id_str in [mem_id, sol_id, frc_id]:
        parsed = ID_GENERATOR.parse_id(id_str)
        if parsed:
            print(f"ID: {id_str}")
            print(f"  System: {parsed['system_type']}")
            print(f"  Timestamp: {parsed['timestamp']}")
            print(f"  Hash: {parsed['hash']}")
            print()
    
    print("Legacy ID Conversion:")
    print("-" * 50)
    
    # Convert legacy IDs
    legacy_vault = "a1b2c3d4e5f6g7h8"
    legacy_soliton = "soliton_1704067200_1234"
    legacy_ingest = "ingested_concept_123_1704067200"
    
    new_vault = ID_GENERATOR.convert_legacy_id(legacy_vault, MemorySystemType.MEMORY)
    new_soliton = ID_GENERATOR.convert_legacy_id(legacy_soliton, MemorySystemType.SOLITON)
    new_ingest = ID_GENERATOR.convert_legacy_id(legacy_ingest, MemorySystemType.INGEST)
    
    print(f"Legacy Vault: {legacy_vault} -> {new_vault}")
    print(f"Legacy Soliton: {legacy_soliton} -> {new_soliton}")
    print(f"Legacy Ingest: {legacy_ingest} -> {new_ingest}")
