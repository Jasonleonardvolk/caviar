#!/usr/bin/env python3
"""memory_sculptor.py - Concept Storage with Vector Embeddings

This module implements the Memory Sculptor for vector embedding storage and search.
It handles storing concepts with FAISS indexing for semantic similarity search.
"""

from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import logging
import shutil
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Storage configuration
_STORE_DIR = Path("/opt/tori/mesh/hot")
_INDEX_PATH = _STORE_DIR / "faiss.index"
_INDEX_BACKUP_PATH = _STORE_DIR / "faiss.index.bak"
_METADATA_PATH = _STORE_DIR / "index_metadata.json"

# Create directories if they don't exist
_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize sentence transformer model
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

# Global FAISS index (loaded lazily)
_faiss_index = None
_index_metadata = {}

def _load_index():
    """Load or create the FAISS index."""
    global _faiss_index, _index_metadata
    
    if _INDEX_PATH.exists():
        try:
            _faiss_index = faiss.read_index(str(_INDEX_PATH))
            logger.info(f"Loaded FAISS index with {_faiss_index.ntotal} vectors")
            
            # Load metadata if it exists
            if _METADATA_PATH.exists():
                with open(_METADATA_PATH, 'r') as f:
                    _index_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            _create_new_index()
    else:
        _create_new_index()

def _create_new_index():
    """Create a new FAISS index."""
    global _faiss_index, _index_metadata
    
    # Create an L2 index with ID mapping
    _faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(_embedding_dim))
    _index_metadata = {
        "created_at": datetime.now().isoformat(),
        "vector_count": 0,
        "last_updated": datetime.now().isoformat()
    }
    _save_index()
    logger.info("Created new FAISS index")

def _save_index():
    """Save the FAISS index and metadata to disk."""
    if _faiss_index is not None:
        try:
            # Backup existing index
            if _INDEX_PATH.exists():
                shutil.copy2(_INDEX_PATH, _INDEX_BACKUP_PATH)
            
            # Save new index
            faiss.write_index(_faiss_index, str(_INDEX_PATH))
            
            # Update and save metadata
            _index_metadata["last_updated"] = datetime.now().isoformat()
            _index_metadata["vector_count"] = _faiss_index.ntotal
            
            with open(_METADATA_PATH, 'w') as f:
                json.dump(_index_metadata, f, indent=2)
                
            logger.info(f"Saved FAISS index with {_faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            # Restore backup if save failed
            if _INDEX_BACKUP_PATH.exists():
                shutil.copy2(_INDEX_BACKUP_PATH, _INDEX_PATH)

def sculpt_and_store(user_id: str, raw_concept: dict, metadata: dict) -> bool:
    """Store a concept with its embedding in the vector index.
    
    Args:
        user_id: User identifier
        raw_concept: Concept data dictionary with at least 'name' field
        metadata: Additional metadata for the concept
        
    Returns:
        bool: True if successfully stored, False otherwise
    """
    global _faiss_index
    
    # Ensure index is loaded
    if _faiss_index is None:
        _load_index()
    
    try:
        # Extract concept name
        concept_name = raw_concept.get('name', '')
        if not concept_name:
            logger.warning("Concept missing 'name' field")
            return False
        
        # Generate embedding
        embedding = _embedder.encode(concept_name).astype('float32')
        
        # Create unique ID based on user and timestamp
        concept_id = f"{user_id}_{int(time.time() * 1000000)}"
        
        # Prepare concept data
        concept_data = {
            "id": concept_id,
            "user_id": user_id,
            "concept": raw_concept,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        # Save concept JSON
        concept_path = _STORE_DIR / f"{concept_id}.json"
        with open(concept_path, 'w') as f:
            json.dump(concept_data, f, indent=2)
        
        # Add to FAISS index (using hash of concept_id as numeric ID)
        numeric_id = hash(concept_id) & 0x7FFFFFFFFFFFFFFF  # Ensure positive
        _faiss_index.add_with_ids(
            np.array([embedding]), 
            np.array([numeric_id], dtype=np.int64)
        )
        
        # Save index periodically (every 10 additions)
        if _faiss_index.ntotal % 10 == 0:
            _save_index()
        
        logger.info(f"Stored concept '{concept_name}' for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store concept: {e}")
        return False

def batch_sculpt_and_store(user_id: str, concepts: List[dict], metadata: dict) -> int:
    """Batch process multiple concepts for storage.
    
    Args:
        user_id: User identifier
        concepts: List of concept dictionaries
        metadata: Additional metadata for all concepts
        
    Returns:
        int: Number of successfully stored concepts
    """
    stored_count = 0
    
    for concept in concepts:
        if sculpt_and_store(user_id, concept, metadata):
            stored_count += 1
    
    # Force save after batch
    if stored_count > 0:
        _save_index()
    
    logger.info(f"Batch stored {stored_count}/{len(concepts)} concepts")
    return stored_count

def search_similar_concepts(query: str, user_id: Optional[str] = None, 
                          k: int = 10, threshold: float = 0.7) -> List[Tuple[dict, float]]:
    """Retrieve similar concepts by vector similarity.
    
    Args:
        query: Query text to search for
        user_id: Optional user ID to filter results
        k: Number of top results to return
        threshold: Minimum similarity score (0-1)
        
    Returns:
        List of (concept_data, similarity_score) tuples
    """
    global _faiss_index
    
    # Ensure index is loaded
    if _faiss_index is None:
        _load_index()
    
    if _faiss_index is None or _faiss_index.ntotal == 0:
        logger.warning("No concepts in index")
        return []
    
    try:
        # Generate query embedding
        query_embedding = _embedder.encode(query).astype('float32')
        
        # Search in FAISS
        k_search = min(k * 3, _faiss_index.ntotal)  # Search more to filter by user
        distances, indices = _faiss_index.search(
            np.array([query_embedding]), 
            k_search
        )
        
        results = []
        
        # Load and filter results
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + dist)
            
            if similarity < threshold:
                continue
            
            # Find matching concept file
            found = False
            for concept_file in _STORE_DIR.glob("*.json"):
                try:
                    with open(concept_file, 'r') as f:
                        concept_data = json.load(f)
                    
                    # Check if this concept matches the index ID
                    if (hash(concept_data['id']) & 0x7FFFFFFFFFFFFFFF) == idx:
                        # Filter by user if specified
                        if user_id and concept_data.get('user_id') != user_id:
                            continue
                        
                        results.append((concept_data, float(similarity)))
                        found = True
                        break
                        
                except Exception as e:
                    logger.error(f"Error reading {concept_file}: {e}")
            
            if found and len(results) >= k:
                break
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def rebuild_index() -> bool:
    """Rebuild the FAISS index from all stored concepts.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting index rebuild...")
    
    try:
        # Create new index
        new_index = faiss.IndexIDMap(faiss.IndexFlatL2(_embedding_dim))
        
        # Process all concept files
        concept_files = list(_STORE_DIR.glob("*.json"))
        embeddings = []
        ids = []
        
        for i, concept_file in enumerate(concept_files):
            try:
                with open(concept_file, 'r') as f:
                    concept_data = json.load(f)
                
                concept_name = concept_data['concept'].get('name', '')
                if not concept_name:
                    continue
                
                # Generate embedding
                embedding = _embedder.encode(concept_name).astype('float32')
                embeddings.append(embedding)
                
                # Use hash of concept ID
                numeric_id = hash(concept_data['id']) & 0x7FFFFFFFFFFFFFFF
                ids.append(numeric_id)
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(concept_files)} concepts")
                    
            except Exception as e:
                logger.error(f"Error processing {concept_file}: {e}")
        
        # Add all embeddings to new index
        if embeddings:
            embeddings_array = np.array(embeddings)
            ids_array = np.array(ids, dtype=np.int64)
            new_index.add_with_ids(embeddings_array, ids_array)
        
        # Replace old index
        global _faiss_index
        _faiss_index = new_index
        _save_index()
        
        logger.info(f"Index rebuild complete. Indexed {len(embeddings)} concepts.")
        return True
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        return False

def get_index_stats() -> dict:
    """Get statistics about the current index.
    
    Returns:
        dict: Statistics including vector count, index size, etc.
    """
    global _faiss_index
    
    # Ensure index is loaded
    if _faiss_index is None:
        _load_index()
    
    stats = {
        "vector_count": _faiss_index.ntotal if _faiss_index else 0,
        "index_size_bytes": _INDEX_PATH.stat().st_size if _INDEX_PATH.exists() else 0,
        "concept_file_count": len(list(_STORE_DIR.glob("*.json"))),
        "storage_dir": str(_STORE_DIR),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": _embedding_dim
    }
    
    if _index_metadata:
        stats.update({
            "created_at": _index_metadata.get("created_at"),
            "last_updated": _index_metadata.get("last_updated")
        })
    
    return stats

def cleanup_orphaned_files():
    """Remove concept files that are not in the index."""
    global _faiss_index
    
    # Ensure index is loaded
    if _faiss_index is None:
        _load_index()
    
    if _faiss_index is None:
        logger.warning("Cannot cleanup without index")
        return
    
    # Get all indexed IDs
    indexed_ids = set()
    if _faiss_index.ntotal > 0:
        # This is a workaround since FAISS doesn't expose stored IDs easily
        # We'll rebuild the set from files
        for concept_file in _STORE_DIR.glob("*.json"):
            try:
                with open(concept_file, 'r') as f:
                    concept_data = json.load(f)
                concept_id = concept_data['id']
                numeric_id = hash(concept_id) & 0x7FFFFFFFFFFFFFFF
                
                # Check if this ID exists in index by searching for it
                # This is inefficient but works for cleanup
                test_embedding = np.zeros((1, _embedding_dim), dtype='float32')
                distances, indices = _faiss_index.search(test_embedding, _faiss_index.ntotal)
                
                if numeric_id in indices[0]:
                    indexed_ids.add(concept_id)
                    
            except Exception as e:
                logger.error(f"Error checking {concept_file}: {e}")
    
    # Remove orphaned files
    removed_count = 0
    for concept_file in _STORE_DIR.glob("*.json"):
        try:
            with open(concept_file, 'r') as f:
                concept_data = json.load(f)
            
            if concept_data['id'] not in indexed_ids:
                concept_file.unlink()
                removed_count += 1
                logger.info(f"Removed orphaned file: {concept_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {concept_file}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleanup complete. Removed {removed_count} orphaned files.")


if __name__ == "__main__":
    # Command-line interface for testing and maintenance
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Sculptor CLI")
    parser.add_argument("command", choices=["stats", "rebuild", "cleanup", "test"],
                       help="Command to execute")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.command == "stats":
        stats = get_index_stats()
        print(json.dumps(stats, indent=2))
        
    elif args.command == "rebuild":
        success = rebuild_index()
        print(f"Rebuild {'successful' if success else 'failed'}")
        
    elif args.command == "cleanup":
        cleanup_orphaned_files()
        
    elif args.command == "test":
        # Test basic functionality
        test_concept = {
            "name": "test concept",
            "description": "A test concept for verification"
        }
        
        print("Testing store...")
        success = sculpt_and_store("test_user", test_concept, {"source": "cli_test"})
        print(f"Store: {'✓' if success else '✗'}")
        
        print("\nTesting search...")
        results = search_similar_concepts("test", k=5)
        print(f"Found {len(results)} similar concepts")
        
        print("\nIndex stats:")
        print(json.dumps(get_index_stats(), indent=2))
