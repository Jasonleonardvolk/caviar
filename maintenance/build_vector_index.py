#!/usr/bin/env python3
"""
Rebuild the FAISS vector index from all *.json concept nodes in /hot.
Schedule weekly via systemd to guarantee index freshness.
"""
import json
import pathlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/tori-vector-index-rebuild.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('vector_index_builder')

# Paths
HOT = pathlib.Path("/opt/tori/mesh/hot")
IDX = HOT / "faiss.index"
IDX_BACKUP = HOT / "faiss.index.backup"

def main():
    """Main function to rebuild the FAISS vector index."""
    start_time = datetime.now()
    logger.info(f"Starting vector index rebuild at {start_time}")
    
    try:
        # Ensure directory exists
        if not HOT.exists():
            logger.error(f"Hot storage directory {HOT} does not exist")
            sys.exit(1)
        
        # Initialize embedding model
        logger.info("Loading sentence transformer model...")
        embed = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = embed.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, embedding dimension: {embedding_dim}")
        
        # Create new FAISS index
        logger.info("Creating new FAISS index...")
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Backup existing index if it exists
        if IDX.exists():
            logger.info(f"Backing up existing index to {IDX_BACKUP}")
            import shutil
            shutil.copy2(IDX, IDX_BACKUP)
        
        # Process all concept files
        concept_count = 0
        error_count = 0
        node_files = list(HOT.glob("*.json"))
        total_files = len(node_files)
        logger.info(f"Found {total_files} JSON files to process")
        
        for i, node_file in enumerate(node_files):
            try:
                # Load node data
                node = json.loads(node_file.read_text())
                
                # Skip non-concept nodes
                if node.get("type") != "concept":
                    continue
                
                # Extract concept name
                concept_name = None
                if "data" in node and isinstance(node["data"], dict):
                    concept_name = node["data"].get("name")
                elif "name" in node:
                    concept_name = node["name"]
                
                if not concept_name:
                    logger.warning(f"No concept name found in {node_file.name}")
                    continue
                
                # Generate embedding
                vec = embed.encode(concept_name).astype("float32")
                
                # Generate numeric ID from node ID
                node_id = node.get("id", node_file.stem)
                numeric_id = int(hash(node_id) & 0x7FFFFFFF)
                
                # Add to index
                index.add_with_ids(vec.reshape(1, -1), np.array([numeric_id], dtype=np.int64))
                concept_count += 1
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{total_files} files processed, {concept_count} concepts indexed")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {node_file.name}: {e}")
        
        # Write the new index
        logger.info(f"Writing index to {IDX}")
        faiss.write_index(index, str(IDX))
        
        # Remove backup if successful
        if IDX_BACKUP.exists() and error_count == 0:
            IDX_BACKUP.unlink()
            logger.info("Removed backup after successful rebuild")
        
        # Summary
        duration = datetime.now() - start_time
        logger.info(f"Rebuild completed in {duration}")
        logger.info(f"Processed {total_files} files")
        logger.info(f"Indexed {concept_count} concepts")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Final index size: {index.ntotal} entries")
        
        print(f"Rebuilt vector index with {index.ntotal} entries")
        
        # Verify index can be loaded
        logger.info("Verifying index can be loaded...")
        test_index = faiss.read_index(str(IDX))
        logger.info(f"Verification successful, index has {test_index.ntotal} entries")
        
    except Exception as e:
        logger.error(f"Fatal error during index rebuild: {e}")
        
        # Restore backup if available
        if IDX_BACKUP.exists():
            logger.info("Restoring backup index...")
            import shutil
            shutil.copy2(IDX_BACKUP, IDX)
            logger.info("Backup restored")
        
        raise

if __name__ == "__main__":
    main()
