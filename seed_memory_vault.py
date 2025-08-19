#!/usr/bin/env python3
"""
Seed the memory vault with baseline concepts
This script loads seed concepts into the unified memory vault
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def seed_memory_vault():
    """Load seed concepts into the memory vault"""
    try:
        # Import memory vault
        from python.core.memory_vault import UnifiedMemoryVault
        
        # Create vault instance with config
        config = {
            'storage_path': 'data/memory_vault',
            'max_working_memory': 100,
            'ghost_memory_ttl': 3600  # 1 hour
        }
        vault = UnifiedMemoryVault(config)
        
        logger.info("✅ Memory vault initialized")
        
        # Load seed concepts
        seed_files = [
            Path("data/concept_db.json"),
            Path("data/universal_seed.json")
        ]
        
        total_concepts = 0
        
        for seed_file in seed_files:
            if not seed_file.exists():
                logger.warning(f"⚠️ Seed file not found: {seed_file}")
                continue
            
            logger.info(f"📂 Loading concepts from: {seed_file}")
            
            with open(seed_file, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
            
            # Store each concept as a memory
            for concept in concepts:
                memory_id = f"seed_concept_{concept['name'].replace(' ', '_')}"
                
                memory_data = {
                    "id": memory_id,
                    "type": "concept",
                    "content": {
                        "name": concept["name"],
                        "category": concept.get("category", "general"),
                        "priority": concept.get("priority", 0.5),
                        "boost_multiplier": concept.get("boost_multiplier", 1.0)
                    },
                    "metadata": {
                        "source": "seed_data",
                        "file": str(seed_file)
                    }
                }
                
                # Store in vault (using async store method)
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(vault.store(
                    content=memory_data,
                    memory_type='semantic',  # Store concepts as semantic memory
                    metadata=memory_data.get('metadata', {})
                ))
                loop.close()
                total_concepts += 1
                logger.debug(f"  ✅ Stored concept: {concept['name']}")
        
        # Save all memories to disk
        saved_count = vault.save_all()
        
        logger.info(f"🎯 Seed injection complete:")
        logger.info(f"  📊 Total concepts loaded: {total_concepts}")
        logger.info(f"  💾 Memories saved to disk: {saved_count}")
        logger.info(f"  📁 Storage path: {vault.storage_path}")
        
        # Verify by checking vault stats
        if hasattr(vault, 'memories'):
            logger.info(f"  🧠 Total memories in vault: {len(vault.memories)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error seeding memory vault: {e}")
        return False

def verify_seed_data():
    """Verify that seed data was properly loaded"""
    try:
        from python.core.memory_vault import UnifiedMemoryVault
        
        # Create vault instance with same config
        config = {
            'storage_path': 'data/memory_vault',
            'max_working_memory': 100,
            'ghost_memory_ttl': 3600
        }
        vault = UnifiedMemoryVault(config)
        
        # Check for seeded concepts by loading from files
        concept_count = 0
        seed_concepts = []
        
        # Check the index for semantic memories
        if hasattr(vault, 'type_index') and 'semantic' in vault.type_index:
            semantic_ids = vault.type_index['semantic']
            logger.info(f"🔍 Found {len(semantic_ids)} semantic memories in index")
            
            # Load a few to verify
            for memory_id in semantic_ids[:5]:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    memory = loop.run_until_complete(vault.retrieve(memory_id))
                    loop.close()
                    
                    if memory and memory.metadata.get('source') == 'seed_data':
                        concept_name = memory.content.get('content', {}).get('name', 'Unknown')
                        seed_concepts.append(concept_name)
                        concept_count += 1
                except Exception as e:
                    logger.debug(f"Could not retrieve {memory_id}: {e}")
        
        if seed_concepts:
            logger.info(f"✅ Verification successful! Found {len(seed_concepts)} seed concepts:")
            for concept in seed_concepts[:5]:  # Show first 5
                logger.info(f"  • {concept}")
            if len(seed_concepts) > 5:
                logger.info(f"  ... and {len(seed_concepts) - 5} more")
        else:
            logger.warning("⚠️ No seed concepts found in vault")
        
        return len(seed_concepts) > 0
        
    except Exception as e:
        logger.error(f"❌ Error verifying seed data: {e}")
        return False

if __name__ == "__main__":
    logger.info("🌱 Starting memory vault seeding process...")
    
    # Seed the vault
    if seed_memory_vault():
        logger.info("✅ Seeding completed successfully!")
        
        # Verify the data
        logger.info("\n🔍 Verifying seed data...")
        if verify_seed_data():
            logger.info("✅ All seed data verified!")
        else:
            logger.warning("⚠️ Seed data verification failed")
    else:
        logger.error("❌ Seeding failed!")
        sys.exit(1)
