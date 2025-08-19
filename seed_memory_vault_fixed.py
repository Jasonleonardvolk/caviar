#!/usr/bin/env python3
"""
Seed the memory vault with baseline concepts - FIXED VERSION 2
This script loads seed concepts into the unified memory vault
Handles the set serialization issue
"""

import json
import logging
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def seed_memory_vault_async():
    """Load seed concepts into the memory vault (async version)"""
    try:
        # Import memory vault
        from python.core.memory_vault import UnifiedMemoryVault, MemoryType
        
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
                # Prepare the concept data
                concept_data = {
                    "name": concept["name"],
                    "category": concept.get("category", "general"),
                    "priority": concept.get("priority", 0.5),
                    "boost_multiplier": concept.get("boost_multiplier", 1.0)
                }
                
                # Prepare metadata
                metadata = {
                    "source": "seed_data",
                    "file": str(seed_file),
                    "concept_type": "seed",
                    "category": concept.get("category", "general")
                }
                
                # Tags as a list (not set)
                tags = ["seed", "concept", concept.get("category", "general")]
                
                # Store in vault using async method
                memory_id = await vault.store(
                    content=concept_data,  # Store the concept data directly
                    memory_type=MemoryType.SEMANTIC,  # Use the enum
                    metadata=metadata,
                    tags=tags,  # Pass as list
                    importance=concept.get("priority", 0.5)
                )
                
                total_concepts += 1
                logger.debug(f"  ✅ Stored concept '{concept['name']}' with ID: {memory_id}")
        
        # Save all memories to disk
        vault.save_all()
        
        logger.info(f"🎯 Seed injection complete:")
        logger.info(f"  📊 Total concepts loaded: {total_concepts}")
        logger.info(f"  📁 Storage path: {vault.storage_path}")
        
        # Get vault stats
        stats = vault.get_statistics()
        logger.info(f"  🧠 Total memories in vault: {stats['total_memories']}")
        logger.info(f"  💾 Storage size: {stats['total_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error seeding memory vault: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_seed_data_async():
    """Verify that seed data was properly loaded (async version)"""
    try:
        from python.core.memory_vault import UnifiedMemoryVault, MemoryType
        
        # Create vault instance with same config
        config = {
            'storage_path': 'data/memory_vault',
            'max_working_memory': 100,
            'ghost_memory_ttl': 3600
        }
        vault = UnifiedMemoryVault(config)
        
        # Search for semantic memories with seed tag
        seed_memories = await vault.search(
            memory_type=MemoryType.SEMANTIC,  # Search semantic memories
            tags=['seed'],
            max_results=100
        )
        
        if seed_memories:
            logger.info(f"✅ Verification successful! Found {len(seed_memories)} seed memories")
            for memory in seed_memories[:5]:  # Show first 5
                content = memory.content
                if isinstance(content, dict) and 'name' in content:
                    concept_name = content.get('name', 'Unknown')
                else:
                    concept_name = str(content)[:50]
                logger.info(f"  • {concept_name}")
            if len(seed_memories) > 5:
                logger.info(f"  ... and {len(seed_memories) - 5} more")
            
            # Show categories
            categories = set()
            for memory in seed_memories:
                if memory.metadata.get('category'):
                    categories.add(memory.metadata['category'])
            if categories:
                logger.info(f"  📁 Categories: {', '.join(sorted(categories))}")
        else:
            logger.warning("⚠️ No seed concepts found in vault")
        
        return len(seed_memories) > 0
        
    except Exception as e:
        logger.error(f"❌ Error verifying seed data: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main async function"""
    logger.info("🌱 Starting memory vault seeding process...")
    
    # Seed the vault
    if await seed_memory_vault_async():
        logger.info("✅ Seeding completed successfully!")
        
        # Verify the data
        logger.info("\n🔍 Verifying seed data...")
        if await verify_seed_data_async():
            logger.info("✅ All seed data verified!")
        else:
            logger.warning("⚠️ Seed data verification failed")
    else:
        logger.error("❌ Seeding failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
