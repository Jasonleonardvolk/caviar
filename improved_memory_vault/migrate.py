"""
Memory Vault Migration Tool
Migrates data from V1 (mixed async/sync) to V2 (fully async)
"""

import asyncio
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryVaultMigrator:
    """Handles migration from V1 to V2 memory vault"""
    
    def __init__(self, v1_path: str, v2_path: str):
        self.v1_path = Path(v1_path)
        self.v2_path = Path(v2_path)
        self.migration_log_path = Path(v2_path).parent / 'migration_log.json'
        
        # Migration statistics
        self.stats = {
            'total_memories': 0,
            'migrated': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
    
    async def migrate(self, cleanup: bool = False) -> Dict[str, Any]:
        """
        Perform the migration
        
        Args:
            cleanup: Whether to remove V1 data after successful migration
        
        Returns:
            Migration statistics
        """
        logger.info(f"Starting migration from {self.v1_path} to {self.v2_path}")
        
        # Dynamic imports to handle path issues
        try:
            from memory_vault import UnifiedMemoryVault, MemoryType
            from memory_vault_v2 import UnifiedMemoryVaultV2
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from memory_vault import UnifiedMemoryVault, MemoryType
            from improved_memory_vault.memory_vault_v2 import UnifiedMemoryVaultV2
        
        # Initialize V1 vault (legacy)
        v1_config = {'storage_path': str(self.v1_path)}
        v1_vault = UnifiedMemoryVault(v1_config)
        
        # Initialize V2 vault (new async)
        v2_config = {'storage_path': str(self.v2_path)}
        v2_vault = UnifiedMemoryVaultV2(v2_config)
        await v2_vault.initialize()
        
        try:
            # Get V1 statistics
            v1_stats = v1_vault.get_statistics()
            self.stats['total_memories'] = v1_stats['total_memories']
            logger.info(f"Found {self.stats['total_memories']} memories to migrate")
            
            # Migrate working memory
            logger.info("Migrating working memory...")
            for memory_id, memory in v1_vault.working_memory.items():
                await self._migrate_memory(memory, v2_vault)
            
            # Migrate ghost memory
            logger.info("Migrating ghost memory...")
            for memory_id, memory in v1_vault.ghost_memory.items():
                await self._migrate_memory(memory, v2_vault)
            
            # Migrate file storage
            logger.info("Migrating file storage...")
            for memory_id in v1_vault.main_index.keys():
                try:
                    # Use asyncio.run since V1 has mixed async
                    memory = asyncio.run(v1_vault._load_memory_from_file(memory_id))
                    if memory:
                        await self._migrate_memory(memory, v2_vault)
                    else:
                        self.stats['skipped'] += 1
                except Exception as e:
                    logger.error(f"Failed to load memory {memory_id}: {e}")
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'memory_id': memory_id,
                        'error': str(e)
                    })
            
            # Migrate deduplication hashes
            logger.info("Migrating deduplication data...")
            v2_vault.seen_hashes = v1_vault.seen_hashes.copy()
            await v2_vault._save_seen_hashes()
            
            # Save migration log
            await self._save_migration_log(v2_vault)
            
            # Final save
            await v2_vault.save_all()
            
            # Cleanup if requested
            if cleanup and self.stats['failed'] == 0:
                logger.info("Cleaning up V1 data...")
                shutil.rmtree(self.v1_path)
                logger.info("V1 data removed")
            
            # Shutdown V2 vault
            await v2_vault.shutdown()
            
            logger.info(f"Migration complete: {self.stats}")
            return self.stats
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.stats['errors'].append({
                'type': 'migration_error',
                'error': str(e)
            })
            await v2_vault.shutdown()
            raise
    
    async def _migrate_memory(self, memory, v2_vault):
        """Migrate a single memory entry"""
        try:
            # Extract tags from metadata
            tags = memory.metadata.get('tags', [])
            
            # Store in V2 vault
            await v2_vault.store(
                content=memory.content,
                memory_type=memory.type,
                metadata=memory.metadata,
                embedding=memory.embedding,
                importance=memory.importance,
                tags=tags
            )
            
            self.stats['migrated'] += 1
            
            # Log progress every 100 memories
            if self.stats['migrated'] % 100 == 0:
                logger.info(f"Progress: {self.stats['migrated']}/{self.stats['total_memories']}")
                
        except Exception as e:
            logger.error(f"Failed to migrate memory {memory.id}: {e}")
            self.stats['failed'] += 1
            self.stats['errors'].append({
                'memory_id': memory.id,
                'error': str(e)
            })
    
    async def _save_migration_log(self, v2_vault):
        """Save migration log for audit trail"""
        log_data = {
            'migration_date': asyncio.get_event_loop().time(),
            'source_path': str(self.v1_path),
            'target_path': str(self.v2_path),
            'statistics': self.stats,
            'v2_stats': await v2_vault.get_statistics()
        }
        
        with open(self.migration_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Migration log saved to {self.migration_log_path}")
    
    async def verify_migration(self) -> bool:
        """Verify that migration was successful"""
        logger.info("Verifying migration...")
        
        # Dynamic imports
        try:
            from memory_vault import UnifiedMemoryVault
            from memory_vault_v2 import UnifiedMemoryVaultV2
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from memory_vault import UnifiedMemoryVault
            from improved_memory_vault.memory_vault_v2 import UnifiedMemoryVaultV2
        
        # Initialize both vaults
        v1_vault = UnifiedMemoryVault({'storage_path': str(self.v1_path)})
        
        v2_vault = UnifiedMemoryVaultV2({'storage_path': str(self.v2_path)})
        await v2_vault.initialize()
        
        try:
            # Get statistics
            v1_stats = v1_vault.get_statistics()
            v2_stats = await v2_vault.get_statistics()
            
            # Basic verification
            v1_total = v1_stats['total_memories']
            v2_total = v2_stats['total_memories']
            
            logger.info(f"V1 memories: {v1_total}")
            logger.info(f"V2 memories: {v2_total}")
            logger.info(f"Migration success rate: {v2_total/v1_total*100:.1f}%")
            
            # Check if all memories migrated
            success = v2_total >= v1_total - self.stats['failed']
            
            if success:
                logger.info("✅ Migration verification passed")
            else:
                logger.warning("⚠️ Migration verification failed - memory count mismatch")
            
            await v2_vault.shutdown()
            return success
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            await v2_vault.shutdown()
            return False


async def main():
    """Main migration entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Memory Vault from V1 to V2')
    parser.add_argument('v1_path', help='Path to V1 memory vault')
    parser.add_argument('v2_path', help='Path for V2 memory vault')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Remove V1 data after successful migration')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing migration')
    
    args = parser.parse_args()
    
    migrator = MemoryVaultMigrator(args.v1_path, args.v2_path)
    
    if args.verify_only:
        success = await migrator.verify_migration()
        sys.exit(0 if success else 1)
    else:
        stats = await migrator.migrate(cleanup=args.cleanup)
        
        # Verify after migration
        success = await migrator.verify_migration()
        
        if stats['failed'] > 0 or not success:
            logger.error(f"Migration completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("Migration completed successfully")
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
