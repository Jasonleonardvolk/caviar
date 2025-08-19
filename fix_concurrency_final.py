#!/usr/bin/env python3
"""
Fix the concurrency configuration
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fix_concurrency_config():
    """Fix the concurrency configuration in concurrency_manager.py"""
    
    file_path = Path("ingest_pdf/pipeline/concurrency_manager.py")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix in the __post_init__ method
    original = """self.cpu_workers = max(1, os.cpu_count() - 1)"""
    replacement = """self.cpu_workers = min(8, max(1, (os.cpu_count() or 4) - 1))"""
    
    if original in content:
        content = content.replace(original, replacement)
        
        # Also fix the chunk processor workers
        original2 = """self.chunk_processor_workers = min(16, os.cpu_count() or 1)"""
        replacement2 = """self.chunk_processor_workers = min(8, os.cpu_count() or 1)"""
        
        if original2 in content:
            content = content.replace(original2, replacement2)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ Fixed concurrency configuration in {file_path}")
        logger.info(f"   CPU workers: capped at 8")
        logger.info(f"   Chunk workers: capped at 8")
        return True
    else:
        logger.warning("Could not find exact pattern, trying alternate fix...")
        
        # Try to find the line number
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "os.cpu_count() - 1" in line and "cpu_workers" in line:
                lines[i] = "            self.cpu_workers = min(8, max(1, (os.cpu_count() or 4) - 1))"
                logger.info(f"Fixed line {i+1}")
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                return True
        
        logger.error("Could not find pattern to fix")
        return False


if __name__ == "__main__":
    logger.info("FIXING CONCURRENCY CONFIGURATION")
    logger.info("="*70)
    
    if fix_concurrency_config():
        logger.info("\n✅ Concurrency fix applied successfully!")
        logger.info("\nThis will:")
        logger.info("- Cap CPU workers at 8 (instead of 19)")
        logger.info("- Cap chunk workers at 8 (instead of 16)")
        logger.info("- Reduce context switching overhead")
        logger.info("- Improve overall performance")
    else:
        logger.error("\n❌ Failed to apply concurrency fix")
