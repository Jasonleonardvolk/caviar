#!/usr/bin/env python3
"""tier_mover.py - Move least-recently-touched concepts from Hot → Warm → Cold storage tiers.

This maintenance script implements tiered storage migration for the concept mesh.
It runs as a oneshot job triggered by the nightly timer.
"""

import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Storage tier paths
HOT_PATH = Path("/opt/tori/mesh/hot")
WARM_PATH = Path("/opt/tori/mesh/warm")
COLD_PATH = Path("/opt/tori/mesh/cold")
ARCHIVE_PATH = Path("/opt/tori/mesh/archive")

# Age thresholds (in days)
HOT_TO_WARM_DAYS = 7    # Move from hot to warm after 7 days
WARM_TO_COLD_DAYS = 30  # Move from warm to cold after 30 days
COLD_TO_ARCHIVE_DAYS = 90  # Move from cold to archive after 90 days

# Maximum files to process per tier in one run
MAX_FILES_PER_TIER = 1000

# File size limits for tiers (in MB)
MAX_HOT_SIZE_MB = 500
MAX_WARM_SIZE_MB = 2000
MAX_COLD_SIZE_MB = 10000


def ensure_directories():
    """Ensure all tier directories exist."""
    for path in [HOT_PATH, WARM_PATH, COLD_PATH, ARCHIVE_PATH]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")


def get_file_age_days(file_path: Path) -> float:
    """Get the age of a file in days based on last access time."""
    try:
        # Try to use access time first
        stat = file_path.stat()
        last_access = stat.st_atime
        
        # Fall back to modification time if access time is not reliable
        if last_access == 0:
            last_access = stat.st_mtime
            
        age_seconds = time.time() - last_access
        return age_seconds / (24 * 3600)
    except Exception as e:
        logger.error(f"Error getting age for {file_path}: {e}")
        return 0


def get_tier_size_mb(tier_path: Path) -> float:
    """Get total size of all files in a tier in MB."""
    total_size = 0
    try:
        for file_path in tier_path.glob("*.json"):
            total_size += file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating tier size for {tier_path}: {e}")
    
    return total_size / (1024 * 1024)


def move_file_to_tier(source_path: Path, dest_tier: Path) -> bool:
    """Move a file from source to destination tier."""
    try:
        dest_path = dest_tier / source_path.name
        
        # Check if file already exists in destination
        if dest_path.exists():
            logger.warning(f"File already exists in destination: {dest_path}")
            # Remove source file since it's a duplicate
            source_path.unlink()
            return True
        
        # Move the file
        shutil.move(str(source_path), str(dest_path))
        logger.info(f"Moved {source_path.name} from {source_path.parent.name} to {dest_tier.name}")
        
        # Try to preserve access time
        try:
            source_stat = source_path.stat()
            os.utime(dest_path, (source_stat.st_atime, source_stat.st_mtime))
        except:
            pass
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to move {source_path} to {dest_tier}: {e}")
        return False


def process_tier_migration(source_tier: Path, dest_tier: Path, age_threshold_days: int, 
                          tier_name: str) -> Dict[str, int]:
    """Process migration from one tier to another based on age."""
    stats = {
        "checked": 0,
        "moved": 0,
        "errors": 0,
        "size_moved_mb": 0
    }
    
    logger.info(f"Processing {tier_name} tier migration (threshold: {age_threshold_days} days)")
    
    # Get all JSON files sorted by access time (oldest first)
    try:
        files = []
        for file_path in source_tier.glob("*.json"):
            age_days = get_file_age_days(file_path)
            if age_days >= age_threshold_days:
                files.append((file_path, age_days))
        
        # Sort by age (oldest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of files to process
        files = files[:MAX_FILES_PER_TIER]
        
        logger.info(f"Found {len(files)} files eligible for migration in {tier_name}")
        
        for file_path, age_days in files:
            stats["checked"] += 1
            
            # Get file size before moving
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if move_file_to_tier(file_path, dest_tier):
                stats["moved"] += 1
                stats["size_moved_mb"] += file_size_mb
            else:
                stats["errors"] += 1
                
    except Exception as e:
        logger.error(f"Error processing {tier_name} tier: {e}")
        stats["errors"] += 1
    
    return stats


def cleanup_old_archives(archive_tier: Path, max_age_days: int = 365) -> Dict[str, int]:
    """Clean up very old files from archive tier."""
    stats = {
        "checked": 0,
        "deleted": 0,
        "size_freed_mb": 0
    }
    
    logger.info(f"Checking for archives older than {max_age_days} days")
    
    try:
        for file_path in archive_tier.glob("*.json"):
            stats["checked"] += 1
            age_days = get_file_age_days(file_path)
            
            if age_days > max_age_days:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                try:
                    file_path.unlink()
                    stats["deleted"] += 1
                    stats["size_freed_mb"] += file_size_mb
                    logger.info(f"Deleted old archive: {file_path.name} (age: {age_days:.1f} days)")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up archives: {e}")
    
    return stats


def get_tier_statistics() -> Dict[str, Dict]:
    """Get statistics for all storage tiers."""
    stats = {}
    
    for tier_name, tier_path in [
        ("hot", HOT_PATH),
        ("warm", WARM_PATH),
        ("cold", COLD_PATH),
        ("archive", ARCHIVE_PATH)
    ]:
        file_count = len(list(tier_path.glob("*.json")))
        size_mb = get_tier_size_mb(tier_path)
        
        stats[tier_name] = {
            "path": str(tier_path),
            "file_count": file_count,
            "size_mb": round(size_mb, 2),
            "size_human": f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
        }
    
    return stats


def check_tier_capacity() -> List[str]:
    """Check if any tier is approaching capacity limits."""
    warnings = []
    
    hot_size = get_tier_size_mb(HOT_PATH)
    if hot_size > MAX_HOT_SIZE_MB * 0.8:
        warnings.append(f"Hot tier is at {hot_size/MAX_HOT_SIZE_MB*100:.1f}% capacity")
    
    warm_size = get_tier_size_mb(WARM_PATH)
    if warm_size > MAX_WARM_SIZE_MB * 0.8:
        warnings.append(f"Warm tier is at {warm_size/MAX_WARM_SIZE_MB*100:.1f}% capacity")
    
    cold_size = get_tier_size_mb(COLD_PATH)
    if cold_size > MAX_COLD_SIZE_MB * 0.8:
        warnings.append(f"Cold tier is at {cold_size/MAX_COLD_SIZE_MB*100:.1f}% capacity")
    
    return warnings


def main():
    """Main entry point for tier mover."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Tori Tier Mover")
    logger.info(f"Run time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Get initial statistics
    initial_stats = get_tier_statistics()
    logger.info("Initial tier statistics:")
    for tier, stats in initial_stats.items():
        logger.info(f"  {tier}: {stats['file_count']} files, {stats['size_human']}")
    
    # Check capacity warnings
    warnings = check_tier_capacity()
    for warning in warnings:
        logger.warning(warning)
    
    # Process migrations
    all_stats = {}
    
    # Hot → Warm
    logger.info("\n" + "-" * 40)
    all_stats["hot_to_warm"] = process_tier_migration(
        HOT_PATH, WARM_PATH, HOT_TO_WARM_DAYS, "Hot→Warm"
    )
    
    # Warm → Cold
    logger.info("\n" + "-" * 40)
    all_stats["warm_to_cold"] = process_tier_migration(
        WARM_PATH, COLD_PATH, WARM_TO_COLD_DAYS, "Warm→Cold"
    )
    
    # Cold → Archive
    logger.info("\n" + "-" * 40)
    all_stats["cold_to_archive"] = process_tier_migration(
        COLD_PATH, ARCHIVE_PATH, COLD_TO_ARCHIVE_DAYS, "Cold→Archive"
    )
    
    # Clean up old archives
    logger.info("\n" + "-" * 40)
    all_stats["archive_cleanup"] = cleanup_old_archives(ARCHIVE_PATH)
    
    # Get final statistics
    final_stats = get_tier_statistics()
    
    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info("TIER MOVER SUMMARY")
    logger.info("=" * 60)
    
    total_moved = sum(s["moved"] for s in all_stats.values() if "moved" in s)
    total_size_moved = sum(s["size_moved_mb"] for s in all_stats.values() if "size_moved_mb" in s)
    total_deleted = all_stats["archive_cleanup"]["deleted"]
    total_size_freed = all_stats["archive_cleanup"]["size_freed_mb"]
    
    logger.info(f"Total files moved: {total_moved}")
    logger.info(f"Total size moved: {total_size_moved:.1f} MB")
    logger.info(f"Total files deleted: {total_deleted}")
    logger.info(f"Total size freed: {total_size_freed:.1f} MB")
    
    logger.info("\nFinal tier statistics:")
    for tier, stats in final_stats.items():
        initial = initial_stats[tier]
        delta_files = stats['file_count'] - initial['file_count']
        delta_size = stats['size_mb'] - initial['size_mb']
        
        logger.info(f"  {tier}: {stats['file_count']} files ({delta_files:+d}), "
                   f"{stats['size_human']} ({delta_size:+.1f} MB)")
    
    # Execution time
    duration = time.time() - start_time
    logger.info(f"\nExecution time: {duration:.1f} seconds")
    
    # Check for errors
    total_errors = sum(s.get("errors", 0) for s in all_stats.values())
    if total_errors > 0:
        logger.warning(f"Completed with {total_errors} errors")
        sys.exit(1)
    else:
        logger.info("Completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
