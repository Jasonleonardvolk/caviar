#!/usr/bin/env python3
"""
Daily seal job for PsiArchive - Run at 23:59 to seal the current day's archive
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.psi_archive_extended import PSI_ARCHIVER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seal_current_day():
    """Seal the current day's archive file"""
    try:
        logger.info(f"🔒 Starting daily seal job at {datetime.now()}")
        
        # Seal current file (forces rotation and sealing)
        if PSI_ARCHIVER.current_file and PSI_ARCHIVER.current_file.exists():
            PSI_ARCHIVER._seal_daily_archive(PSI_ARCHIVER.current_file)
            logger.info(f"✅ Successfully sealed {PSI_ARCHIVER.current_file}")
        else:
            logger.warning("⚠️ No current file to seal")
        
        # Also ensure yesterday is sealed (in case of missed run)
        PSI_ARCHIVER.seal_yesterday()
        
        logger.info("✅ Daily seal job completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Daily seal job failed: {e}")
        return False


if __name__ == "__main__":
    success = seal_current_day()
    sys.exit(0 if success else 1)
