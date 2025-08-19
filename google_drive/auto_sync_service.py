"""
Google Drive Auto-Sync Service
Runs continuous synchronization in the background
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from google_drive.drive_sync import GoogleDriveSync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('google_drive/auto_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoSyncService:
    """Background service for automatic Google Drive synchronization"""
    
    def __init__(self):
        self.sync_manager = GoogleDriveSync()
        self.running = False
        self.sync_task = None
        
    async def start(self):
        """Start the auto-sync service"""
        logger.info("Starting Google Drive Auto-Sync Service...")
        
        # Authenticate first
        try:
            self.sync_manager.authenticate()
            logger.info("Authentication successful")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return
        
        self.running = True
        
        # Check if auto-sync is enabled
        if not self.sync_manager.config['sync_settings']['enabled']:
            logger.warning("Sync is disabled in configuration")
            return
        
        # Start sync loop
        await self.sync_loop()
    
    async def sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                logger.info("=" * 50)
                logger.info(f"Starting sync cycle at {datetime.now()}")
                
                # Sync each configured folder
                for folder_config in self.sync_manager.config['local_sync_folders']:
                    if not folder_config['sync_enabled']:
                        continue
                    
                    local_path = Path(folder_config['local_path'])
                    drive_path = folder_config['drive_path']
                    
                    if not local_path.exists():
                        logger.warning(f"Local path does not exist: {local_path}")
                        continue
                    
                    logger.info(f"Syncing {local_path} -> {drive_path}")
                    
                    # Perform sync
                    results = self.sync_manager.sync_folder(local_path)
                    
                    # Log results
                    if results['uploaded']:
                        logger.info(f"Uploaded {len(results['uploaded'])} files")
                    if results['downloaded']:
                        logger.info(f"Downloaded {len(results['downloaded'])} files")
                    if results['updated']:
                        logger.info(f"Updated {len(results['updated'])} files")
                    if results['errors']:
                        logger.error(f"Errors: {results['errors']}")
                
                # Check if backup is needed
                backup_settings = self.sync_manager.config['backup_settings']
                if backup_settings['auto_backup']:
                    # Check if it's time for backup
                    last_backup_file = Path('google_drive/.last_backup')
                    should_backup = False
                    
                    if not last_backup_file.exists():
                        should_backup = True
                    else:
                        last_backup = datetime.fromisoformat(
                            last_backup_file.read_text().strip()
                        )
                        hours_since_backup = (datetime.now() - last_backup).total_seconds() / 3600
                        if hours_since_backup >= backup_settings['backup_interval_hours']:
                            should_backup = True
                    
                    if should_backup:
                        logger.info("Creating backup...")
                        if self.sync_manager.backup_project():
                            last_backup_file.write_text(datetime.now().isoformat())
                            logger.info("Backup completed successfully")
                        else:
                            logger.error("Backup failed")
                
                # Wait for next sync interval
                interval = self.sync_manager.config['sync_settings']['sync_interval_minutes']
                logger.info(f"Next sync in {interval} minutes")
                logger.info("=" * 50)
                
                await asyncio.sleep(interval * 60)
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                # Wait a minute before retrying
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop the auto-sync service"""
        logger.info("Stopping Google Drive Auto-Sync Service...")
        self.running = False
        if self.sync_task:
            self.sync_task.cancel()

def signal_handler(service):
    """Handle shutdown signals"""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}")
        service.stop()
        sys.exit(0)
    return handler

async def main():
    """Main entry point"""
    service = AutoSyncService()
    
    # Set up signal handlers for graceful shutdown
    handler = signal_handler(service)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # Start the service
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        service.stop()
    except Exception as e:
        logger.error(f"Service error: {e}")
        service.stop()

if __name__ == "__main__":
    print("=" * 50)
    print("Google Drive Auto-Sync Service")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    print("")
    
    # Run the service
    asyncio.run(main())
