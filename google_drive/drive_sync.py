import os
import json
import pickle
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import hashlib
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriveFile:
    """Represents a file in Google Drive"""
    id: str
    name: str
    mime_type: str
    size: int
    modified_time: datetime
    md5_checksum: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    is_folder: bool = False
    local_path: Optional[Path] = None

class GoogleDriveSync:
    """Google Drive synchronization manager for the project"""
    
    def __init__(self, config_path: str = "google_drive/drive_config.json"):
        """Initialize Google Drive sync manager"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.service = None
        self.creds = None
        self.sync_cache = {}
        self.last_sync = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def authenticate(self) -> None:
        """Authenticate with Google Drive API"""
        token_file = Path(self.config['authentication']['token_file'])
        creds_file = Path(self.config['authentication']['credentials_file'])
        
        # Check if token exists and is valid
        if token_file.exists():
            with open(token_file, 'rb') as token:
                self.creds = pickle.load(token)
        
        # If credentials are not valid, refresh or get new ones
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                logger.info("Refreshing authentication token...")
                self.creds.refresh(Request())
            else:
                if not creds_file.exists():
                    logger.error(f"Credentials file not found: {creds_file}")
                    logger.info("Please download credentials.json from Google Cloud Console")
                    logger.info("Visit: https://console.cloud.google.com/apis/credentials")
                    raise FileNotFoundError("Missing credentials.json")
                
                logger.info("Starting OAuth2 authentication flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file),
                    self.config['authentication']['scopes']
                )
                self.creds = flow.run_local_server(port=8080)
            
            # Save the credentials for the next run
            token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(token_file, 'wb') as token:
                pickle.dump(self.creds, token)
            logger.info("Authentication successful!")
        
        # Build the service
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def list_files(self, folder_id: str = 'root', max_results: int = 100) -> List[DriveFile]:
        """List files in a Google Drive folder"""
        if not self.service:
            self.authenticate()
        
        try:
            results = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, md5Checksum, parents)"
            ).execute()
            
            files = []
            for item in results.get('files', []):
                files.append(DriveFile(
                    id=item['id'],
                    name=item['name'],
                    mime_type=item['mimeType'],
                    size=int(item.get('size', 0)),
                    modified_time=datetime.fromisoformat(item['modifiedTime'].replace('Z', '+00:00')),
                    md5_checksum=item.get('md5Checksum'),
                    parent_ids=item.get('parents', []),
                    is_folder=item['mimeType'] == 'application/vnd.google-apps.folder'
                ))
            
            return files
            
        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return []
    
    def create_folder(self, name: str, parent_id: str = 'root') -> Optional[str]:
        """Create a folder in Google Drive"""
        if not self.service:
            self.authenticate()
        
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        try:
            file = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder: {name} (ID: {file.get('id')})")
            return file.get('id')
            
        except HttpError as error:
            logger.error(f"Failed to create folder: {error}")
            return None
    
    def upload_file(self, local_path: Path, parent_id: str = 'root', 
                   file_id: Optional[str] = None) -> Optional[str]:
        """Upload or update a file to Google Drive"""
        if not self.service:
            self.authenticate()
        
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return None
        
        mime_type = self._get_mime_type(local_path)
        file_metadata = {
            'name': local_path.name,
            'parents': [parent_id] if not file_id else []
        }
        
        media = MediaFileUpload(
            str(local_path),
            mimetype=mime_type,
            resumable=True
        )
        
        try:
            if file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=file_id,
                    body={'name': local_path.name},
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"Updated file: {local_path.name}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"Uploaded file: {local_path.name}")
            
            return file.get('id')
            
        except HttpError as error:
            logger.error(f"Failed to upload file: {error}")
            return None
    
    def download_file(self, file_id: str, local_path: Path) -> bool:
        """Download a file from Google Drive"""
        if not self.service:
            self.authenticate()
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.info(f"Download progress: {int(status.progress() * 100)}%")
            
            # Write to file
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            logger.info(f"Downloaded file: {local_path.name}")
            return True
            
        except HttpError as error:
            logger.error(f"Failed to download file: {error}")
            return False
    
    def sync_folder(self, local_folder: Path, drive_folder_id: str = 'root') -> Dict[str, Any]:
        """Synchronize a local folder with Google Drive"""
        if not self.service:
            self.authenticate()
        
        sync_results = {
            'uploaded': [],
            'downloaded': [],
            'updated': [],
            'skipped': [],
            'errors': []
        }
        
        # Get list of files in Drive folder
        drive_files = {f.name: f for f in self.list_files(drive_folder_id)}
        
        # Get list of local files
        local_files = {}
        for file_path in local_folder.rglob('*'):
            if file_path.is_file():
                # Check exclude patterns
                if any(file_path.match(pattern) for pattern in self.config['sync_settings']['exclude_patterns']):
                    continue
                
                # Check include patterns
                if self.config['sync_settings']['include_patterns']:
                    if not any(file_path.match(pattern) for pattern in self.config['sync_settings']['include_patterns']):
                        continue
                
                relative_path = file_path.relative_to(local_folder)
                local_files[relative_path.name] = file_path
        
        # Sync based on direction setting
        sync_direction = self.config['sync_settings']['sync_direction']
        
        if sync_direction in ['bidirectional', 'local_to_drive']:
            # Upload new or modified local files
            for name, local_path in local_files.items():
                try:
                    if name in drive_files:
                        # File exists in Drive, check if update needed
                        drive_file = drive_files[name]
                        local_md5 = self._calculate_md5(local_path)
                        
                        if local_md5 != drive_file.md5_checksum:
                            self.upload_file(local_path, drive_folder_id, drive_file.id)
                            sync_results['updated'].append(str(local_path))
                        else:
                            sync_results['skipped'].append(str(local_path))
                    else:
                        # New file, upload it
                        self.upload_file(local_path, drive_folder_id)
                        sync_results['uploaded'].append(str(local_path))
                        
                except Exception as e:
                    logger.error(f"Error syncing {local_path}: {e}")
                    sync_results['errors'].append(f"{local_path}: {str(e)}")
        
        if sync_direction in ['bidirectional', 'drive_to_local']:
            # Download new or modified Drive files
            for name, drive_file in drive_files.items():
                if drive_file.is_folder:
                    continue
                    
                local_path = local_folder / name
                
                try:
                    if name not in local_files:
                        # New file in Drive, download it
                        self.download_file(drive_file.id, local_path)
                        sync_results['downloaded'].append(str(local_path))
                    elif local_path.exists():
                        # File exists locally, check if update needed
                        local_md5 = self._calculate_md5(local_path)
                        
                        if local_md5 != drive_file.md5_checksum:
                            # Resolve conflict based on settings
                            if self.config['sync_settings']['conflict_resolution'] == 'newest_wins':
                                local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
                                if drive_file.modified_time > local_mtime:
                                    self.download_file(drive_file.id, local_path)
                                    sync_results['downloaded'].append(str(local_path))
                                else:
                                    self.upload_file(local_path, drive_folder_id, drive_file.id)
                                    sync_results['updated'].append(str(local_path))
                            
                except Exception as e:
                    logger.error(f"Error syncing {name}: {e}")
                    sync_results['errors'].append(f"{name}: {str(e)}")
        
        self.last_sync = datetime.now()
        return sync_results
    
    async def auto_sync(self) -> None:
        """Run automatic synchronization based on configured interval"""
        while True:
            try:
                for folder_config in self.config['local_sync_folders']:
                    if folder_config['sync_enabled']:
                        local_path = Path(folder_config['local_path'])
                        if local_path.exists():
                            logger.info(f"Syncing {local_path}...")
                            results = self.sync_folder(local_path)
                            logger.info(f"Sync complete: {results}")
                
                # Wait for next sync interval
                interval = self.config['sync_settings']['sync_interval_minutes']
                await asyncio.sleep(interval * 60)
                
            except Exception as e:
                logger.error(f"Auto-sync error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def backup_project(self) -> bool:
        """Create a backup of the project in Google Drive"""
        if not self.service:
            self.authenticate()
        
        try:
            # Create backup folder name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"tori_backup_{timestamp}"
            
            # Create backup folder in Drive
            backup_folder_id = self.create_folder(
                backup_name, 
                self.config['drive_folder']['parent_folder_id']
            )
            
            if not backup_folder_id:
                return False
            
            # Sync all configured folders to backup
            for folder_config in self.config['local_sync_folders']:
                local_path = Path(folder_config['local_path'])
                if local_path.exists():
                    # Create subfolder in backup
                    subfolder_name = local_path.name
                    subfolder_id = self.create_folder(subfolder_name, backup_folder_id)
                    
                    if subfolder_id:
                        results = self.sync_folder(local_path, subfolder_id)
                        logger.info(f"Backed up {local_path}: {results}")
            
            logger.info(f"Backup completed: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file"""
        mime_types = {
            '.txt': 'text/plain',
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.ts': 'application/typescript',
            '.json': 'application/json',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.css': 'text/css',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.zip': 'application/zip',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip'
        }
        
        suffix = file_path.suffix.lower()
        return mime_types.get(suffix, 'application/octet-stream')
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def main():
    """Main function to run Google Drive sync"""
    sync_manager = GoogleDriveSync()
    
    # Authenticate with Google Drive
    sync_manager.authenticate()
    
    # List files in root folder
    print("\n=== Files in Google Drive ===")
    files = sync_manager.list_files()
    for file in files[:10]:  # Show first 10 files
        print(f"- {file.name} ({file.mime_type})")
    
    # Sync configured folders
    print("\n=== Starting Synchronization ===")
    for folder_config in sync_manager.config['local_sync_folders']:
        if folder_config['sync_enabled']:
            local_path = Path(folder_config['local_path'])
            if local_path.exists():
                print(f"\nSyncing {local_path}...")
                results = sync_manager.sync_folder(local_path)
                print(f"Results: {json.dumps(results, indent=2)}")
    
    # Create a backup if configured
    if sync_manager.config['backup_settings']['auto_backup']:
        print("\n=== Creating Backup ===")
        if sync_manager.backup_project():
            print("Backup completed successfully!")
        else:
            print("Backup failed!")
    
    print("\n=== Sync Complete ===")


if __name__ == "__main__":
    main()
