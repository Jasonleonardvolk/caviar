# Google Drive Integration Setup

This guide will help you set up Google Drive integration for your project.

## Prerequisites

1. Python 3.7 or higher
2. Google account
3. Google Cloud Console access

## Installation

### Step 1: Install Required Dependencies

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Step 2: Set Up Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"

### Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen first:
   - Choose "External" for user type
   - Fill in the required fields
   - Add your email to test users
4. For Application type, choose "Desktop app"
5. Name it (e.g., "Tori Drive Sync")
6. Click "Create"
7. Download the credentials JSON file
8. Save it as `google_drive/credentials.json` in your project

## Configuration

The configuration file `drive_config.json` contains all settings for the sync:

### Main Configuration Sections:

1. **drive_folder**: Basic folder information
2. **sync_settings**: Controls how files are synchronized
3. **authentication**: OAuth2 settings
4. **local_sync_folders**: Maps local folders to Drive paths
5. **backup_settings**: Automatic backup configuration
6. **logging**: Logging preferences

### Sync Direction Options:

- `bidirectional`: Sync both ways (default)
- `local_to_drive`: Only upload to Drive
- `drive_to_local`: Only download from Drive

### Conflict Resolution:

- `newest_wins`: Keep the most recently modified version
- `local_wins`: Always keep local version
- `drive_wins`: Always keep Drive version

## Usage

### Basic Synchronization

```python
from google_drive.drive_sync import GoogleDriveSync

# Initialize sync manager
sync = GoogleDriveSync()

# Authenticate (will open browser on first run)
sync.authenticate()

# Sync a specific folder
from pathlib import Path
results = sync.sync_folder(Path("C:/Users/jason/Desktop/tori/kha"))
print(f"Sync results: {results}")
```

### Running from Command Line

```bash
# Run one-time sync
python google_drive/drive_sync.py

# For continuous sync (Windows)
python -c "import asyncio; from google_drive.drive_sync import GoogleDriveSync; sync = GoogleDriveSync(); asyncio.run(sync.auto_sync())"
```

### Create PowerShell Script for Easy Access

Create `sync_drive.ps1`:

```powershell
# Google Drive Sync Script
$scriptPath = "${IRIS_ROOT}\google_drive\drive_sync.py"
python $scriptPath
```

### Create Batch File for Windows

Create `sync_drive.bat`:

```batch
@echo off
cd /d ${IRIS_ROOT}
python google_drive\drive_sync.py
pause
```

## Automatic Synchronization

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily, on startup)
4. Action: Start a program
5. Program: `python.exe`
6. Arguments: `${IRIS_ROOT}\google_drive\drive_sync.py`
7. Start in: `${IRIS_ROOT}`

### Python Background Service

Create `auto_sync_service.py`:

```python
import asyncio
from google_drive.drive_sync import GoogleDriveSync

async def main():
    sync = GoogleDriveSync()
    await sync.auto_sync()

if __name__ == "__main__":
    asyncio.run(main())
```

## API Methods

### List Files
```python
files = sync.list_files(folder_id='root', max_results=100)
for file in files:
    print(f"{file.name} - {file.mime_type}")
```

### Upload File
```python
from pathlib import Path
file_id = sync.upload_file(Path("local_file.txt"), parent_id='root')
```

### Download File
```python
success = sync.download_file(file_id='xxx', local_path=Path("downloaded.txt"))
```

### Create Folder
```python
folder_id = sync.create_folder("New Folder", parent_id='root')
```

### Backup Project
```python
success = sync.backup_project()
```

## Exclude/Include Patterns

Configure in `drive_config.json`:

```json
"exclude_patterns": [
    "*.tmp",
    "*.cache",
    "__pycache__",
    ".git",
    "node_modules"
],
"include_patterns": [
    "*.py",
    "*.js",
    "*.json",
    "*.md"
]
```

## Troubleshooting

### Authentication Issues

1. Delete `google_drive/token.json` and re-authenticate
2. Ensure credentials.json is in the correct location
3. Check that Google Drive API is enabled

### Sync Conflicts

1. Check `conflict_resolution` setting in config
2. Review logs in `google_drive/sync.log`
3. Manually resolve by choosing sync direction

### Permission Errors

1. Ensure you have write permissions to local folders
2. Check Google Drive storage quota
3. Verify OAuth scopes include necessary permissions

## Security Notes

1. Never commit `credentials.json` or `token.json` to version control
2. Add to `.gitignore`:
   ```
   google_drive/credentials.json
   google_drive/token.json
   google_drive/*.log
   ```
3. Use environment variables for sensitive data if needed

## Advanced Features

### Custom Sync Logic

Extend the `GoogleDriveSync` class:

```python
class CustomDriveSync(GoogleDriveSync):
    def custom_sync_logic(self):
        # Your custom implementation
        pass
```

### Integration with Project

Add to your main application:

```python
# In your main app
from google_drive.drive_sync import GoogleDriveSync

class ToriApp:
    def __init__(self):
        self.drive_sync = GoogleDriveSync()
    
    def save_to_drive(self, data):
        # Save data and sync to Drive
        self.drive_sync.sync_folder(Path("data"))
```

## Support

For issues or questions:
1. Check the logs in `google_drive/sync.log`
2. Review Google Drive API documentation
3. Ensure all dependencies are installed correctly

## Next Steps

1. Download and place `credentials.json` in the google_drive folder
2. Run `python google_drive/drive_sync.py` to test authentication
3. Configure sync folders in `drive_config.json`
4. Set up automatic synchronization as needed
