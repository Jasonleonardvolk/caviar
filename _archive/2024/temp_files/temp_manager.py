# temp_manager.py
"""
TORI Temp File Manager - Bulletproof temp file handling for Windows
This module forces ALL temp operations to use our controlled directory
"""
import os
import tempfile
import shutil
import logging
import traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TORI_TEMP")

class TempFileManager:
    """Centralized temp file manager that forces all operations to our controlled directory"""
    
    def __init__(self, project_root: str = r'{PROJECT_ROOT}'):
        self.project_root = Path(project_root)
        self.temp_root = self.project_root / 'tmp'
        self.cleanup_mode = 'manual'  # 'manual' or 'auto'
        self.temp_files_registry = []
        
        # Create temp directory if it doesn't exist
        self.temp_root.mkdir(exist_ok=True)
        
        # Force Python to use our temp directory
        self._override_temp_dirs()
        
        # Create subdirectories for organization
        self.pdf_temp = self.temp_root / 'pdf_processing'
        self.upload_temp = self.temp_root / 'uploads'
        self.extract_temp = self.temp_root / 'extracts'
        
        for dir in [self.pdf_temp, self.upload_temp, self.extract_temp]:
            dir.mkdir(exist_ok=True)
        
        # Log initialization
        logger.info(f"[TORI TEMP] Initialized with root: {self.temp_root}")
        logger.info(f"[TORI TEMP] Cleanup mode: {self.cleanup_mode}")
        logger.info(f"[TORI TEMP] Python tempfile.gettempdir(): {tempfile.gettempdir()}")
        
        # Create session log
        self.session_log = self.temp_root / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._log_to_file(f"Session started at {datetime.now()}")
    
    def _override_temp_dirs(self):
        """Force ALL temp operations to use our directory"""
        temp_str = str(self.temp_root)
        
        # Set all possible temp environment variables
        os.environ['TMPDIR'] = temp_str
        os.environ['TEMP'] = temp_str
        os.environ['TMP'] = temp_str
        os.environ['TEMPDIR'] = temp_str
        
        # Force tempfile module to use our directory
        tempfile.tempdir = temp_str
        
        # Log the override
        logger.info(f"[TORI TEMP] Overrode all temp dirs to: {temp_str}")
    
    def _log_to_file(self, message: str):
        """Log to session file for debugging"""
        with open(self.session_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def create_temp_file(self, prefix: str = "tori_", suffix: str = "", 
                        subdir: str = None, content: bytes = None) -> str:
        """Create a temp file with full logging and error handling"""
        try:
            # Determine directory
            if subdir:
                target_dir = self.temp_root / subdir
                target_dir.mkdir(exist_ok=True)
            else:
                target_dir = self.temp_root
            
            # Create the temp file
            with tempfile.NamedTemporaryFile(
                dir=str(target_dir),
                prefix=prefix,
                suffix=suffix,
                delete=False
            ) as tmp:
                temp_path = tmp.name
                
                # Write content if provided
                if content:
                    tmp.write(content)
                
                # Register the file
                self.temp_files_registry.append({
                    'path': temp_path,
                    'created': datetime.now().isoformat(),
                    'size': len(content) if content else 0,
                    'purpose': f"{prefix}{suffix}"
                })
                
                # Log creation
                log_msg = f"Created temp file: {temp_path} (size: {len(content) if content else 0} bytes)"
                logger.info(f"[TORI TEMP] {log_msg}")
                self._log_to_file(log_msg)
                
                return temp_path
                
        except Exception as e:
            error_msg = f"Failed to create temp file: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"[TORI TEMP] {error_msg}")
            self._log_to_file(f"ERROR: {error_msg}")
            raise
    
    def cleanup_file(self, filepath: str, force: bool = False):
        """Clean up a specific temp file"""
        if self.cleanup_mode == 'manual' and not force:
            logger.info(f"[TORI TEMP] Skipping cleanup (manual mode): {filepath}")
            return
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"[TORI TEMP] Cleaned up: {filepath}")
                self._log_to_file(f"Cleaned up: {filepath}")
        except Exception as e:
            logger.error(f"[TORI TEMP] Failed to cleanup {filepath}: {e}")
    
    def cleanup_old_files(self, hours: int = 24):
        """Clean up files older than specified hours"""
        import time
        cutoff_time = time.time() - (hours * 3600)
        
        cleaned = 0
        for root, dirs, files in os.walk(self.temp_root):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned += 1
                except Exception as e:
                    logger.error(f"[TORI TEMP] Failed to clean old file {filepath}: {e}")
        
        logger.info(f"[TORI TEMP] Cleaned {cleaned} old files")
        return cleaned
    
    def get_status(self) -> Dict[str, Any]:
        """Get current temp directory status"""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(self.temp_root):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except:
                    pass
        
        return {
            'temp_root': str(self.temp_root),
            'cleanup_mode': self.cleanup_mode,
            'total_files': file_count,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'session_files': len(self.temp_files_registry),
            'python_tempdir': tempfile.gettempdir()
        }
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save an uploaded file to temp directory"""
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        return self.create_temp_file(
            prefix="upload_",
            suffix=f"_{safe_filename}",
            subdir="uploads",
            content=file_content
        )

# Global instance
temp_manager = TempFileManager()

# Helper functions for easy access
def get_temp_path(prefix: str = "tori_", suffix: str = "") -> str:
    """Quick helper to get a temp file path"""
    return temp_manager.create_temp_file(prefix=prefix, suffix=suffix)

def save_temp_file(content: bytes, prefix: str = "tori_", suffix: str = "") -> str:
    """Quick helper to save content to a temp file"""
    return temp_manager.create_temp_file(prefix=prefix, suffix=suffix, content=content)

def cleanup_temp_file(filepath: str):
    """Quick helper to cleanup a temp file"""
    temp_manager.cleanup_file(filepath)