"""
File hashing utilities for ELFIN.

This module provides utilities for hashing file contents, primarily used
for caching purposes to avoid redundant processing of unchanged files.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Union, List


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate the SHA-1 hash of a file's contents.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        The SHA-1 hash of the file's contents as a hex string
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    # Convert string path to Path object if necessary
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Calculate SHA-1 hash
    hasher = hashlib.sha1()
    
    # Read file in binary mode to avoid encoding issues
    with open(file_path, 'rb') as f:
        # Read and update hash in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    
    # Return the hash as a hex string
    return hasher.hexdigest()


class FileHashCache:
    """
    Cache for file hashes to track file changes.
    
    This class stores and manages SHA-1 hashes of files to efficiently
    determine if files have changed since they were last processed.
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize the hash cache.
        
        Args:
            cache_file: Path to the cache file. If None, defaults to ~/.elfin_cache/dim.json
        """
        # Default cache directory
        if cache_file is None:
            home_dir = Path.home()
            cache_dir = home_dir / '.elfin_cache'
            cache_dir.mkdir(exist_ok=True)
            self.cache_file = cache_dir / 'dim.json'
        else:
            self.cache_file = cache_file
        
        # Initialize the cache
        self.hashes: Dict[str, str] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load the cache from disk."""
        import json
        
        # Create empty cache if file doesn't exist
        if not self.cache_file.exists():
            self.hashes = {}
            return
        
        # Load existing cache
        try:
            with open(self.cache_file, 'r') as f:
                self.hashes = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Reset cache if it's corrupted or missing
            self.hashes = {}
    
    def _save_cache(self) -> None:
        """Save the cache to disk."""
        import json
        
        # Ensure the cache directory exists
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Save the cache
        with open(self.cache_file, 'w') as f:
            json.dump(self.hashes, f)
    
    def get_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get the stored hash for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The stored hash, or None if the file isn't in the cache
        """
        # Convert to absolute path string for consistent keys
        abs_path = str(Path(file_path).resolve())
        return self.hashes.get(abs_path)
    
    def update_hash(self, file_path: Union[str, Path], file_hash: Optional[str] = None) -> str:
        """
        Update the hash for a file.
        
        Args:
            file_path: Path to the file
            file_hash: The hash to store. If None, calculates the hash
            
        Returns:
            The new hash
        """
        # Convert to absolute path string for consistent keys
        abs_path = str(Path(file_path).resolve())
        
        # Calculate hash if not provided
        if file_hash is None:
            file_hash = get_file_hash(file_path)
        
        # Update the cache
        self.hashes[abs_path] = file_hash
        self._save_cache()
        
        return file_hash
    
    def is_unchanged(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is unchanged since it was last cached.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file's current hash matches the cached hash,
            False otherwise or if the file isn't in the cache
        """
        # Get the cached hash
        cached_hash = self.get_hash(file_path)
        
        # If the file isn't in the cache, it's considered changed
        if cached_hash is None:
            return False
        
        # Calculate the current hash
        current_hash = get_file_hash(file_path)
        
        # Compare hashes
        return cached_hash == current_hash
    
    def update_if_changed(self, file_path: Union[str, Path]) -> bool:
        """
        Update the hash for a file if it has changed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file was updated, False if it was unchanged
        """
        # Calculate the current hash
        current_hash = get_file_hash(file_path)
        
        # Get the cached hash
        abs_path = str(Path(file_path).resolve())
        cached_hash = self.hashes.get(abs_path)
        
        # Update the cache if the file has changed or isn't in the cache
        if cached_hash != current_hash:
            self.hashes[abs_path] = current_hash
            self._save_cache()
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear the cache."""
        self.hashes.clear()
        self._save_cache()
    
    def remove(self, file_path: Union[str, Path]) -> None:
        """
        Remove a file from the cache.
        
        Args:
            file_path: Path to the file to remove
        """
        # Convert to absolute path string for consistent keys
        abs_path = str(Path(file_path).resolve())
        
        # Remove the file from the cache
        if abs_path in self.hashes:
            del self.hashes[abs_path]
            self._save_cache()
