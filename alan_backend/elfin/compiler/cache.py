"""
ELFIN Module Cache.

This module provides caching functionality for compiled ELFIN modules
to improve performance through reuse of previously compiled modules.
"""

import os
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, Set, Tuple
from pathlib import Path
import time


class ModuleCache:
    """
    A cache for compiled ELFIN modules.
    
    This cache stores compiled modules and their metadata to avoid
    redundant parsing and compilation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize a module cache.
        
        Args:
            cache_dir: Directory for storing cached modules (optional)
        """
        # Use default cache directory if none provided
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            self.cache_dir = os.path.join(home_dir, ".elfin", "cache")
        else:
            self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        
        # Load the cache index
        self.index_file = os.path.join(self.cache_dir, "index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the cache index from disk.
        
        Returns:
            The cache index as a dictionary
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache index: {e}")
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def _compute_hash(self, file_path: str) -> str:
        """
        Compute a hash of the file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            A hash string representing the file content
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"Error computing hash for {file_path}: {e}")
            # Return a unique string based on the current time
            # This will ensure the file is not found in the cache
            return f"error-{time.time()}"
    
    def _get_cache_path(self, file_path: str, file_hash: str) -> str:
        """
        Get the path to the cached module file.
        
        Args:
            file_path: Path to the source file
            file_hash: Hash of the file content
            
        Returns:
            Path to the cached module file
        """
        # Create a directory structure based on the file path
        rel_path = os.path.normpath(file_path)
        safe_path = rel_path.replace(os.sep, "_").replace(":", "_")
        
        # Use the hash to ensure uniqueness
        cache_file = f"{safe_path}_{file_hash}.pkl"
        return os.path.join(self.cache_dir, cache_file)
    
    def has_module(self, file_path: str) -> bool:
        """
        Check if a module is in the cache.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            True if the module is in the cache, False otherwise
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the file exists
        if not os.path.exists(abs_path):
            return False
        
        # Compute the hash of the file content
        file_hash = self._compute_hash(abs_path)
        
        # Check if the file is in the index with the same hash
        if abs_path in self.index and self.index[abs_path]["hash"] == file_hash:
            # Check if the cached file exists
            cache_path = self._get_cache_path(abs_path, file_hash)
            if os.path.exists(cache_path):
                return True
        
        return False
    
    def get_module(self, file_path: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Get a module from the cache.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            A tuple of (compiled_module, metadata) if found, (None, {}) otherwise
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the file is in the memory cache
        if abs_path in self.memory_cache:
            return self.memory_cache[abs_path]
        
        # Check if the file exists
        if not os.path.exists(abs_path):
            return None, {}
        
        # Compute the hash of the file content
        file_hash = self._compute_hash(abs_path)
        
        # Check if the file is in the index with the same hash
        if abs_path in self.index and self.index[abs_path]["hash"] == file_hash:
            # Get the path to the cached module
            cache_path = self._get_cache_path(abs_path, file_hash)
            
            # Load the cached module
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        compiled_module, metadata = pickle.load(f)
                        
                        # Store in memory cache
                        self.memory_cache[abs_path] = (compiled_module, metadata)
                        
                        return compiled_module, metadata
            except Exception as e:
                print(f"Error loading cached module {abs_path}: {e}")
        
        return None, {}
    
    def put_module(self, file_path: str, compiled_module: Any, metadata: Dict[str, Any]) -> None:
        """
        Put a module in the cache.
        
        Args:
            file_path: Path to the source file
            compiled_module: The compiled module
            metadata: Metadata about the module (dependencies, etc.)
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the file exists
        if not os.path.exists(abs_path):
            return
        
        # Compute the hash of the file content
        file_hash = self._compute_hash(abs_path)
        
        # Update the index
        self.index[abs_path] = {
            "hash": file_hash,
            "timestamp": time.time()
        }
        
        # Get the path to the cached module
        cache_path = self._get_cache_path(abs_path, file_hash)
        
        # Save the compiled module
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((compiled_module, metadata), f)
                
            # Store in memory cache
            self.memory_cache[abs_path] = (compiled_module, metadata)
            
            # Save the index
            self._save_index()
        except Exception as e:
            print(f"Error caching module {abs_path}: {e}")
    
    def invalidate(self, file_path: str) -> None:
        """
        Invalidate a module in the cache.
        
        Args:
            file_path: Path to the source file
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Remove from memory cache
        if abs_path in self.memory_cache:
            del self.memory_cache[abs_path]
        
        # Check if the file is in the index
        if abs_path in self.index:
            # Get the hash from the index
            file_hash = self.index[abs_path]["hash"]
            
            # Get the path to the cached module
            cache_path = self._get_cache_path(abs_path, file_hash)
            
            # Remove the cached module
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"Error removing cached module {abs_path}: {e}")
            
            # Remove from the index
            del self.index[abs_path]
            
            # Save the index
            self._save_index()
    
    def clear(self) -> None:
        """Clear the entire cache."""
        # Clear the memory cache
        self.memory_cache.clear()
        
        # Clear the index
        self.index.clear()
        
        # Remove all cached modules
        for file in os.listdir(self.cache_dir):
            if file.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    print(f"Error removing cached file {file}: {e}")
        
        # Save the empty index
        self._save_index()
    
    def get_dependencies(self, file_path: str) -> Set[str]:
        """
        Get the dependencies of a module from the cache.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            A set of paths to modules that this module depends on
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if the file is in the cache
        _, metadata = self.get_module(abs_path)
        
        # Return dependencies if available
        return metadata.get("dependencies", set())
    
    def get_dependents(self, file_path: str) -> Set[str]:
        """
        Get the modules that depend on a module.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            A set of paths to modules that depend on this module
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Find modules that depend on this module
        dependents = set()
        
        # Check all modules in the cache
        for module_path in self.index:
            # Skip the module itself
            if module_path == abs_path:
                continue
            
            # Check if this module is a dependency
            dependencies = self.get_dependencies(module_path)
            if abs_path in dependencies:
                dependents.add(module_path)
        
        return dependents
    
    def invalidate_dependents(self, file_path: str, recursive: bool = True) -> None:
        """
        Invalidate a module and optionally its dependents.
        
        Args:
            file_path: Path to the source file
            recursive: Whether to recursively invalidate dependents
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Invalidate the module
        self.invalidate(abs_path)
        
        # Recursively invalidate dependents
        if recursive:
            for dependent in self.get_dependents(abs_path):
                self.invalidate_dependents(dependent, recursive=True)
