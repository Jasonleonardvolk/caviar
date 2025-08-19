"""
Adapter Loader - manages dynamic loading and switching of model adapters with caching and rollback.

This module provides a class AdapterLoader to handle:
 - Loading adapters (e.g., model weight deltas or configurations) with caching of recently used adapters (LRU).
 - Atomic symlink swapping to point to the currently active adapter on disk.
 - Integrity checking via SHA256 hashes (to ensure adapter files are not corrupted or tampered).
 - Rollback functionality to revert to the previous adapter if a new adapter fails or is invalid.
 - Audit logging of adapter load/unload events for traceability.
"""
import os
import json
import time
import hashlib
import logging
import threading
import datetime

# Process-wide lock to prevent concurrent adapter swaps
_ADAPTER_SWAP_LOCK = threading.Lock()

class AdapterLoader:
    def __init__(self, adapters_dir: str, max_loaded: int = 2, active_link_name: str = "active_adapter"):
        """
        Initialize the AdapterLoader.
        :param adapters_dir: Directory where adapter files/folders are stored. Also where active symlink will be.
        :param max_loaded: Maximum number of adapter models to keep loaded in memory at once (LRU cache size).
        :param active_link_name: Name of the symlink file for the active adapter.
        """
        self.adapters_dir = adapters_dir
        self.max_loaded = max_loaded
        self.active_link = os.path.join(adapters_dir, active_link_name)
        self.loaded_adapters = {}   # name -> loaded adapter object (placeholder or actual model)
        self.last_used = {}        # name -> last used timestamp
        self.active_adapter = None  # name of currently active adapter
        self.previous_adapter = None  # name of previously active adapter (for rollback)
        # Load manifest metadata if present
        self.manifest = {}
        manifest_path = os.path.join(adapters_dir, "metadata.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        if 'name' in entry:
                            self.manifest[entry['name']] = entry
                elif isinstance(data, dict):
                    self.manifest = data
            except Exception as e:
                logging.error("Failed to load adapter manifest: %s", e)
                self.manifest = {}
        # Ensure adapter directory exists
        os.makedirs(adapters_dir, exist_ok=True)
        logging.info("AdapterLoader initialized (dir=%s, adapters_in_manifest=%d)", adapters_dir, len(self.manifest))

    def list_adapters(self):
        """List available adapter names (from manifest or directory listing)."""
        if self.manifest:
            return list(self.manifest.keys())
        # If no manifest, list subdirectories/files in adapters_dir (excluding active link)
        try:
            entries = os.listdir(self.adapters_dir)
            names = []
            for entry in entries:
                if entry == os.path.basename(self.active_link) or entry == "metadata.json":
                    continue
                names.append(entry)
            return names
        except Exception as e:
            logging.error("Error listing adapters: %s", e)
            return []

    def _compute_sha256(self, file_path: str) -> str:
        """Compute SHA-256 hash of the given file."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _load_from_disk(self, adapter_path: str):
        """
        Simulate loading an adapter from disk into memory.
        In a real system, this would load model weights or configure the model with the adapter.
        Here we just return a dummy object or content to represent the loaded adapter.
        """
        logging.info("Loading adapter from %s", adapter_path)
        # For simulation, we can read the file content if it's a file, or list directory files.
        if os.path.isdir(adapter_path):
            # If directory, just mark it as loaded (could load actual model files)
            return {"adapter_dir": adapter_path}
        else:
            # If file, read its content (or part of it) as demonstration
            try:
                with open(adapter_path, 'rb') as f:
                    data = f.read(1024)  # read first 1KB for example
                return {"adapter_file": adapter_path, "content_sample": data}
            except Exception as e:
                logging.error("Error loading adapter file: %s", e)
                raise

    def _unload_adapter(self, name: str):
        """Unload an adapter from memory (placeholder for actual resource cleanup)."""
        if name in self.loaded_adapters:
            # In a real scenario, perform cleanup (e.g., free GPU memory).
            del self.loaded_adapters[name]
            logging.info("Unloaded adapter '%s' from memory (LRU cache evict).", name)
        if name in self.last_used:
            del self.last_used[name]

    def _switch_symlink(self, target_path: str):
        """Atomically update the active adapter symlink to point to target_path."""
        tmp_link = os.path.join(self.adapters_dir, ".tmp_active_link")
        try:
            # Remove temp link if exists
            if os.path.lexists(tmp_link):
                os.remove(tmp_link)
            os.symlink(target_path, tmp_link)
            os.replace(tmp_link, self.active_link)
        except OSError as e:
            logging.error("Symlink switch failed: %s", e)
            raise

    def load_adapter(self, name: str, user: str = None):
        """
        Thread-safe adapter loading with global lock.
        Load and activate the adapter by name.
        - Validates adapter integrity via manifest (SHA256) if available.
        - Loads the adapter into memory (if not already loaded).
        - Updates the active symlink to point to the adapter's files.
        - Evicts least recently used adapter if cache size exceeded.
        - Logs and returns the loaded adapter object.
        """
        with _ADAPTER_SWAP_LOCK:  # Hard guard against concurrent swaps
            adapter_path = None
            # Determine adapter path
            if name in self.manifest and 'path' in self.manifest[name]:
                # If manifest specifies a custom path (relative to adapters_dir or absolute)
                path_val = self.manifest[name]['path']
                adapter_path = path_val if os.path.isabs(path_val) else os.path.join(self.adapters_dir, path_val)
            else:
                adapter_path = os.path.join(self.adapters_dir, name)
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter '{name}' not found at {adapter_path}")
            # Integrity check via SHA256 if available
            if name in self.manifest and 'sha256' in self.manifest[name]:
                expected_hash = self.manifest[name]['sha256']
                # Determine file to hash: if adapter_path is directory, try a known file
                file_to_hash = adapter_path
                if os.path.isdir(adapter_path):
                    # If manifest provides a specific file name
                    manifest_entry = self.manifest.get(name, {})
                    main_file = manifest_entry.get('file') or manifest_entry.get('main')
                    if main_file:
                        candidate = os.path.join(adapter_path, main_file)
                        if os.path.isfile(candidate):
                            file_to_hash = candidate
                    elif os.path.isfile(os.path.join(adapter_path, name)):
                        file_to_hash = os.path.join(adapter_path, name)
                    else:
                        # If no clue, find the largest file in directory as a guess
                        files = [os.path.join(adapter_path, f) for f in os.listdir(adapter_path) if os.path.isfile(os.path.join(adapter_path, f))]
                        if files:
                            file_to_hash = max(files, key=os.path.getsize)
                try:
                    actual_hash = self._compute_sha256(file_to_hash)
                except Exception as e:
                    logging.error("Could not compute hash for %s: %s", file_to_hash, e)
                    raise
                if actual_hash.lower() != expected_hash.lower():
                    logging.error("Integrity check failed for adapter '%s': SHA256 mismatch.", name)
                    raise ValueError(f"Adapter '{name}' failed integrity check.")
            # Prepare to load adapter
            prev = self.active_adapter
            self.previous_adapter = prev
            # If adapter already loaded in memory, reuse it; otherwise load from disk
            adapter_obj = self.loaded_adapters.get(name)
            if adapter_obj is None:
                adapter_obj = self._load_from_disk(adapter_path)
                self.loaded_adapters[name] = adapter_obj
            # Update LRU usage
            self.last_used[name] = time.time()
            # Switch active symlink to new adapter path
            try:
                self._switch_symlink(adapter_path)
            except Exception as e:
                # On symlink failure, remove loaded adapter if it was newly loaded
                if name in self.loaded_adapters and adapter_obj is not None:
                    # If symlink fails, we unload the adapter to be safe (though it might still remain loaded in memory if desired)
                    self._unload_adapter(name)
                raise
            # Set new active adapter
            self.active_adapter = name
            # Manage cache size (LRU eviction if needed)
            if len(self.loaded_adapters) > self.max_loaded:
                # Find least recently used adapter that is not the active one
                oldest_name = None
                oldest_time = time.time()
                for adapter_name, last_time in self.last_used.items():
                    if adapter_name == name:
                        continue
                    if last_time < oldest_time:
                        oldest_time = last_time
                        oldest_name = adapter_name
                if oldest_name:
                    self._unload_adapter(oldest_name)
            
            # Append to persistent audit log file
            audit_log_path = os.path.join(os.environ.get("TORI_LOG_DIR", "logs"), "inference", "adapter_swap.log")
            os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
            with open(audit_log_path, 'a') as f:
                timestamp = datetime.datetime.utcnow().isoformat()
                f.write(f"{timestamp}Z | USER={user or 'unknown'} | ACTION=SWAP | ADAPTER={name} | PREVIOUS={prev}\n")
            
            logging.info("AUDIT: User %s activated adapter '%s'.", user or "unknown", name)
            return adapter_obj

    def rollback_adapter(self):
        """Rollback to the previously active adapter (if available)."""
        if not self.previous_adapter:
            logging.warning("No previous adapter to roll back to.")
            return
        target = self.previous_adapter
        bad = self.active_adapter
        try:
            self.load_adapter(target)
            logging.warning("AUDIT: Rolled back from adapter '%s' to '%s'.", bad, target)
        except Exception as e:
            logging.error("Rollback to adapter '%s' failed: %s", target, e)
        # After one rollback attempt, do not keep chaining older adapters
        self.previous_adapter = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile
    # Create a temporary directory to simulate adapters directory
    temp_dir = os.path.join(tempfile.gettempdir(), "adapters_demo")
    os.makedirs(temp_dir, exist_ok=True)
    # Create dummy adapter files
    adapter1_file = os.path.join(temp_dir, "adapter1.bin")
    adapter2_file = os.path.join(temp_dir, "adapter2.bin")
    with open(adapter1_file, 'wb') as f: 
        f.write(b"adapter1_data")
    with open(adapter2_file, 'wb') as f: 
        f.write(b"adapter2_data")
    # Calculate hashes
    h1 = hashlib.sha256(b"adapter1_data").hexdigest()
    h2 = hashlib.sha256(b"adapter2_data").hexdigest()
    # Write manifest file
    manifest_data = {
        "adapter1.bin": {"path": "adapter1.bin", "sha256": h1},
        "adapter2.bin": {"path": "adapter2.bin", "sha256": h2}
    }
    with open(os.path.join(temp_dir, "metadata.json"), 'w') as mf:
        json.dump(manifest_data, mf, indent=2)
    # Initialize loader
    loader = AdapterLoader(temp_dir, max_loaded=1)
    # List available adapters
    print("Available adapters:", loader.list_adapters())
    # Load first adapter
    a1 = loader.load_adapter("adapter1.bin", user="demo")
    print("Active adapter object:", a1)
    # Load second adapter (this should evict the first due to max_loaded=1)
    a2 = loader.load_adapter("adapter2.bin", user="demo")
    print("Active adapter object:", a2)
    # Attempt rollback to first adapter
    loader.rollback_adapter()
    print("After rollback, active adapter should be adapter1.bin again. Active:", loader.active_adapter)
