"""
Mesh Update Trigger and Export Hook

This module provides functionality to automatically export a summary of a "mesh" (knowledge graph or context data structure)
whenever it is updated. It includes both an immediate trigger for use within update routines and a background watcher as a fallback.

Components:
 - MeshExporter.export_summary(mesh, path): Generates and writes a summary of the mesh to a JSON file. Uses atomic write to avoid partial files.
 - MeshUpdateWatcher: A background daemon thread that periodically checks for mesh changes and triggers export if needed (useful as a fallback or in environments where direct trigger calls are not guaranteed).

Integration:
Call MeshExporter.export_summary() at points in code where the mesh is updated (after modifications) to ensure the summary is fresh.
Optionally, run MeshUpdateWatcher in the background to periodically ensure updates (or use a cron job) as a safety net.
"""
import os
import json
import threading
import time
import logging
import hashlib
from datetime import datetime
from pathlib import Path

class MeshExporter:
    @staticmethod
    def _log_to_file(message: str, level: str = "INFO"):
        """Write to persistent audit log file."""
        log_dir = Path(os.environ.get("TORI_LOG_DIR", "logs")) / "mesh"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "mesh_export.log"
        
        timestamp = datetime.utcnow().isoformat()
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp}Z | {level} | {message}\n")
            f.flush()
            os.fsync(f.fileno())
    
    @staticmethod
    def generate_summary(mesh) -> dict:
        """
        Generate a summary representation of the mesh data structure.
        This could include counts of nodes/edges, or any essential summary info.
        If the mesh object has its own summary method, use it.
        """
        # If the mesh has a custom summary method, use that
        if hasattr(mesh, "to_summary"):
            try:
                summary = mesh.to_summary()
                if isinstance(summary, dict):
                    return summary
            except Exception as e:
                logging.error("Mesh to_summary() failed: %s", e)
        # Otherwise, derive a basic summary based on common attributes or structure
        summary = {}
        try:
            # If mesh is a dict-like structure
            if isinstance(mesh, dict):
                summary['keys'] = list(mesh.keys())
                summary['size'] = len(mesh)
            # If mesh has 'nodes' or 'edges' attributes (e.g., a graph)
            if hasattr(mesh, "nodes") and hasattr(mesh, "edges"):
                try:
                    summary['node_count'] = len(mesh.nodes)  # assuming mesh.nodes is iterable
                except Exception:
                    summary['node_count'] = None
                try:
                    summary['edge_count'] = len(mesh.edges)
                except Exception:
                    summary['edge_count'] = None
            # Add a timestamp or version if available
            if hasattr(mesh, "version"):
                summary['version'] = getattr(mesh, "version")
            if hasattr(mesh, "last_updated"):
                summary['last_updated'] = getattr(mesh, "last_updated")
        except Exception as e:
            logging.warning("MeshExporter: Partial summary generated with error: %s", e)
        return summary

    @staticmethod
    def export_summary(mesh, path: str):
        """
        Export the mesh summary to a JSON file at the given path.
        This operation is atomic (writes to temp file and then renames).
        """
        summary_data = MeshExporter.generate_summary(mesh)
        # Prepare output directory if not exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        temp_path = path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
            logging.info("Mesh summary exported to %s", path)
            
            # Write to persistent audit log
            node_count = summary_data.get('node_count', len(summary_data.get('keys', [])))
            version = summary_data.get('version', 'unknown')
            MeshExporter._log_to_file(
                f"EXPORT | path={path} | version={version} | nodes={node_count} | size={os.path.getsize(path)} bytes"
            )
        except Exception as e:
            logging.error("Failed to write mesh summary to %s: %s", path, e)
            MeshExporter._log_to_file(
                f"EXPORT_FAILED | path={path} | error={str(e)}",
                level="ERROR"
            )
            # Clean up temp file if exists
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

class MeshUpdateWatcher(threading.Thread):
    """
    Background thread that periodically checks the mesh for changes and triggers summary export.
    Useful as a fallback mechanism to ensure the mesh summary stays updated.
    """
    def __init__(self, mesh, output_path: str, interval: float = 60.0):
        super().__init__()
        self.mesh = mesh
        self.output_path = output_path
        self.interval = interval
        self.daemon = True  # Daemonize thread so it won't block program exit
        # For change detection
        self._last_summary_hash = None
        self._last_version = getattr(mesh, "version", None)
        self._last_updated_ts = getattr(mesh, "last_updated", None)
    
    def run(self):
        logging.info("MeshUpdateWatcher started, checking every %s seconds.", self.interval)
        MeshExporter._log_to_file(
            f"WATCHER_START | interval={self.interval}s | output={self.output_path}"
        )
        while True:
            try:
                # Determine if mesh changed:
                changed = False
                # If mesh has version or timestamp, use those for quick check
                if hasattr(self.mesh, "version"):
                    current_version = getattr(self.mesh, "version")
                    if current_version != self._last_version:
                        changed = True
                        self._last_version = current_version
                if hasattr(self.mesh, "last_updated"):
                    current_ts = getattr(self.mesh, "last_updated")
                    if current_ts != self._last_updated_ts:
                        changed = True
                        self._last_updated_ts = current_ts
                # If no explicit version/timestamp, use summary hash
                if not hasattr(self.mesh, "version") and not hasattr(self.mesh, "last_updated"):
                    summary_data = MeshExporter.generate_summary(self.mesh)
                    # Compute a hash of the summary for comparison
                    summary_bytes = json.dumps(summary_data, sort_keys=True).encode('utf-8')
                    current_hash = hashlib.md5(summary_bytes).hexdigest()
                    if self._last_summary_hash != current_hash:
                        changed = True
                        self._last_summary_hash = current_hash
                if changed:
                    MeshExporter.export_summary(self.mesh, self.output_path)
                    MeshExporter._log_to_file(
                        f"WATCHER_TRIGGERED | version={getattr(self.mesh, 'version', 'unknown')}"
                    )
            except Exception as e:
                logging.error("MeshUpdateWatcher error: %s", e)
                MeshExporter._log_to_file(
                    f"WATCHER_ERROR | error={str(e)}",
                    level="ERROR"
                )
            # Sleep until next check
            time.sleep(self.interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # Create a dummy mesh (dictionary for demonstration)
    mesh = {"nodes": {"id1": "Node1"}, "edges": {}}
    # Initial export
    MeshExporter.export_summary(mesh, "mesh_summary.json")
    # Start background watcher
    watcher = MeshUpdateWatcher(mesh, "mesh_summary.json", interval=5.0)
    watcher.start()
    # Simulate mesh updates
    time.sleep(1)
    mesh["nodes"]["id2"] = "Node2"  # add a node
    mesh["last_updated"] = time.time()
    time.sleep(6)  # wait for watcher to detect and export
    mesh["nodes"]["id3"] = "Node3"
    mesh["last_updated"] = time.time()
    time.sleep(6)
    print("Final mesh summary content:", open("mesh_summary.json").read())