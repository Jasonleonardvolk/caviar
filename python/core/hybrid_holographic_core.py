"""
hybrid_holographic_core.py

Central orchestration that other services call.
This file shows how to import from the new barrels and integrate all components.
"""

from python.core import (
    phase_to_depth,
    load_phase_data,
    MeshExporter,
    MeshUpdateWatcher,
    AdapterLoader,
    encode_phase_map,
    inject_curvature,
)
import logging
import numpy as np
from typing import Optional, Dict, Any

class HybridHolographicCore:
    """
    Main orchestrator for the hybrid holographic system.
    Integrates phase processing, mesh management, and adapter loading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hybrid holographic core with configuration.
        
        Args:
            config: Dictionary containing:
                - adapters_dir: Directory for model adapters
                - mesh_summary_path: Path for mesh summary exports
                - wavelength: Optical wavelength for phase-to-depth conversion
                - f0: Reference frequency for phase-to-depth
                - mesh_update_interval: Seconds between mesh update checks
        """
        self.config = config
        self.loader: Optional[AdapterLoader] = None
        self.watcher: Optional[MeshUpdateWatcher] = None
        self.current_mesh = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, mesh_obj=None):
        """
        Initialize core components: adapter loader and mesh watcher.
        
        Args:
            mesh_obj: Initial mesh object (optional)
        """
        # Initialize adapter loader
        self.loader = AdapterLoader(
            adapters_dir=self.config.get('adapters_dir', './adapters'),
            max_loaded=self.config.get('max_adapters_loaded', 2),
            active_link_name="active_adapter"
        )
        
        # Initialize mesh if provided
        if mesh_obj:
            self.current_mesh = mesh_obj
        
        # Start mesh update watcher
        mesh_summary_path = self.config.get('mesh_summary_path', './data/mesh_summary.json')
        interval = self.config.get('mesh_update_interval', 300.0)
        
        self.watcher = MeshUpdateWatcher(
            self.current_mesh, 
            mesh_summary_path, 
            interval=interval
        )
        self.watcher.start()
        
        # Export initial mesh summary
        MeshExporter.export_summary(self.current_mesh, mesh_summary_path)
        
        self.logger.info("HybridHolographicCore initialized successfully")
        return self.loader, self.watcher
    
    def compute_depth_from_phase(self, phase_source_path: str) -> np.ndarray:
        """
        Convert phase map to depth map using Fourier-based unwrapping.
        
        Args:
            phase_source_path: Path to phase data file (.npy or image)
            
        Returns:
            Depth map as numpy array
        """
        wavelength = self.config.get('wavelength', 0.5e-6)  # Default 500nm
        f0 = self.config.get('f0', 1.0)
        
        phase = load_phase_data(phase_source_path)
        depth = phase_to_depth(phase, wavelength=wavelength, f0=f0)
        
        self.logger.info(f"Computed depth map from phase: shape={depth.shape}, "
                        f"range=[{depth.min():.3e}, {depth.max():.3e}]")
        return depth
    
    def switch_adapter(self, adapter_name: str, user: str = None):
        """
        Atomically switch to a different adapter.
        
        Args:
            adapter_name: Name of adapter to load
            user: Username for audit trail
            
        Returns:
            Loaded adapter object
        """
        if not self.loader:
            raise RuntimeError("Core not initialized. Call initialize() first.")
        
        try:
            adapter = self.loader.load_adapter(adapter_name, user=user)
            self.logger.info(f"Successfully switched to adapter: {adapter_name}")
            return adapter
        except Exception as e:
            self.logger.error(f"Failed to switch adapter: {e}")
            # Attempt rollback
            self.loader.rollback_adapter()
            raise
    
    def update_mesh(self, changes: Dict[str, Any], export_immediately: bool = True):
        """
        Update the mesh and optionally trigger immediate export.
        
        Args:
            changes: Dictionary of changes to apply to mesh
            export_immediately: If True, export summary immediately
        """
        # Apply changes to mesh
        if isinstance(self.current_mesh, dict):
            self.current_mesh.update(changes)
        else:
            # If mesh has custom update method
            if hasattr(self.current_mesh, 'update'):
                self.current_mesh.update(changes)
        
        # Mark as updated
        import time
        if isinstance(self.current_mesh, dict):
            self.current_mesh['last_updated'] = time.time()
        elif hasattr(self.current_mesh, 'last_updated'):
            self.current_mesh.last_updated = time.time()
        
        # Export if requested
        if export_immediately:
            mesh_summary_path = self.config.get('mesh_summary_path', './data/mesh_summary.json')
            MeshExporter.export_summary(self.current_mesh, mesh_summary_path)
            self.logger.info("Mesh updated and summary exported")
    
    def process_holographic_frame(self, phase_data: np.ndarray, 
                                 apply_curvature: bool = False) -> Dict[str, np.ndarray]:
        """
        Process a holographic frame with optional curvature injection.
        
        Args:
            phase_data: Input phase data array
            apply_curvature: Whether to apply curvature injection
            
        Returns:
            Dictionary containing processed data (depth, phase, etc.)
        """
        result = {}
        
        # Convert phase to depth
        wavelength = self.config.get('wavelength', 0.5e-6)
        f0 = self.config.get('f0', 1.0)
        depth = phase_to_depth(phase_data, wavelength, f0)
        result['depth'] = depth
        
        # Optionally apply curvature injection
        if apply_curvature and inject_curvature:
            try:
                curved_phase = inject_curvature(phase_data, self.current_mesh)
                result['curved_phase'] = curved_phase
            except Exception as e:
                self.logger.warning(f"Curvature injection failed: {e}")
        
        # Encode phase if encoder available
        if encode_phase_map:
            try:
                encoded = encode_phase_map(phase_data)
                result['encoded_phase'] = encoded
            except Exception as e:
                self.logger.warning(f"Phase encoding failed: {e}")
        
        result['original_phase'] = phase_data
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status including active adapter, mesh info, etc.
        
        Returns:
            Status dictionary
        """
        status = {
            'initialized': self.loader is not None,
            'active_adapter': self.loader.active_adapter if self.loader else None,
            'loaded_adapters': list(self.loader.loaded_adapters.keys()) if self.loader else [],
            'mesh_size': len(self.current_mesh) if isinstance(self.current_mesh, dict) else 'N/A',
            'watcher_running': self.watcher.is_alive() if self.watcher else False,
        }
        
        # Add mesh summary
        if isinstance(self.current_mesh, dict):
            status['mesh_summary'] = MeshExporter.generate_summary(self.current_mesh)
        
        return status
    
    def shutdown(self):
        """
        Clean shutdown of the core system.
        """
        # Stop mesh watcher
        if self.watcher and self.watcher.is_alive():
            # Note: Since it's a daemon thread, it will stop when main program exits
            self.logger.info("Mesh watcher will stop with program exit")
        
        # Final mesh export
        if self.current_mesh:
            mesh_summary_path = self.config.get('mesh_summary_path', './data/mesh_summary.json')
            MeshExporter.export_summary(self.current_mesh, mesh_summary_path)
            self.logger.info("Final mesh summary exported")
        
        self.logger.info("HybridHolographicCore shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import os
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    adapters_dir = os.path.join(temp_dir, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    
    # Create dummy adapters
    for i in range(3):
        adapter_file = os.path.join(adapters_dir, f"adapter_{i}.bin")
        with open(adapter_file, 'wb') as f:
            f.write(f"adapter_{i}_data".encode())
    
    # Configuration
    config = {
        'adapters_dir': adapters_dir,
        'mesh_summary_path': os.path.join(temp_dir, 'mesh_summary.json'),
        'wavelength': 0.5e-6,  # 500nm
        'f0': 1.0,
        'mesh_update_interval': 5.0,  # Check every 5 seconds for testing
        'max_adapters_loaded': 2,
    }
    
    # Initialize core
    core = HybridHolographicCore(config)
    
    # Create initial mesh
    initial_mesh = {
        'nodes': {'node1': {'data': 'test1'}},
        'edges': [],
        'metadata': {'version': '1.0.0'}
    }
    
    # Initialize with mesh
    core.initialize(initial_mesh)
    
    # Test adapter switching
    print("\n=== Testing Adapter Switching ===")
    core.switch_adapter('adapter_0.bin', user='test_user')
    print(f"Active adapter: {core.loader.active_adapter}")
    
    # Test mesh update
    print("\n=== Testing Mesh Update ===")
    core.update_mesh({'nodes': {'node2': {'data': 'test2'}}})
    
    # Test phase processing
    print("\n=== Testing Phase Processing ===")
    # Create synthetic phase data
    phase_test = np.random.rand(100, 100) * 2 * np.pi
    result = core.process_holographic_frame(phase_test, apply_curvature=False)
    print(f"Processed frame keys: {result.keys()}")
    print(f"Depth shape: {result['depth'].shape}")
    
    # Get system status
    print("\n=== System Status ===")
    status = core.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    # Wait a bit to see mesh watcher in action
    import time
    print("\n=== Testing Mesh Watcher ===")
    time.sleep(6)
    
    # Update mesh again
    core.update_mesh({'nodes': {'node3': {'data': 'test3'}}})
    
    # Final status
    print("\n=== Final Status ===")
    final_status = core.get_system_status()
    print(f"Mesh size: {final_status['mesh_size']}")
    print(f"Watcher running: {final_status['watcher_running']}")
    
    # Shutdown
    core.shutdown()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("\n=== Test Complete ===")