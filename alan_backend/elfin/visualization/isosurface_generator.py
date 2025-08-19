"""
Isosurface generator for the visualization dashboard.

This module provides utilities for generating 3D isosurfaces from barrier and
Lyapunov functions for visualization.
"""

import numpy as np
import logging
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import trimesh for mesh generation
try:
    import trimesh
    import trimesh.creation
    import trimesh.exchange
    TRIMESH_AVAILABLE = True
except ImportError:
    logger.warning("Trimesh not available, fallback isosurface generation will be limited")
    TRIMESH_AVAILABLE = False

# Try to import skimage for marching cubes
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-image not available, fallback isosurface generation will be limited")
    SKIMAGE_AVAILABLE = False


def barrier_function(x: float, y: float, z: float, params: Dict[str, float] = None) -> float:
    """
    Example barrier function: B(x,y,z) = R^2 - x^2 - y^2 - z^2 (spherical region)
    
    Args:
        x, y, z: Coordinates
        params: Optional parameters for the barrier function
    
    Returns:
        Barrier function value
    """
    params = params or {}
    radius = params.get('radius', 2.0)
    return radius**2 - x**2 - y**2 - z**2


def lyapunov_function(x: float, y: float, z: float, params: Dict[str, float] = None) -> float:
    """
    Example Lyapunov function: V(x,y,z) = x^2 + y^2 + z^2 (quadratic)
    
    Args:
        x, y, z: Coordinates
        params: Optional parameters for the Lyapunov function
    
    Returns:
        Lyapunov function value
    """
    params = params or {}
    scale = params.get('scale', 1.0)
    return scale * (x**2 + y**2 + z**2)


def compute_field_values(function_type: str, grid_size: int = 30, 
                        bounds: Tuple[float, float] = (-3.0, 3.0),
                        params: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute field values on a 3D grid.
    
    Args:
        function_type: 'barrier' or 'lyapunov'
        grid_size: Number of points along each dimension
        bounds: (min, max) bounds for all dimensions
        params: Optional parameters for the function
    
    Returns:
        Tuple of (field values array, grid points)
    """
    # Create grid
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    z = np.linspace(bounds[0], bounds[1], grid_size)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute function values
    func = barrier_function if function_type == 'barrier' else lyapunov_function
    values = np.zeros_like(X)
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                values[i, j, k] = func(X[i, j, k], Y[i, j, k], Z[i, j, k], params)
    
    return values, (X, Y, Z)


def generate_isosurface_mesh(function_type: str, level_value: float, 
                           grid_size: int = 30, output_format: str = 'glb',
                           system_params: Dict[str, float] = None) -> Optional[Path]:
    """
    Generate an isosurface mesh from a barrier or Lyapunov function.
    
    Args:
        function_type: 'barrier' or 'lyapunov'
        level_value: Isosurface level
        grid_size: Number of points along each dimension
        output_format: Output format ('glb', 'obj', 'stl')
        system_params: Additional parameters for the system
    
    Returns:
        Path to the generated mesh file, or None if generation failed
    """
    if not (TRIMESH_AVAILABLE and SKIMAGE_AVAILABLE):
        logger.error("Cannot generate isosurface: required dependencies missing")
        return None
    
    # Compute field values
    try:
        logger.info(f"Generating {function_type} isosurface at level {level_value}")
        values, (X, Y, Z) = compute_field_values(function_type, grid_size, params=system_params)
        
        # Generate isosurface using marching cubes
        spacing = (
            (X.max() - X.min()) / (grid_size - 1),
            (Y.max() - Y.min()) / (grid_size - 1),
            (Z.max() - Z.min()) / (grid_size - 1)
        )
        
        vertices, faces, normals, _ = measure.marching_cubes(
            values, level_value, spacing=spacing
        )
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        # Center the mesh
        mesh.vertices -= mesh.centroid
        
        # Normalize size
        scale = 1.0 / max(mesh.extents)
        mesh.vertices *= scale
        
        # Create output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as tmp:
            output_path = tmp.name
        
        # Export mesh
        if output_format == 'glb':
            mesh.export(output_path, file_type='glb')
        elif output_format == 'obj':
            mesh.export(output_path, file_type='obj')
        elif output_format == 'stl':
            mesh.export(output_path, file_type='stl')
        else:
            mesh.export(output_path, file_type='glb')
        
        logger.info(f"Generated isosurface mesh: {output_path}")
        return Path(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate isosurface: {e}")
        return None


def generate_fallback_sphere(level_value: float, radius: float = 2.0, 
                           output_format: str = 'glb') -> Optional[Path]:
    """
    Generate a fallback sphere mesh when other methods fail.
    
    Args:
        level_value: Used to scale the sphere
        radius: Base radius
        output_format: Output format
    
    Returns:
        Path to the generated mesh file, or None if generation failed
    """
    if not TRIMESH_AVAILABLE:
        return None
    
    try:
        # Scale radius based on level value
        scaled_radius = radius * max(0.1, min(1.0, level_value))
        
        # Create sphere
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=scaled_radius)
        
        # Create output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}') as tmp:
            output_path = tmp.name
        
        # Export mesh
        mesh.export(output_path, file_type=output_format)
        
        logger.info(f"Generated fallback sphere: {output_path}")
        return Path(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate fallback sphere: {e}")
        return None


def ensure_outputs_directory(system_id: str) -> Path:
    """
    Ensure the outputs directory exists for a given system.
    
    Args:
        system_id: System ID
    
    Returns:
        Path to the system's output directory
    """
    outputs_dir = Path('./outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    system_dir = outputs_dir / system_id
    system_dir.mkdir(exist_ok=True)
    
    return system_dir


def get_or_generate_isosurface(system_id: str, function_type: str, 
                             level_value: float, regenerate: bool = False) -> Optional[Path]:
    """
    Get an existing isosurface file or generate a new one.
    
    Args:
        system_id: System ID
        function_type: 'barrier' or 'lyapunov'
        level_value: Isosurface level
        regenerate: Force regeneration even if file exists
    
    Returns:
        Path to the isosurface file, or None if not available
    """
    # Ensure outputs directory
    system_dir = ensure_outputs_directory(system_id)
    
    # Format level value for filename
    level_str = f"{level_value:.2f}".replace('-', 'n')
    
    # Construct filename
    filename = f"isosurface_{function_type}_{level_str}.glb"
    file_path = system_dir / filename
    
    # Return existing file if available and not regenerating
    if file_path.exists() and not regenerate:
        return file_path
    
    # Generate new isosurface
    try:
        # Try to generate proper isosurface
        mesh_path = generate_isosurface_mesh(
            function_type, level_value, grid_size=30, output_format='glb'
        )
        
        if mesh_path is None:
            # Fall back to simple sphere
            mesh_path = generate_fallback_sphere(level_value, output_format='glb')
        
        if mesh_path is None:
            return None
        
        # Copy to outputs directory
        with open(mesh_path, 'rb') as src, open(file_path, 'wb') as dst:
            dst.write(src.read())
        
        # Clean up temporary file
        os.unlink(mesh_path)
        
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to get or generate isosurface: {e}")
        return None


if __name__ == "__main__":
    # Test isosurface generation
    logging.basicConfig(level=logging.INFO)
    
    barrier_path = generate_isosurface_mesh('barrier', 0.5, grid_size=30)
    print(f"Barrier isosurface: {barrier_path}")
    
    lyapunov_path = generate_isosurface_mesh('lyapunov', 1.0, grid_size=30)
    print(f"Lyapunov isosurface: {lyapunov_path}")
