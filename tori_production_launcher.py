#!/usr/bin/env python3
"""
TORI Production Launcher - Clean, no experimental flags
"""
import os
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/penrose.yaml") -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'penrose': {
                'enable': True,
                'rank': 14,
                'min_spectral_gap': 1e-5,
                'laplacian_nodes': 6000
            },
            'performance': {
                'blas_threads': 'auto',
                'cache_projector': True
            },
            'logging': {
                'level': 'INFO',
                'log_gap': True
            }
        }
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config: dict) -> None:
    """Configure environment based on config"""
    perf = config.get('performance', {})
    
    # BLAS threads
    threads = perf.get('blas_threads', 'auto')
    if threads == 'auto':
        import multiprocessing
        threads = multiprocessing.cpu_count() // 2  # physical cores
    
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    
    logger.info(f"BLAS threads set to {threads}")

def initialize_penrose(config: dict) -> None:
    """Initialize Penrose with production config"""
    penrose_cfg = config.get('penrose', {})
    
    if not penrose_cfg.get('enable', True):
        logger.info("Penrose disabled in config")
        return
    
    # Import here to avoid issues if disabled
    from python.core.exotic_topologies_v2 import build_penrose_laplacian_large
    from python.core.penrose_microkernel_v3_production import configure, get_info
    
    # Configure microkernel
    rank = penrose_cfg.get('rank', 14)
    min_gap = penrose_cfg.get('min_spectral_gap', 1e-5)
    configure(rank=rank, min_spectral_gap=min_gap)
    
    # Build Laplacian
    nodes = penrose_cfg.get('laplacian_nodes', 6000)
    logger.info(f"Building Penrose Laplacian with {nodes} nodes...")
    L = build_penrose_laplacian_large(target_nodes=nodes)
    
    # Store globally for use
    import python.core.tori_globals as tg
    tg.PENROSE_LAPLACIAN = L
    
    info = get_info()
    logger.info(f"Penrose initialized: {info}")

def run_tori_main():
    """Main TORI entry point"""
    # Load config
    config = load_config()
    
    # Setup environment
    setup_environment(config)
    
    # Initialize Penrose if enabled
    initialize_penrose(config)
    
    # Run main TORI logic
    logger.info("TORI ready for production")
    
    # Your main TORI code here...
    # from python.core.tori_main import main
    # main()

if __name__ == "__main__":
    run_tori_main()
