"""app_integrator.py - Integrates all ALAN components into a unified application.

This module provides the main entry point for ALAN, connecting all the components:
- PDF ingestion pipeline
- Memory sculptor (scheduled cleanup)
- Eigenfunction labeler
- Phase-coherent concept graph

It initializes these components and ensures they work together properly.
"""

import os
import logging
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    # Try absolute import first
    from pipeline import ingest_pdf_and_update_index
except ImportError:
    # Fallback to relative import
    from .pipeline import ingest_pdf_and_update_index
try:
    # Try absolute import first
    from memory_sculptor import MemorySculptor, run_memory_cleanup
except ImportError:
    # Fallback to relative import
    from .memory_sculptor import MemorySculptor, run_memory_cleanup
try:
    # Try absolute import first
    from eigenfunction_labeler import EigenfunctionLabeler, label_concept_eigenfunctions
except ImportError:
    # Fallback to relative import
    from .eigenfunction_labeler import EigenfunctionLabeler, label_concept_eigenfunctions
try:
    # Try absolute import first
    from phase_walk import PhaseCoherentWalk
except ImportError:
    # Fallback to relative import
    from .phase_walk import PhaseCoherentWalk
try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/alan.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alan_integrator")

class AlanSystem:
    """
    Main ALAN system class that integrates all components.
    
    This class:
    1. Initializes all components
    2. Schedules memory cleanup
    3. Provides a unified API for interacting with ALAN
    4. Manages system state and configuration
    """
    
    def __init__(
        self,
        concept_store_path: str = "data/concepts.npz",
        config_path: Optional[str] = None,
        auto_start: bool = True
    ):
        """
        Initialize the ALAN system.
        
        Args:
            concept_store_path: Path to the concept store
            config_path: Optional path to configuration file
            auto_start: Whether to automatically start services
        """
        self.concept_store_path = concept_store_path
        self.config = self._load_config(config_path)
        
        # Create log directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/memory_sculptor", exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Auto-start if requested
        if auto_start:
            self.start()
            
        logger.info(f"ALAN system initialized (store: {concept_store_path})")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "memory_sculptor": {
                "cleanup_interval_hours": 1.0,
                "min_entropy_threshold": 0.2,
                "redundancy_threshold": 0.85
            },
            "pdf_ingestion": {
                "max_concepts": 12,
                "dim": 16,
                "min_quality_score": 0.6,
                "apply_gating": True,
                "coherence_threshold": 0.7
            },
            "eigenfunction_labeler": {
                "min_concept_count": 2,
                "min_term_frequency": 2
            },
            "phase_walk": {
                "default_steps": 5,
                "coherence_threshold": 0.7
            }
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                
            # Merge with defaults
            for section, values in loaded_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
                    
            logger.info(f"Loaded configuration from {config_path}")
            return default_config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return default_config
    
    def _init_components(self) -> None:
        """Initialize all ALAN components."""
        # Memory sculptor (scheduled cleanup)
        mem_config = self.config["memory_sculptor"]
        self.memory_sculptor = MemorySculptor(
            concept_store_path=self.concept_store_path,
            cleanup_interval_hours=mem_config["cleanup_interval_hours"],
            min_entropy_threshold=mem_config["min_entropy_threshold"],
            redundancy_threshold=mem_config["redundancy_threshold"],
            log_dir="logs/memory_sculptor"
        )
        
        # Eigenfunction labeler
        self.eigenfunction_labeler = EigenfunctionLabeler()
        
        # Phase-coherent concept graph (will be populated as needed)
        self.concept_graph = PhaseCoherentWalk()
        
        # Save component references
        self.components = {
            "memory_sculptor": self.memory_sculptor,
            "eigenfunction_labeler": self.eigenfunction_labeler,
            "concept_graph": self.concept_graph
        }
        
    def start(self) -> None:
        """Start all ALAN services."""
        # Start memory sculptor scheduler
        self.memory_sculptor.start_scheduler()
        logger.info("Started memory sculptor scheduler")
        
        # Log system start
        logger.info(f"ALAN system started at {datetime.now().isoformat()}")
        
    def stop(self) -> None:
        """Stop all ALAN services."""
        # Stop memory sculptor scheduler
        self.memory_sculptor.stop_scheduler()
        logger.info("Stopped memory sculptor scheduler")
        
        # Log system stop
        logger.info(f"ALAN system stopped at {datetime.now().isoformat()}")
    
    def ingest_pdf(
        self, 
        pdf_path: str,
        json_output: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a PDF into ALAN's knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            json_output: Whether to generate JSON output
            
        Returns:
            Dictionary with ingestion results
        """
        config = self.config["pdf_ingestion"]
        json_out = f"{Path(pdf_path).stem}_concepts.json" if json_output else None
        
        result = ingest_pdf_and_update_index(
            pdf_path=pdf_path,
            index_path=self.concept_store_path,
            max_concepts=config["max_concepts"],
            dim=config["dim"],
            json_out=json_out,
            min_quality_score=config["min_quality_score"],
            apply_gating=config["apply_gating"],
            coherence_threshold=config["coherence_threshold"]
        )
        
        logger.info(f"Ingested PDF: {pdf_path} ({result.get('concept_count', 0)} concepts)")
        return result
    
    def force_memory_cleanup(self) -> Dict[str, Any]:
        """
        Force an immediate memory cleanup cycle.
        
        Returns:
            Dictionary with cleanup results
        """
        event = self.memory_sculptor.run_cleanup_cycle(force=True)
        logger.info(f"Forced memory cleanup: {event.original_concept_count} → {event.final_concept_count} concepts")
        return event.to_dict()
    
    def get_concept_path(
        self, 
        start_concept: str, 
        steps: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Generate a phase-coherent path starting from a concept.
        
        Args:
            start_concept: Starting concept name
            steps: Number of steps in the path
            
        Returns:
            List of (concept_name, coherence) pairs
        """
        # This would normally load concepts from the store and build the graph
        # For demonstration, we'll just use a placeholder warning
        logger.warning("get_concept_path() needs implementation with actual concept loading")
        return [(start_concept, 1.0)]  # Placeholder
    
    def label_eigenfunctions(self) -> Dict[str, str]:
        """
        Label all eigenfunctions in the system.
        
        Returns:
            Dictionary mapping eigenfunction IDs to human-readable labels
        """
        # This would normally load concepts and label eigenfunctions
        # For demonstration, we'll just use a placeholder warning
        logger.warning("label_eigenfunctions() needs implementation with actual concept loading")
        return {}  # Placeholder
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the ALAN system.
        
        Returns:
            Dictionary with system status information
        """
        # Memory sculptor status
        next_cleanup = self.memory_sculptor.next_cleanup_time
        time_to_next = (next_cleanup - datetime.now()).total_seconds() if next_cleanup else None
        
        # System metrics (these would normally be populated from actual data)
        # For demonstration, we're using placeholder values
        metrics = {
            "concept_count": 0,  # Placeholder
            "eigenfunction_count": 0,  # Placeholder
            "last_ingestion": None,  # Placeholder
            "memory_sculptor": {
                "running": self.memory_sculptor.scheduler_running,
                "next_cleanup": next_cleanup.isoformat() if next_cleanup else None,
                "seconds_to_next_cleanup": time_to_next,
                "cleanup_count": len(self.memory_sculptor.cleanup_history)
            }
        }
        
        return metrics

# Module-level convenience functions

def start_alan(
    concept_store_path: str = "data/concepts.npz",
    config_path: Optional[str] = None
) -> AlanSystem:
    """
    Start the ALAN system with default configuration.
    
    Args:
        concept_store_path: Path to the concept store
        config_path: Optional path to configuration file
        
    Returns:
        Initialized AlanSystem instance
    """
    system = AlanSystem(
        concept_store_path=concept_store_path,
        config_path=config_path,
        auto_start=True
    )
    return system

def process_pdf_directory(
    alan_system: AlanSystem,
    directory_path: str,
    recursive: bool = False,
    file_extensions: List[str] = [".pdf"]
) -> Dict[str, Any]:
    """
    Process all PDFs in a directory.
    
    Args:
        alan_system: AlanSystem instance
        directory_path: Path to directory containing PDFs
        recursive: Whether to process subdirectories
        file_extensions: List of file extensions to process
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "total_files": 0,
        "processed_files": 0,
        "rejected_files": 0,
        "total_concepts": 0,
        "files": []
    }
    
    # Find all matching files
    paths = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in file_extensions):
                paths.append(os.path.join(directory_path, file))
                
    results["total_files"] = len(paths)
    
    # Process each file
    for path in paths:
        try:
            result = alan_system.ingest_pdf(path, json_output=False)
            
            if result.get("status") == "rejected":
                results["rejected_files"] += 1
            else:
                results["processed_files"] += 1
                results["total_concepts"] += result.get("concept_count", 0)
                
            results["files"].append({
                "path": path,
                "status": result.get("status", "unknown"),
                "concept_count": result.get("concept_count", 0)
            })
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            results["files"].append({
                "path": path,
                "status": "error",
                "error": str(e)
            })
            
    # Run memory cleanup after processing
    alan_system.force_memory_cleanup()
    
    return results

if __name__ == "__main__":
    # Example usage
    alan = start_alan()
    print("ALAN system started. Press Ctrl+C to exit.")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ALAN...")
        alan.stop()
        print("ALAN system stopped.")
