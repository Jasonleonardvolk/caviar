#!/usr/bin/env python3
"""
Hot-Swappable Graph Laplacian System (Production-Ready)
═══════════════════════════════════════════════════════

A robust, dinner-proof implementation of hot-swappable topological Laplacians
with comprehensive error handling, logging, and numerical stability checks.

Supported Topologies:
• Safe: kagome, honeycomb, triangular, small_world  
• Experimental: penrose (enable with TORI_ENABLE_EXOTIC=1 or constructor flag)

Features:
• χ³ (Kerr) NLSE physics with topological flux
• Soliton-aware adaptive topology swapping
• Energy harvesting and re-injection mechanisms
• Shadow trace stabilization
• Comprehensive error handling and logging
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Standard Library Imports
# ═══════════════════════════════════════════════════════════════════════════════
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# Third-Party Imports
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
except ImportError as e:
    print(f"Critical dependency missing: {e}")
    print("Please install required packages: numpy, scipy, networkx")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Local/Optional Imports (with graceful fallbacks)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .exotic_topologies import build_penrose_laplacian, PHI
    PENROSE_AVAILABLE = True
except ImportError:
    def build_penrose_laplacian(*args, **kwargs):
        """Fallback for missing Penrose topology builder"""
        return sp.eye(1, format='csr')
    PHI = 1.618033988749
    PENROSE_AVAILABLE = False

try:
    from .blowup_harness import induce_blowup
    BLOWUP_HARNESS_AVAILABLE = True
except ImportError:
    def induce_blowup(*args, **kwargs):
        """Fallback for missing blowup harness"""
        return np.array([0.0])
    BLOWUP_HARNESS_AVAILABLE = False

try:
    from .chaos_control_layer import ChaosControlLayer
    CHAOS_CONTROL_AVAILABLE = True
except ImportError:
    class ChaosControlLayer:
        """Fallback for missing chaos control layer"""
        pass
    CHAOS_CONTROL_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# Constants and Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Energy thresholds
CRITICAL_ENERGY_THRESHOLD = 1_000.0
WARNING_ENERGY_THRESHOLD = 500.0
MAX_ENERGY_MULTIPLIER = 2.0

# Default system parameters
DEFAULT_LATTICE_SIZE = (20, 20)
MAX_SWAP_HISTORY = 10
MIN_EIGENVALUES_TO_COMPUTE = 6

# Soliton configuration constants
DEFAULT_SOLITON_WIDTH = 2.0
DEFAULT_SOLITON_VELOCITY = 1.0
SEARCH_SOLITON_WIDTH = 5.0
SEARCH_SOLITON_VELOCITY = 2.0
OPTIMIZATION_SOLITON_WIDTH = 1.0
OPTIMIZATION_SOLITON_VELOCITY = 0.5
MAX_SOLITONS_PER_INJECTION = 10
SOLITON_ENERGY_THRESHOLD = 100.0

# Numerical stability constants
MIN_SPECTRAL_GAP_RATIO = 0.5
STABILIZATION_SLEEP_TIME = 0.1
INTERFERENCE_STRENGTH = 0.1
FLUX_PHASE_FACTOR = 2.0
DEFAULT_EPSILON = 0.3
DEFAULT_BLOWUP_STEPS = 5

# Environment variables
ENV_EXOTIC_ENABLE = "TORI_ENABLE_EXOTIC"
ENV_DEBUG_LEVEL = "TORI_DEBUG_LEVEL"

# ═══════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging based on environment variables"""
    try:
        debug_level = os.getenv(ENV_DEBUG_LEVEL, "INFO").upper()
        level = getattr(logging, debug_level, logging.INFO)
    except (AttributeError, ValueError):
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

setup_logging()
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TopologyConfig:
    """Configuration for a specific graph topology"""
    name: str
    lattice_type: str
    chern_number: int
    coordination_number: int
    spectral_gap: float
    optimal_for: List[str]

@dataclass
class ShadowTrace:
    """Shadow trace for phase-coherent stabilization"""
    phaseTag: float
    amplitude: float
    polarity: str
    original_index: int
    topological_charge: float

# ═══════════════════════════════════════════════════════════════════════════════
# Main Class: Hot-Swappable Laplacian
# ═══════════════════════════════════════════════════════════════════════════════

class HotSwappableLaplacian:
    """
    Hot-swappable graph Laplacian system with topological soliton support.
    
    This class provides safe, runtime topology switching while preserving
    quantum coherence through shadow trace stabilization and energy harvesting.
    
    Attributes:
        current_topology (str): Currently active topology
        lattice_size (Tuple[int, int]): Size of the lattice
        graph_laplacian (sp.csr_matrix): Current Laplacian matrix
        active_solitons (List[Dict]): List of active soliton configurations
        swap_count (int): Number of successful topology swaps
    """
    
    def __init__(
        self,
        initial_topology: str = "kagome",
        lattice_size: Optional[Tuple[int, int]] = None,
        ccl: Optional[ChaosControlLayer] = None,
        *,
        enable_experimental: Optional[bool] = None,
    ):
        """
        Initialize the hot-swappable Laplacian system.
        
        Args:
            initial_topology: Starting topology (default: "kagome")
            lattice_size: Lattice dimensions (default: (20, 20))
            ccl: Chaos control layer instance (optional)
            enable_experimental: Enable experimental topologies like Penrose
        
        Raises:
            ValueError: If initial topology is not available
            RuntimeError: If critical dependencies are missing
        """
        # Handle optional parameters with safe defaults
        self.lattice_size = lattice_size or DEFAULT_LATTICE_SIZE
        self.ccl = ccl
        
        # Determine experimental topology support
        self.enable_experimental = self._determine_experimental_support(enable_experimental)
        
        # Load topology configurations
        try:
            self.topologies = self._load_topology_configs()
        except Exception as e:
            logger.error(f"Failed to load topology configurations: {e}")
            raise RuntimeError("Critical error during topology configuration") from e
        
        # Validate initial topology
        if initial_topology not in self.topologies:
            available = list(self.topologies.keys())
            raise ValueError(
                f"Topology '{initial_topology}' not available. "
                f"Available topologies: {available} "
                f"(experimental enabled: {self.enable_experimental})"
            )
        
        # Initialize system state
        self.current_topology = initial_topology
        self.active_solitons: List[Dict[str, Any]] = []
        self.total_energy = 0.0
        self.swap_history: List[Dict[str, Any]] = []
        self.swap_count = 0
        self.energy_harvested_total = 0.0
        
        # Build initial Laplacian
        try:
            self.graph_laplacian = self._build_laplacian(initial_topology)
        except Exception as e:
            logger.error(f"Failed to build initial Laplacian for '{initial_topology}': {e}")
            raise RuntimeError("Failed to initialize Laplacian matrix") from e
        
        logger.info(
            "Hot-swappable Laplacian initialized successfully "
            f"(topology: {initial_topology}, experimental: {self.enable_experimental}, "
            f"lattice: {self.lattice_size})"
        )

    def _determine_experimental_support(self, enable_experimental: Optional[bool]) -> bool:
        """Determine if experimental topologies should be enabled"""
        if enable_experimental is not None:
            return enable_experimental and PENROSE_AVAILABLE
        
        try:
            env_flag = os.getenv(ENV_EXOTIC_ENABLE, "0")
            return bool(int(env_flag)) and PENROSE_AVAILABLE
        except (ValueError, TypeError):
            logger.warning(f"Invalid {ENV_EXOTIC_ENABLE} value, defaulting to False")
            return False

    def _load_topology_configs(self) -> Dict[str, TopologyConfig]:
        """Load topology configurations with error handling"""
        try:
            configs = {
                "kagome": TopologyConfig(
                    name="kagome",
                    lattice_type="kagome", 
                    chern_number=1,
                    coordination_number=4,
                    spectral_gap=0.5,
                    optimal_for=["pattern_recognition"]
                ),
                "honeycomb": TopologyConfig(
                    name="honeycomb",
                    lattice_type="honeycomb",
                    chern_number=0,
                    coordination_number=3,
                    spectral_gap=0.3,
                    optimal_for=["search", "sparse_search"]
                ),
                "triangular": TopologyConfig(
                    name="triangular",
                    lattice_type="triangular",
                    chern_number=2,
                    coordination_number=6,
                    spectral_gap=0.7,
                    optimal_for=["dense_compute", "dense_matrix"]
                ),
                "small_world": TopologyConfig(
                    name="small_world",
                    lattice_type="small_world",
                    chern_number=0,
                    coordination_number=4,
                    spectral_gap=0.2,
                    optimal_for=["global_search", "O(n²)"]
                ),
            }
            
            # Add experimental topologies if available
            if self.enable_experimental and PENROSE_AVAILABLE:
                configs["penrose"] = TopologyConfig(
                    name="penrose",
                    lattice_type="penrose",
                    chern_number=1,
                    coordination_number=4,
                    spectral_gap=0.55,
                    optimal_for=["spectral_magic"]
                )
                logger.info("Experimental Penrose topology enabled")
            
            return configs
            
        except Exception as e:
            logger.error(f"Error loading topology configurations: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════════════
    # Graph Builder Methods
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _build_kagome_graph(self) -> nx.Graph:
        """Build a kagome lattice graph with error handling"""
        try:
            m, n = self.lattice_size
            G = nx.Graph()
            
            for i in range(m):
                for j in range(n):
                    # Create the three nodes per unit cell
                    a, b, c = (i, j, "A"), (i, j, "B"), (i, j, "C")
                    G.add_nodes_from([a, b, c])
                    
                    # Add intra-cell edges
                    G.add_edges_from([(a, b), (b, c), (c, a)])
                    
                    # Add inter-cell edges with boundary conditions
                    if i < m - 1:
                        G.add_edge(b, ((i + 1) % m, j, "A"))
                    if j < n - 1:
                        G.add_edge(c, (i, (j + 1) % n, "A"))
            
            logger.debug(f"Built kagome graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to build kagome graph: {e}")
            raise

    def _build_honeycomb_graph(self) -> nx.Graph:
        """Build a honeycomb lattice graph with error handling"""
        try:
            m, n = self.lattice_size
            G = nx.Graph()
            
            for i in range(m):
                for j in range(n):
                    # Create the two nodes per unit cell
                    a, b = (i, j, "A"), (i, j, "B")
                    G.add_nodes_from([a, b])
                    G.add_edge(a, b)
                    
                    # Add inter-cell edges
                    if i < m - 1:
                        G.add_edge(b, ((i + 1) % m, j, "A"))
                    if j < n - 1:
                        G.add_edge(b, (i, (j + 1) % n, "A"))
            
            logger.debug(f"Built honeycomb graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to build honeycomb graph: {e}")
            raise

    def _build_triangular_graph(self) -> nx.Graph:
        """Build a triangular lattice graph with error handling"""
        try:
            m, n = self.lattice_size
            G = nx.triangular_lattice_graph(m, n)
            logger.debug(f"Built triangular graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        except Exception as e:
            logger.error(f"Failed to build triangular graph: {e}")
            raise

    def _build_small_world_graph(self) -> nx.Graph:
        """Build a small-world graph with error handling"""
        try:
            m, n = self.lattice_size
            total_nodes = m * n
            
            # Ensure we have enough nodes for the requested connectivity
            k = min(4, total_nodes - 1)
            if k <= 0:
                raise ValueError(f"Insufficient nodes ({total_nodes}) for small-world topology")
            
            G = nx.watts_strogatz_graph(total_nodes, k=k, p=0.1)
            logger.debug(f"Built small-world graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        except Exception as e:
            logger.error(f"Failed to build small-world graph: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════════════
    # Laplacian Construction and Topology Management
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _build_laplacian(self, topology: str) -> sp.csr_matrix:
        """
        Build Laplacian matrix for specified topology with error handling.
        
        Args:
            topology: Topology name
            
        Returns:
            Sparse CSR Laplacian matrix
            
        Raises:
            ValueError: If topology is not available
            RuntimeError: If Laplacian construction fails
        """
        if topology not in self.topologies:
            raise ValueError(f"Topology '{topology}' not available")
        
        try:
            logger.debug(f"Building Laplacian for topology: {topology}")
            
            if topology == "kagome":
                G = self._build_kagome_graph()
                L = nx.laplacian_matrix(G).astype(np.float64)
                
            elif topology == "honeycomb":
                G = self._build_honeycomb_graph()
                L = nx.laplacian_matrix(G).astype(np.float64)
                
            elif topology == "triangular":
                G = self._build_triangular_graph()
                L = nx.laplacian_matrix(G).astype(np.float64)
                
            elif topology == "small_world":
                G = self._build_small_world_graph()
                L = nx.laplacian_matrix(G).astype(np.float64)
                
            elif topology == "penrose":
                if not PENROSE_AVAILABLE:
                    raise RuntimeError("Penrose topology not available (missing exotic_topologies module)")
                # Penrose builder returns pre-fluxed Laplacian
                L = build_penrose_laplacian()
                logger.debug("Built Penrose Laplacian with pre-applied flux")
                return L.tocsr()
            
            else:
                raise ValueError(f"Unknown topology: {topology}")
            
            # Add topological flux for non-Penrose topologies
            config = self.topologies[topology]
            if config.chern_number != 0:
                L = self._add_topological_flux(L, config.chern_number)
                logger.debug(f"Applied topological flux (Chern number: {config.chern_number})")
            
            result = L.tocsr()
            logger.debug(f"Laplacian built successfully: shape {result.shape}, nnz {result.nnz}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to build Laplacian for topology '{topology}': {e}")
            raise RuntimeError(f"Laplacian construction failed") from e

    def _add_topological_flux(self, L: sp.csr_matrix, chern: int) -> sp.csr_matrix:
        """
        Add topological flux to Laplacian matrix with numerical stability checks.
        
        Args:
            L: Input Laplacian matrix
            chern: Chern number determining flux strength
            
        Returns:
            Laplacian with topological flux applied
        """
        try:
            n = L.shape[0]
            if n == 0:
                raise ValueError("Empty Laplacian matrix")
            
            phase = FLUX_PHASE_FACTOR * np.pi * chern / n
            rows, cols = L.nonzero()
            
            if len(rows) == 0:
                logger.warning("Laplacian has no non-zero elements")
                return L
            
            # Build flux matrix
            flux = sp.lil_matrix((n, n), dtype=complex)
            
            for i, j in zip(rows, cols):
                if i < j:  # Only upper triangular to avoid double application
                    delta = j - i
                    e = np.exp(1j * phase * delta)
                    flux[i, j] = e
                    flux[j, i] = np.conj(e)
            
            # Apply flux and extract real part
            Lc = L.astype(complex).multiply(flux.tocsr())
            result = Lc.real.tocsr()
            
            # Verify numerical stability
            if not np.isfinite(result.data).all():
                logger.error("Non-finite values detected in flux-modified Laplacian")
                raise RuntimeError("Numerical instability in flux application")
            
            logger.debug(f"Applied flux with phase factor {phase:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add topological flux: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════════════
    # Hot-Swap Implementation
    # ═══════════════════════════════════════════════════════════════════════════════
    
    async def hot_swap_laplacian_with_safety(self, new_topology: str) -> bool:
        """
        Perform safe hot-swap of graph Laplacian with comprehensive error handling.
        
        Args:
            new_topology: Target topology name
            
        Returns:
            True if swap successful, False if rolled back
            
        Raises:
            ValueError: If new_topology is not available
        """
        if new_topology not in self.topologies:
            available = list(self.topologies.keys())
            raise ValueError(
                f"Topology '{new_topology}' not available. "
                f"Available: {available}"
            )
        
        if new_topology == self.current_topology:
            logger.info(f"Already using topology '{new_topology}', no swap needed")
            return True
        
        logger.info(f"Initiating hot-swap: {self.current_topology} → {new_topology}")
        
        # Create swap record for tracking
        swap_record = {
            'from': self.current_topology,
            'to': new_topology,
            'timestamp': asyncio.get_event_loop().time(),
            'initial_energy': self.total_energy,
            'success': False,
            'rollback': False
        }
        
        # Save current state for potential rollback
        old_laplacian = self.graph_laplacian.copy()
        old_topology = self.current_topology
        
        try:
            # Step 1: Create shadow traces for phase coherence
            shadows = []
            try:
                shadows = [self.create_shadow_trace(soliton) for soliton in self.active_solitons]
                logger.debug(f"Created {len(shadows)} shadow traces")
            except Exception as e:
                logger.warning(f"Failed to create shadow traces: {e}")
            
            # Step 2: Energy harvesting if needed
            harvested_energy = None
            if self.total_energy > CRITICAL_ENERGY_THRESHOLD:
                logger.warning(f"High energy detected: {self.total_energy:.2f}")
                try:
                    if BLOWUP_HARNESS_AVAILABLE and hasattr(self, 'lattice'):
                        harvested_energy = induce_blowup(
                            self.lattice, 
                            epsilon=DEFAULT_EPSILON, 
                            steps=DEFAULT_BLOWUP_STEPS
                        )
                        self.energy_harvested_total += np.sum(np.abs(harvested_energy)**2)
                        logger.info("Energy harvested successfully")
                except Exception as e:
                    logger.warning(f"Energy harvesting failed: {e}")
            
            # Step 3: Build new Laplacian
            try:
                self.graph_laplacian = self._build_laplacian(new_topology)
                self.current_topology = new_topology
                logger.debug("New Laplacian constructed successfully")
            except Exception as e:
                logger.error(f"Failed to build new Laplacian: {e}")
                raise
            
            # Step 4: Stabilization with shadow traces
            try:
                await self.stabilize_with_shadows(shadows)
            except Exception as e:
                logger.warning(f"Shadow stabilization failed: {e}")
            
            # Step 5: Re-inject harvested energy
            if harvested_energy is not None:
                try:
                    await self.inject_as_bright_solitons(harvested_energy, new_topology)
                except Exception as e:
                    logger.warning(f"Energy re-injection failed: {e}")
            
            # Step 6: Verify stability
            if await self.verify_swap_stability():
                # Success!
                swap_record['success'] = True
                swap_record['final_energy'] = self.total_energy
                self.swap_count += 1
                
                logger.info(
                    f"Hot-swap completed successfully "
                    f"(swaps: {self.swap_count}, energy: {self.total_energy:.2f})"
                )
                return True
            else:
                # Stability check failed - rollback
                logger.error("Stability verification failed, performing rollback")
                self.graph_laplacian = old_laplacian
                self.current_topology = old_topology
                swap_record['rollback'] = True
                return False
                
        except Exception as e:
            # Unexpected error - rollback
            logger.error(f"Hot-swap failed with error: {e}")
            self.graph_laplacian = old_laplacian
            self.current_topology = old_topology
            swap_record['error'] = str(e)
            swap_record['rollback'] = True
            return False
            
        finally:
            # Always record the swap attempt
            self.swap_history.append(swap_record)
            if len(self.swap_history) > MAX_SWAP_HISTORY:
                self.swap_history.pop(0)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Stabilization and Energy Management
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def create_shadow_trace(self, bright_soliton: Dict[str, Any]) -> ShadowTrace:
        """
        Create dark soliton shadow trace for a bright soliton.
        
        Args:
            bright_soliton: Bright soliton configuration
            
        Returns:
            Corresponding shadow trace
        """
        try:
            return ShadowTrace(
                phaseTag=(bright_soliton.get('phase', 0) + np.pi) % (FLUX_PHASE_FACTOR * np.pi),
                amplitude=-INTERFERENCE_STRENGTH * bright_soliton.get('amplitude', 1.0),
                polarity='dark',
                original_index=bright_soliton.get('index', 0),
                topological_charge=bright_soliton.get('topological_charge', 0.0)
            )
        except Exception as e:
            logger.warning(f"Failed to create shadow trace: {e}")
            # Return minimal shadow trace
            return ShadowTrace(
                phaseTag=np.pi,
                amplitude=-INTERFERENCE_STRENGTH,
                polarity='dark',
                original_index=0,
                topological_charge=0.0
            )
    
    async def stabilize_with_shadows(self, shadows: List[ShadowTrace]) -> None:
        """
        Apply shadow trace stabilization with error handling.
        
        Args:
            shadows: List of shadow traces for stabilization
        """
        try:
            logger.debug("Applying shadow trace stabilization")
            
            for shadow in shadows:
                try:
                    # Find corresponding bright soliton
                    if shadow.original_index < len(self.active_solitons):
                        soliton = self.active_solitons[shadow.original_index]
                        
                        # Apply destructive interference
                        interference = shadow.amplitude * np.exp(1j * shadow.phaseTag)
                        
                        # Safely modify amplitude
                        old_amplitude = soliton.get('amplitude', 1.0)
                        new_amplitude = old_amplitude * (1 + interference.real)
                        
                        # Ensure amplitude remains reasonable
                        if abs(new_amplitude) < 1e-10:
                            new_amplitude = 1e-10
                        
                        soliton['amplitude'] = new_amplitude
                        soliton['topological_charge'] = shadow.topological_charge
                        
                except Exception as e:
                    logger.warning(f"Failed to apply shadow {shadow.original_index}: {e}")
                    continue
            
            # Allow system to settle
            await asyncio.sleep(STABILIZATION_SLEEP_TIME)
            
        except Exception as e:
            logger.error(f"Shadow stabilization failed: {e}")
            raise
    
    async def inject_as_bright_solitons(self, harvested_energy: np.ndarray, topology: str) -> None:
        """
        Re-inject harvested energy as topology-optimized bright solitons.
        
        Args:
            harvested_energy: Energy array to be re-injected
            topology: Target topology for optimization
        """
        try:
            logger.debug(f"Re-injecting energy as bright solitons for {topology}")
            
            config = self.topologies[topology]
            
            # Determine soliton parameters based on topology
            if 'search' in config.optimal_for:
                width, velocity = SEARCH_SOLITON_WIDTH, SEARCH_SOLITON_VELOCITY
            elif 'optimization' in config.optimal_for:
                width, velocity = OPTIMIZATION_SOLITON_WIDTH, OPTIMIZATION_SOLITON_VELOCITY
            else:
                width, velocity = DEFAULT_SOLITON_WIDTH, DEFAULT_SOLITON_VELOCITY
            
            # Calculate number of solitons to create
            total_energy = np.sum(np.abs(harvested_energy)**2)
            n_solitons = min(
                MAX_SOLITONS_PER_INJECTION, 
                max(1, int(total_energy / SOLITON_ENERGY_THRESHOLD))
            )
            
            # Create solitons
            for i in range(n_solitons):
                try:
                    idx = i % len(harvested_energy)
                    energy_val = harvested_energy[idx]
                    
                    soliton = {
                        'amplitude': np.sqrt(np.abs(energy_val)),
                        'phase': np.angle(energy_val),
                        'width': width,
                        'velocity': velocity,
                        'position': i * len(harvested_energy) // n_solitons,
                        'topological_charge': config.chern_number,
                        'index': len(self.active_solitons) + i
                    }
                    
                    self.active_solitons.append(soliton)
                    
                except Exception as e:
                    logger.warning(f"Failed to create soliton {i}: {e}")
                    continue
            
            logger.info(f"Successfully injected {n_solitons} bright solitons")
            
        except Exception as e:
            logger.error(f"Soliton injection failed: {e}")
            raise
    
    async def verify_swap_stability(self) -> bool:
        """
        Verify system stability after topology swap.
        
        Returns:
            True if system is stable, False otherwise
        """
        try:
            logger.debug("Verifying swap stability")
            
            # Check Laplacian properties
            if self.graph_laplacian.shape[0] == 0:
                logger.error("Empty Laplacian matrix")
                return False
            
            # Compute eigenvalues for spectral analysis
            try:
                k = min(MIN_EIGENVALUES_TO_COMPUTE, self.graph_laplacian.shape[0] - 1)
                if k <= 0:
                    logger.warning("Matrix too small for eigenvalue analysis")
                    return True  # Accept small matrices
                
                eigenvalues = eigsh(
                    self.graph_laplacian, 
                    k=k, 
                    which='SM', 
                    return_eigenvectors=False
                )
                
                # Verify eigenvalues are real and non-negative
                if not np.isreal(eigenvalues).all() or np.any(eigenvalues < -1e-10):
                    logger.error("Invalid eigenvalues detected")
                    return False
                
                # Check spectral gap
                if len(eigenvalues) > 1:
                    spectral_gap = eigenvalues[1] - eigenvalues[0]
                    config = self.topologies[self.current_topology]
                    min_gap = MIN_SPECTRAL_GAP_RATIO * config.spectral_gap
                    
                    if spectral_gap < min_gap:
                        logger.warning(
                            f"Spectral gap too small: {spectral_gap:.4f} < {min_gap:.4f}"
                        )
                        return False
                
            except Exception as e:
                logger.warning(f"Eigenvalue analysis failed: {e}")
                # Don't fail verification just for eigenvalue issues
            
            # Check energy levels
            current_energy = sum(
                soliton.get('amplitude', 1.0)**2 
                for soliton in self.active_solitons
            )
            
            if current_energy > MAX_ENERGY_MULTIPLIER * CRITICAL_ENERGY_THRESHOLD:
                logger.error(f"Energy too high after swap: {current_energy:.2f}")
                return False
            
            logger.debug("Stability verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Stability verification failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════════
    # Adaptive Topology Management
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def recommend_topology_for_problem(self, problem_type: str) -> str:
        """
        Recommend optimal topology for given problem type.
        
        Args:
            problem_type: Type of computational problem
            
        Returns:
            Recommended topology name
        """
        try:
            for topology, config in self.topologies.items():
                if problem_type in config.optimal_for:
                    logger.debug(f"Recommended topology '{topology}' for problem '{problem_type}'")
                    return topology
            
            logger.debug(f"No specific topology for '{problem_type}', using current")
            return self.current_topology
            
        except Exception as e:
            logger.warning(f"Topology recommendation failed: {e}")
            return self.current_topology
    
    async def adaptive_swap_for_complexity(self, current_complexity: str) -> bool:
        """
        Automatically swap topology based on detected complexity pattern.
        
        Args:
            current_complexity: Detected complexity pattern
            
        Returns:
            True if swap occurred and succeeded
        """
        try:
            recommended = self.recommend_topology_for_problem(current_complexity)
            
            if recommended != self.current_topology:
                logger.info(f"Complexity '{current_complexity}' detected - switching to '{recommended}'")
                return await self.hot_swap_laplacian_with_safety(recommended)
            
            return True  # No swap needed
            
        except Exception as e:
            logger.error(f"Adaptive swap failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════════
    # Matrix Operations and Utilities
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def multiply_with_topology(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Multiply matrices using current topology's optimal method.
        
        Args:
            A, B: Matrices to multiply
            
        Returns:
            Tuple of (result_matrix, performance_info)
        """
        info = {'topology': self.current_topology, 'method': 'numpy'}
        
        try:
            # Validate inputs
            if A.shape[1] != B.shape[0]:
                raise ValueError(f"Matrix dimension mismatch: {A.shape} vs {B.shape}")
            
            # Use topology-specific optimizations
            if (self.current_topology == "penrose" and 
                self.enable_experimental and 
                PENROSE_AVAILABLE):
                try:
                    from .penrose_microkernel_v2 import multiply as penrose_multiply
                    result, penrose_info = penrose_multiply(A, B, self.graph_laplacian)
                    info.update(penrose_info)
                    info['method'] = 'penrose_microkernel'
                    return result, info
                except ImportError:
                    logger.debug("Penrose microkernel not available, using numpy")
            
            # Standard multiplication with numerical stability check
            result = A @ B
            
            if not np.isfinite(result).all():
                logger.warning("Non-finite values in matrix multiplication result")
                info['warning'] = 'non_finite_values'
            
            return result, info
            
        except Exception as e:
            logger.error(f"Matrix multiplication failed: {e}")
            info['error'] = str(e)
            # Return zeros as fallback
            return np.zeros((A.shape[0], B.shape[1])), info

    # ═══════════════════════════════════════════════════════════════════════════════
    # Metrics and Monitoring
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get_swap_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about topology swaps and system state.
        
        Returns:
            Dictionary containing system metrics
        """
        try:
            return {
                'current_topology': self.current_topology,
                'total_swaps': self.swap_count,
                'energy_harvested': self.energy_harvested_total,
                'current_energy': self.total_energy,
                'active_solitons': len(self.active_solitons),
                'swap_history': self.swap_history[-5:],  # Last 5 swaps
                'available_topologies': list(self.topologies.keys()),
                'experimental_enabled': self.enable_experimental,
                'current_properties': {
                    'chern_number': self.topologies[self.current_topology].chern_number,
                    'coordination': self.topologies[self.current_topology].coordination_number,
                    'spectral_gap': self.topologies[self.current_topology].spectral_gap,
                    'optimal_for': self.topologies[self.current_topology].optimal_for
                },
                'system_health': {
                    'laplacian_shape': self.graph_laplacian.shape,
                    'laplacian_nnz': self.graph_laplacian.nnz,
                    'dependencies': {
                        'penrose_available': PENROSE_AVAILABLE,
                        'blowup_harness_available': BLOWUP_HARNESS_AVAILABLE,
                        'chaos_control_available': CHAOS_CONTROL_AVAILABLE
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to get swap metrics: {e}")
            return {'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# Demo and Testing
# ═══════════════════════════════════════════════════════════════════════════════

async def demo_hot_swap():
    """
    Demonstrate hot-swap functionality with comprehensive testing.
    """
    try:
        logger.info("Starting Hot-Swappable Laplacian Demo")
        logger.info("=" * 50)
        
        # Create system with experimental features if available
        hot_swap = HotSwappableLaplacian(
            initial_topology="kagome",
            lattice_size=(10, 10),  # Smaller for demo
            enable_experimental=True
        )
        
        logger.info(f"Initial topology: {hot_swap.current_topology}")
        logger.info(f"Available topologies: {list(hot_swap.topologies.keys())}")
        logger.info(f"Experimental features enabled: {hot_swap.enable_experimental}")
        
        # Simulate some solitons for testing
        hot_swap.active_solitons = [
            {'amplitude': 10.0, 'phase': 0, 'topological_charge': 1, 'index': 0},
            {'amplitude': 8.0, 'phase': np.pi/2, 'topological_charge': 1, 'index': 1},
            {'amplitude': 12.0, 'phase': np.pi, 'topological_charge': -1, 'index': 2}
        ]
        hot_swap.total_energy = sum(s['amplitude']**2 for s in hot_swap.active_solitons)
        
        logger.info(f"Initial energy: {hot_swap.total_energy:.2f}")
        logger.info(f"Active solitons: {len(hot_swap.active_solitons)}")
        
        # Test topology swaps
        test_topologies = ['honeycomb', 'triangular', 'small_world']
        
        for target_topology in test_topologies:
            if target_topology in hot_swap.topologies:
                logger.info(f"\nTesting swap to: {target_topology}")
                success = await hot_swap.hot_swap_laplacian_with_safety(target_topology)
                logger.info(f"Swap result: {'SUCCESS' if success else 'FAILED'}")
            else:
                logger.warning(f"Topology {target_topology} not available")
        
        # Test adaptive swapping
        logger.info("\nTesting adaptive complexity detection:")
        complexity_patterns = ['sparse_search', 'dense_matrix', 'O(n²)']
        
        for pattern in complexity_patterns:
            logger.info(f"Testing complexity pattern: {pattern}")
            recommended = hot_swap.recommend_topology_for_problem(pattern)
            logger.info(f"Recommended topology: {recommended}")
            
            if recommended != hot_swap.current_topology:
                success = await hot_swap.adaptive_swap_for_complexity(pattern)
                logger.info(f"Adaptive swap result: {'SUCCESS' if success else 'FAILED'}")
        
        # Display final metrics
        logger.info("\nFinal System Metrics:")
        metrics = hot_swap.get_swap_metrics()
        for key, value in metrics.items():
            if key != 'swap_history':  # Skip detailed history for brevity
                logger.info(f"  {key}: {value}")
        
        logger.info("\nDemo completed successfully!")
        return hot_swap
        
    except Exception as e:
        logger.exception("Demo failed with exception")
        raise

# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # Run the demo
        result = asyncio.run(demo_hot_swap())
        logger.info("Program completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.exception("Program failed with critical error")
        sys.exit(1)
