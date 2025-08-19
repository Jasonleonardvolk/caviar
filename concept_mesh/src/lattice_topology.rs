// concept-mesh/src/lattice_topology.rs

use crate::soliton_memory::{SolitonLattice, SolitonMemory};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use lazy_static::lazy_static;

// Pre-allocated reusable structures to avoid allocation storms
lazy_static! {
    static ref BLEND_WORKSPACE: Mutex<BlendWorkspace> = Mutex::new(BlendWorkspace::new());
}

struct BlendWorkspace {
    blended_map: HashMap<(usize, usize), f64>,
    edge_set: HashSet<(usize, usize)>,
    matrix_buffer: Vec<Vec<f64>>,
}

impl BlendWorkspace {
    fn new() -> Self {
        Self {
            blended_map: HashMap::with_capacity(100000), // Pre-size for large lattices
            edge_set: HashSet::with_capacity(100000),
            matrix_buffer: Vec::new(),
        }
    }
    
    fn clear(&mut self) {
        self.blended_map.clear();
        self.edge_set.clear();
        // Don't deallocate, just clear
    }
    
    fn ensure_matrix_buffer(&mut self, size: usize) {
        if self.matrix_buffer.len() < size {
            self.matrix_buffer.resize_with(size, || vec![0.0; size]);
        }
        // Clear existing values
        for row in &mut self.matrix_buffer[0..size] {
            for val in &mut row[0..size] {
                *val = 0.0;
            }
        }
    }
}

// Re-export 3D functionality
pub use crate::lattice_topology_3d::{find_corner_states_3d, generate_kagome_3d};

pub trait LatticeTopology: Send + Sync {
    /// Generate coupling matrix for n nodes
    fn generate_coupling(&self, n: usize) -> HashMap<(usize, usize), f64>;

    /// Get topology name
    fn name(&self) -> &str;

    /// Place a soliton optimally in this topology
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice);

    /// Compute optimal capacity for this topology
    fn optimal_capacity(&self) -> usize;

    /// Clone as boxed trait object
    fn clone_box(&self) -> Box<dyn LatticeTopology>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct KagomeTopology {
    t1: f64, // Intra-triangle coupling
    t2: f64, // Inter-triangle coupling
}

impl KagomeTopology {
    pub fn new() -> Self {
        Self {
            t1: 1.0, // Strong intra-cell
            t2: 0.5, // Breathing ratio t1/t2 = 2.0 for flat band
        }
    }
}

impl LatticeTopology for KagomeTopology {
    fn generate_coupling(&self, n: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();

        // Group nodes into triangles (3 per unit cell)
        for i in (0..n).step_by(3) {
            if i + 2 < n {
                // Intra-triangle connections
                coupling.insert((i, i + 1), self.t1);
                coupling.insert((i + 1, i), self.t1);
                coupling.insert((i + 1, i + 2), self.t1);
                coupling.insert((i + 2, i + 1), self.t1);
                coupling.insert((i + 2, i), self.t1);
                coupling.insert((i, i + 2), self.t1);

                // Inter-triangle connections
                if i + 3 < n {
                    coupling.insert((i, i + 3), self.t2);
                    coupling.insert((i + 3, i), self.t2);
                }
            }
        }

        coupling
    }

    fn name(&self) -> &str {
        "Kagome"
    }

    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Place in flat-band mode for maximum stability
        soliton.stability = 0.95; // Very stable in Kagome flat band
    }

    fn optimal_capacity(&self) -> usize {
        3000 // Can support ~3k stable solitons
    }

    fn clone_box(&self) -> Box<dyn LatticeTopology> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HexagonalTopology {
    coupling_strength: f64,
}

impl HexagonalTopology {
    pub fn new() -> Self {
        Self {
            coupling_strength: 0.7,
        }
    }
}

impl LatticeTopology for HexagonalTopology {
    fn generate_coupling(&self, n: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();

        // Honeycomb lattice - each node connects to 3 neighbors
        let rows = (n as f64).sqrt() as usize;
        let cols = n / rows;

        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;

                // Connect to right neighbor
                if j + 1 < cols {
                    let right = i * cols + (j + 1);
                    coupling.insert((idx, right), self.coupling_strength);
                    coupling.insert((right, idx), self.coupling_strength);
                }

                // Connect to bottom-right (zigzag)
                if i + 1 < rows && (i + j) % 2 == 0 {
                    let br = (i + 1) * cols + j;
                    coupling.insert((idx, br), self.coupling_strength);
                    coupling.insert((br, idx), self.coupling_strength);
                }

                // Connect to bottom-left (zigzag)
                if i + 1 < rows && j > 0 && (i + j) % 2 == 1 {
                    let bl = (i + 1) * cols + (j - 1);
                    coupling.insert((idx, bl), self.coupling_strength);
                    coupling.insert((bl, idx), self.coupling_strength);
                }
            }
        }

        coupling
    }

    fn name(&self) -> &str {
        "Hexagonal"
    }

    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Good for propagation
        soliton.stability = 0.8;
    }

    fn optimal_capacity(&self) -> usize {
        5000 // Higher capacity but less stable
    }

    fn clone_box(&self) -> Box<dyn LatticeTopology> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SquareTopology {
    coupling_strength: f64,
}

impl SquareTopology {
    pub fn new() -> Self {
        Self {
            coupling_strength: 0.6,
        }
    }
}

impl LatticeTopology for SquareTopology {
    fn generate_coupling(&self, n: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        let size = (n as f64).sqrt() as usize;

        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;

                // Right neighbor
                if j + 1 < size {
                    let right = i * size + (j + 1);
                    coupling.insert((idx, right), self.coupling_strength);
                    coupling.insert((right, idx), self.coupling_strength);
                }

                // Bottom neighbor
                if i + 1 < size {
                    let bottom = (i + 1) * size + j;
                    coupling.insert((idx, bottom), self.coupling_strength);
                    coupling.insert((bottom, idx), self.coupling_strength);
                }
            }
        }

        coupling
    }

    fn name(&self) -> &str {
        "Square"
    }

    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        soliton.stability = 0.75;
    }

    fn optimal_capacity(&self) -> usize {
        4000
    }

    fn clone_box(&self) -> Box<dyn LatticeTopology> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AllToAllTopology {
    coupling_strength: f64,
}

impl AllToAllTopology {
    pub fn new(strength: f64) -> Self {
        Self {
            coupling_strength: strength,
        }
    }
}

impl LatticeTopology for AllToAllTopology {
    fn generate_coupling(&self, n: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();

        // Connect every node to every other node
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    coupling.insert((i, j), self.coupling_strength);
                }
            }
        }

        coupling
    }

    fn name(&self) -> &str {
        "AllToAll"
    }

    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // High interaction, lower stability
        soliton.stability = 0.6;
    }

    fn optimal_capacity(&self) -> usize {
        1000 // Limited due to high interaction
    }

    fn clone_box(&self) -> Box<dyn LatticeTopology> {
        Box::new(self.clone())
    }
}

/// Smooth Laplacian blending between topologies - REUSABLE VERSION
pub fn blend_coupling_matrices(
    current: &HashMap<(usize, usize), f64>,
    target: &HashMap<(usize, usize), f64>,
    alpha: f64,
) -> HashMap<(usize, usize), f64> {
    // Use the pre-allocated workspace to avoid allocation storms
    let mut workspace = BLEND_WORKSPACE.lock().unwrap();
    workspace.clear();
    
    // Reuse the pre-allocated edge set
    workspace.edge_set.extend(current.keys());
    workspace.edge_set.extend(target.keys());

    // Blend each edge weight into the reusable map
    for edge in &workspace.edge_set {
        let current_weight = current.get(edge).unwrap_or(&0.0);
        let target_weight = target.get(edge).unwrap_or(&0.0);
        let blended_weight = (1.0 - alpha) * current_weight + alpha * target_weight;

        if blended_weight > 0.001 {
            // Threshold for sparsity
            workspace.blended_map.insert(*edge, blended_weight);
        }
    }
    
    // Clone the result (only allocation we can't avoid)
    let result = workspace.blended_map.clone();
    
    // Clear for next use but keep capacity
    workspace.clear();
    
    result
}

/// Fast batch morphing - do multiple steps at once to reduce FFI overhead
pub fn batch_morph_coupling_matrices(
    current: &HashMap<(usize, usize), f64>,
    target: &HashMap<(usize, usize), f64>,
    alpha_start: f64,
    alpha_end: f64,
    steps: usize,
) -> Vec<HashMap<(usize, usize), f64>> {
    let mut results = Vec::with_capacity(steps);
    let alpha_step = (alpha_end - alpha_start) / (steps as f64 - 1.0);
    
    for i in 0..steps {
        let alpha = alpha_start + (i as f64) * alpha_step;
        results.push(blend_coupling_matrices(current, target, alpha));
    }
    
    results
}

// Re-export for compatibility
pub use self::generate_kagome_3d as generate_kagome_lattice;
