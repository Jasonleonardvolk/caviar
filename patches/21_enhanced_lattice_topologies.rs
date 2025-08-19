// Enhanced Lattice Topology Implementation
// Adds concrete implementations for Kagome, Hexagonal, and Square lattices

use std::collections::HashMap;
use crate::soliton_memory::{SolitonMemory, SolitonLattice};

// Add to lattice_topology.rs

#[derive(Debug, Clone)]
pub struct KagomeTopology {
    pub intra_cell_coupling: f64,  // t1: strong coupling within triangles
    pub inter_cell_coupling: f64,  // t2: weak coupling between triangles
    pub breathing_ratio: f64,      // t1/t2, ideally 2.0 for flat band
}

impl KagomeTopology {
    pub fn new() -> Self {
        Self {
            intra_cell_coupling: 1.0,
            inter_cell_coupling: 0.5,
            breathing_ratio: 2.0,
        }
    }
    
    pub fn with_parameters(t1: f64, t2: f64) -> Self {
        Self {
            intra_cell_coupling: t1,
            inter_cell_coupling: t2,
            breathing_ratio: t1 / t2,
        }
    }
}

impl LatticeTopology for KagomeTopology {
    fn name(&self) -> &str {
        "kagome"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        
        // Kagome lattice: arrange nodes in triangles (3 nodes per unit cell)
        let triangles = size / 3;
        
        for t in 0..triangles {
            let base = t * 3;
            
            // Intra-triangle coupling (strong)
            for i in 0..3 {
                for j in i+1..3 {
                    coupling.insert((base + i, base + j), self.intra_cell_coupling);
                    coupling.insert((base + j, base + i), self.intra_cell_coupling);
                }
            }
            
            // Inter-triangle coupling (weak) - connect to next triangle
            if t + 1 < triangles {
                let next_base = (t + 1) * 3;
                coupling.insert((base + 2, next_base), self.inter_cell_coupling);
                coupling.insert((next_base, base + 2), self.inter_cell_coupling);
            }
        }
        
        // Handle remaining nodes if size is not divisible by 3
        for i in triangles * 3..size {
            if i > 0 {
                coupling.insert((i, i - 1), self.inter_cell_coupling);
                coupling.insert((i - 1, i), self.inter_cell_coupling);
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Place soliton in a flat-band localized state
        // In Kagome, these are the compact localized states (CLS)
        
        let num_memories = lattice.memories.len();
        let triangle_idx = num_memories / 3;
        let position_in_triangle = num_memories % 3;
        
        // Position soliton based on its role in the triangle
        soliton.position = (triangle_idx as f64) * 10.0 + (position_in_triangle as f64) * 3.0;
        
        // High stability in Kagome flat band
        soliton.stability = 0.95;
        
        // Adjust width for localization
        soliton.width = 1.0 / self.breathing_ratio;
    }
}

#[derive(Debug, Clone)]
pub struct HexagonalTopology {
    pub coupling_strength: f64,
    pub coordination_number: usize,  // 3 for honeycomb
}

impl HexagonalTopology {
    pub fn new() -> Self {
        Self {
            coupling_strength: 0.5,
            coordination_number: 3,
        }
    }
}

impl LatticeTopology for HexagonalTopology {
    fn name(&self) -> &str {
        "hexagonal"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        
        // Hexagonal/honeycomb lattice: each node connects to 3 neighbors
        // Simplified 1D embedding of hex structure
        for i in 0..size {
            // Connect to neighbors in a pattern that mimics hex connectivity
            let neighbors = match i % 6 {
                0 => vec![1, 5, 6],
                1 => vec![0, 2, 7],
                2 => vec![1, 3, 8],
                3 => vec![2, 4, 9],
                4 => vec![3, 5, 10],
                5 => vec![4, 0, 11],
                _ => vec![],
            };
            
            for &j in &neighbors {
                if j < size {
                    coupling.insert((i, j), self.coupling_strength);
                }
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Hexagonal placement - good for propagation
        soliton.stability = 0.8;
        soliton.position = (lattice.memories.len() as f64) * 5.0;
    }
}

#[derive(Debug, Clone)]
pub struct SquareTopology {
    pub coupling_strength: f64,
    pub grid_width: usize,
}

impl SquareTopology {
    pub fn new() -> Self {
        Self {
            coupling_strength: 0.3,
            grid_width: 10,  // Default 10x10 grid
        }
    }
}

impl LatticeTopology for SquareTopology {
    fn name(&self) -> &str {
        "square"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        
        // Square lattice: each node connects to 4 neighbors (N,S,E,W)
        let width = self.grid_width;
        
        for i in 0..size {
            let row = i / width;
            let col = i % width;
            
            // North
            if row > 0 {
                let j = (row - 1) * width + col;
                coupling.insert((i, j), self.coupling_strength);
            }
            
            // South
            if row + 1 < size / width {
                let j = (row + 1) * width + col;
                if j < size {
                    coupling.insert((i, j), self.coupling_strength);
                }
            }
            
            // East
            if col + 1 < width {
                let j = row * width + (col + 1);
                if j < size {
                    coupling.insert((i, j), self.coupling_strength);
                }
            }
            
            // West
            if col > 0 {
                let j = row * width + (col - 1);
                coupling.insert((i, j), self.coupling_strength);
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Square lattice placement
        let idx = lattice.memories.len();
        let row = idx / self.grid_width;
        let col = idx % self.grid_width;
        
        soliton.position = (row as f64) * 10.0 + (col as f64);
        soliton.stability = 0.7;  // Moderate stability
    }
}

// Small-world topology (bonus implementation)
#[derive(Debug, Clone)]
pub struct SmallWorldTopology {
    pub base_coupling: f64,
    pub rewiring_probability: f64,
    pub num_neighbors: usize,
}

impl SmallWorldTopology {
    pub fn new() -> Self {
        Self {
            base_coupling: 0.05,
            rewiring_probability: 0.1,
            num_neighbors: 4,
        }
    }
}

impl LatticeTopology for SmallWorldTopology {
    fn name(&self) -> &str {
        "small_world"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        use rand::Rng;
        let mut coupling = HashMap::new();
        let mut rng = rand::thread_rng();
        
        // Start with ring lattice
        for i in 0..size {
            for k in 1..=self.num_neighbors/2 {
                let j = (i + k) % size;
                coupling.insert((i, j), self.base_coupling);
                coupling.insert((j, i), self.base_coupling);
            }
        }
        
        // Rewire with probability p
        let edges: Vec<(usize, usize)> = coupling.keys().cloned().collect();
        for (i, j) in edges {
            if i < j && rng.gen::<f64>() < self.rewiring_probability {
                // Remove edge
                coupling.remove(&(i, j));
                coupling.remove(&(j, i));
                
                // Add random edge
                let k = rng.gen_range(0..size);
                if k != i && k != j {
                    coupling.insert((i, k), self.base_coupling);
                    coupling.insert((k, i), self.base_coupling);
                }
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Small-world: balance between local and global
        soliton.stability = 0.85;
        soliton.position = rand::random::<f64>() * 100.0;
    }
}

// Laplacian blending for smooth topology transitions
impl SolitonLattice {
    pub fn blend_topologies(
        &mut self,
        current: &dyn LatticeTopology,
        target: &dyn LatticeTopology,
        blend_factor: f64,  // 0.0 = current, 1.0 = target
    ) {
        let size = self.memories.len();
        
        // Generate coupling matrices for both topologies
        let current_coupling = current.generate_coupling(size);
        let target_coupling = target.generate_coupling(size);
        
        // Clear existing coupling
        self.coupling_matrix.clear();
        
        // Blend the coupling matrices
        let mut all_edges = std::collections::HashSet::new();
        all_edges.extend(current_coupling.keys());
        all_edges.extend(target_coupling.keys());
        
        for edge in all_edges {
            let current_weight = current_coupling.get(edge).unwrap_or(&0.0);
            let target_weight = target_coupling.get(edge).unwrap_or(&0.0);
            
            // Linear interpolation
            let blended_weight = (1.0 - blend_factor) * current_weight + blend_factor * target_weight;
            
            if blended_weight > 0.001 {  // Threshold to keep matrix sparse
                // Convert indices to string IDs
                let (i, j) = edge;
                let key = (format!("mem_{}", i), format!("mem_{}", j));
                self.coupling_matrix.insert(key, blended_weight);
            }
        }
        
        info!("Blended topology from {} to {} (factor: {:.2})", 
              current.name(), target.name(), blend_factor);
    }
    
    // Smooth topology morphing over time
    pub async fn morph_topology(
        &mut self,
        target: Box<dyn LatticeTopology>,
        duration_steps: usize,
    ) {
        let current = self.current_topology.clone();
        
        info!("Starting topology morph from {} to {} over {} steps", 
              current.name(), target.name(), duration_steps);
        
        for step in 0..=duration_steps {
            let blend_factor = (step as f64) / (duration_steps as f64);
            self.blend_topologies(current.as_ref(), target.as_ref(), blend_factor);
            
            // Small delay to allow system to adapt
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        
        self.current_topology = target;
        info!("Topology morph complete");
    }
}
