// Fix for lattice_topology.rs - Add SmallWorldTopology and complete blending

use std::collections::HashMap;

// Small-World Topology Implementation
pub struct SmallWorldTopology {
    base_coupling: f64,
    random_links: usize,
    rewire_probability: f64,
}

impl SmallWorldTopology {
    pub fn new(base_coupling: f64) -> Self {
        Self {
            base_coupling,
            random_links: 4,  // Each node gets 4 random long-range connections
            rewire_probability: 0.1,  // 10% chance to rewire each edge
        }
    }
}

impl LatticeTopology for SmallWorldTopology {
    fn name(&self) -> &str {
        "small_world"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        
        // Start with ring topology (each node connected to k nearest neighbors)
        let k = 4; // Connect to 2 neighbors on each side
        for i in 0..size {
            for j in 1..=k/2 {
                let neighbor1 = (i + j) % size;
                let neighbor2 = (i + size - j) % size;
                
                coupling.insert((i, neighbor1), self.base_coupling);
                coupling.insert((neighbor1, i), self.base_coupling);
                coupling.insert((i, neighbor2), self.base_coupling);
                coupling.insert((neighbor2, i), self.base_coupling);
            }
        }
        
        // Rewire some edges to create shortcuts (Watts-Strogatz model)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let edges: Vec<_> = coupling.keys().cloned().collect();
        for (i, j) in edges {
            if i < j && rng.gen::<f64>() < self.rewire_probability {
                // Remove original edge
                coupling.remove(&(i, j));
                coupling.remove(&(j, i));
                
                // Add random edge
                let new_target = rng.gen_range(0..size);
                if new_target != i && !coupling.contains_key(&(i, new_target)) {
                    coupling.insert((i, new_target), self.base_coupling);
                    coupling.insert((new_target, i), self.base_coupling);
                }
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, _lattice: &SolitonLattice) {
        // Small-world placement: slightly lower initial stability due to long-range connections
        soliton.stability = 0.75;
        // Position randomly to leverage small-world properties
        soliton.position = rand::random::<f64>() * 100.0;
    }
}

// Fix for SolitonLattice - Add blend progress tracking and step_topology_blend
impl SolitonLattice {
    pub fn morph_topology(&mut self, new_topology: Box<dyn LatticeTopology>, blend_rate: f64) {
        self.target_topology = Some(new_topology);
        self.blend_rate = blend_rate;
        self.blend_progress = 0.0;  // Add this field to SolitonLattice struct
        info!("Initiating topology morph to {} with blend rate {}", 
              self.target_topology.as_ref().unwrap().name(), blend_rate);
    }
    
    pub fn step_topology_blend(&mut self) {
        if let Some(ref target) = self.target_topology {
            // Increment blend progress
            self.blend_progress += self.blend_rate;
            
            if self.blend_progress >= 1.0 {
                // Complete the transition
                self.blend_progress = 1.0;
                
                // Generate new coupling matrix from target topology
                let new_coupling = target.generate_coupling(self.memories.len());
                
                // Replace coupling matrix
                self.coupling_matrix.clear();
                for ((i, j), weight) in new_coupling {
                    let key = (format!("node_{}", i), format!("node_{}", j));
                    self.coupling_matrix.insert(key, weight);
                }
                
                // Update current topology
                self.current_topology = self.target_topology.take().unwrap();
                self.blend_progress = 0.0;
                
                info!("Topology morph complete to {}", self.current_topology.name());
            } else {
                // Gradual interpolation of coupling matrices
                let alpha = self.blend_progress;
                
                // Store current coupling temporarily
                let old_coupling = self.coupling_matrix.clone();
                
                // Generate target coupling
                let target_coupling = target.generate_coupling(self.memories.len());
                let mut new_target_map = HashMap::new();
                for ((i, j), weight) in target_coupling {
                    let key = (format!("node_{}", i), format!("node_{}", j));
                    new_target_map.insert(key, weight);
                }
                
                // Interpolate between old and new
                for (key, old_weight) in &old_coupling {
                    let new_weight = new_target_map.get(key).unwrap_or(&0.0);
                    let interpolated = old_weight * (1.0 - alpha) + new_weight * alpha;
                    self.coupling_matrix.insert(key.clone(), interpolated);
                }
                
                // Add any new edges that weren't in old coupling
                for (key, new_weight) in &new_target_map {
                    if !old_coupling.contains_key(key) {
                        self.coupling_matrix.insert(key.clone(), new_weight * alpha);
                    }
                }
                
                debug!("Topology blend progress: {:.1}%", self.blend_progress * 100.0);
            }
        }
    }
}

// Add blend_progress field to SolitonLattice struct definition:
// pub blend_progress: f64,
