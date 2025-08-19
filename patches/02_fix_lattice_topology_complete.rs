// Complete fixes for lattice_topology.rs
// Addresses 2 issues: SmallWorldTopology implementation and step_topology_blend completion

use std::collections::HashMap;
use rand::Rng;
use crate::soliton_memory::{SolitonMemory, SolitonLattice};

// Add to SolitonLattice struct fields:
// blend_cache: Option<HashMap<(String, String), f64>>,  // Cached blended coupling matrix
// blend_cache_progress: f64,  // Progress value when cache was created

// FIX 1: Complete SmallWorldTopology implementation
pub struct SmallWorldTopology {
    base_coupling: f64,
    num_neighbors: usize,  // k in Watts-Strogatz model
    rewire_probability: f64,  // p in Watts-Strogatz model
}

impl SmallWorldTopology {
    pub fn new(base_coupling: f64) -> Self {
        Self {
            base_coupling,
            num_neighbors: 6,  // Each node initially connected to 6 neighbors
            rewire_probability: 0.1,  // 10% chance to rewire
        }
    }
    
    pub fn with_parameters(base_coupling: f64, num_neighbors: usize, rewire_prob: f64) -> Self {
        Self {
            base_coupling,
            num_neighbors,
            rewire_probability: rewire_prob.clamp(0.0, 1.0),
        }
    }
}

impl LatticeTopology for SmallWorldTopology {
    fn name(&self) -> &str {
        "small_world"
    }
    
    fn generate_coupling(&self, size: usize) -> HashMap<(usize, usize), f64> {
        let mut coupling = HashMap::new();
        let mut rng = rand::thread_rng();
        
        // Step 1: Create regular ring lattice
        // Each node connected to k/2 neighbors on each side
        let half_neighbors = self.num_neighbors / 2;
        
        for i in 0..size {
            for j in 1..=half_neighbors {
                let neighbor_right = (i + j) % size;
                let neighbor_left = (i + size - j) % size;
                
                // Add bidirectional connections
                coupling.insert((i, neighbor_right), self.base_coupling);
                coupling.insert((neighbor_right, i), self.base_coupling);
                coupling.insert((i, neighbor_left), self.base_coupling);
                coupling.insert((neighbor_left, i), self.base_coupling);
            }
        }
        
        // Step 2: Rewire edges with probability p (Watts-Strogatz)
        let edges: Vec<(usize, usize)> = coupling.keys()
            .filter(|(i, j)| i < j)  // Only consider each edge once
            .cloned()
            .collect();
        
        for (i, j) in edges {
            if rng.gen::<f64>() < self.rewire_probability {
                // Remove the original edge
                coupling.remove(&(i, j));
                coupling.remove(&(j, i));
                
                // Find a new target that's not already connected to i
                let mut attempts = 0;
                while attempts < 100 {  // Prevent infinite loop
                    let new_target = rng.gen_range(0..size);
                    
                    if new_target != i && 
                       new_target != j &&
                       !coupling.contains_key(&(i, new_target)) {
                        // Add the rewired edge
                        coupling.insert((i, new_target), self.base_coupling);
                        coupling.insert((new_target, i), self.base_coupling);
                        break;
                    }
                    attempts += 1;
                }
                
                // If we couldn't find a valid target, restore original edge
                if attempts >= 100 {
                    coupling.insert((i, j), self.base_coupling);
                    coupling.insert((j, i), self.base_coupling);
                }
            }
        }
        
        coupling
    }
    
    fn place_soliton(&self, soliton: &mut SolitonMemory, lattice: &SolitonLattice) {
        // Small-world placement strategy: 
        // Balance between local clustering and global accessibility
        soliton.stability = 0.8;  // Good baseline stability
        
        // Position based on existing network structure
        let num_memories = lattice.memories.len();
        if num_memories > 0 {
            // Find a position that balances local and global connections
            let position = (num_memories as f64).sqrt() * 10.0;
            soliton.position = position % 100.0;
        } else {
            soliton.position = rand::random::<f64>() * 100.0;
        }
    }
}

impl SolitonLattice {
    // FIX 2: Complete step_topology_blend implementation
    pub fn step_topology_blend(&mut self) {
        if let Some(ref target) = self.target_topology {
            // Increment blend progress
            self.blend_progress += self.blend_rate;
            
            if self.blend_progress >= 1.0 {
                // Complete the transition
                self.complete_topology_transition();
            } else {
                // Gradual interpolation
                self.interpolate_topology();
            }
            
            // Update all memory comfort metrics during transition
            let memory_ids: Vec<String> = self.memories.keys().cloned().collect();
            for id in memory_ids {
                if let Some(memory) = self.memories.get_mut(&id) {
                    memory.update_comfort(self);
                }
            }
        }
    }
    
    fn complete_topology_transition(&mut self) {
        if let Some(target) = self.target_topology.take() {
            info!("Completing topology transition to {}", target.name());
            
            self.blend_progress = 1.0;
            
            // Generate new coupling matrix
            let new_coupling = target.generate_coupling(self.memories.len());
            
            // Convert to string-based keys
            self.coupling_matrix.clear();
            for ((i, j), weight) in new_coupling {
                let key = (format!("node_{}", i), format!("node_{}", j));
                self.coupling_matrix.insert(key, weight);
            }
            
            // Update current topology
            self.current_topology = target;
            self.blend_progress = 0.0;
            
            // Reset all perturbation values
            for (_, memory) in self.memories.iter_mut() {
                memory.comfort_metrics.perturbation = 0.0;
            }
            
            info!("Topology transition complete");
        }
    }
    
    fn interpolate_topology(&mut self) {
        if let Some(ref target) = self.target_topology {
            let alpha = self.blend_progress;
            let beta = 1.0 - alpha;
            
            debug!("Topology blend progress: {:.1}%", alpha * 100.0);
            
            // Check if we can use cached blend
            let cache_tolerance = 0.01;  // Recompute if progress changed by more than 1%
            let needs_recompute = self.blend_cache.is_none() || 
                                  (self.blend_cache_progress - alpha).abs() > cache_tolerance;
            
            if needs_recompute {
                // Store current coupling for energy calculation
                let old_coupling = self.coupling_matrix.clone();
                
                // Generate target coupling (only when needed)
                let target_coupling_raw = target.generate_coupling(self.memories.len());
                let mut target_coupling = HashMap::new();
                
                for ((i, j), weight) in target_coupling_raw {
                    let key = (format!("node_{}", i), format!("node_{}", j));
                    target_coupling.insert(key, weight);
                }
                
                // Build interpolated matrix
                let mut interpolated_matrix = HashMap::new();
            
                // Add all keys from both old and new
                let mut all_keys = std::collections::HashSet::new();
                all_keys.extend(old_coupling.keys().cloned());
                all_keys.extend(target_coupling.keys().cloned());
                
                for key in all_keys {
                    let old_weight = old_coupling.get(&key).unwrap_or(&0.0);
                    let new_weight = target_coupling.get(&key).unwrap_or(&0.0);
                    
                    // Smooth interpolation with energy preservation
                    let interpolated = old_weight * beta + new_weight * alpha;
                    
                    // Only keep connections above threshold
                    if interpolated > 0.001 {
                        interpolated_matrix.insert(key, interpolated);
                    }
                }
                
                // Energy harvesting during transition
                let transition_energy = self.calculate_transition_energy(&old_coupling, &interpolated_matrix);
                self.total_charge += transition_energy * 0.8;  // 80% efficiency as configured
                
                // Cache the blended matrix
                self.blend_cache = Some(interpolated_matrix.clone());
                self.blend_cache_progress = alpha;
                
                // Apply the interpolated matrix
                self.coupling_matrix = interpolated_matrix;
                
                debug!("Harvested {} energy during transition (cache updated)", transition_energy * 0.8);
            } else {
                // Use cached blend
                if let Some(ref cached_matrix) = self.blend_cache {
                    self.coupling_matrix = cached_matrix.clone();
                    debug!("Using cached blend matrix");
                }
            }
        }
    }
    
    fn calculate_transition_energy(&self, old: &HashMap<(String, String), f64>, 
                                  new: &HashMap<(String, String), f64>) -> f64 {
        let mut energy = 0.0;
        
        // Energy is the sum of absolute differences in coupling
        for (key, old_val) in old {
            let new_val = new.get(key).unwrap_or(&0.0);
            energy += (old_val - new_val).abs();
        }
        
        // Also account for new connections
        for (key, new_val) in new {
            if !old.contains_key(key) {
                energy += new_val.abs();
            }
        }
        
        energy
    }
}
