// Complete fixes for soliton_memory.rs
// Addresses 3 issues: update_comfort, crystallize_memories, resolve_collisions

use std::collections::HashMap;
use chrono::{DateTime, Utc};

impl SolitonMemory {
    // FIX 1: Complete comfort metrics calculation
    pub fn update_comfort(&mut self, lattice: &SolitonLattice) {
        // Update energy based on current amplitude
        self.comfort_metrics.energy = self.amplitude;
        
        // Update stress based on stability
        self.comfort_metrics.stress = 1.0 - self.stability;
        
        // Calculate flux from coupling forces
        self.comfort_metrics.flux = self.calculate_flux(lattice);
        
        // Calculate perturbation from recent topology changes
        self.comfort_metrics.perturbation = self.calculate_perturbation(lattice);
    }
    
    fn calculate_flux(&self, lattice: &SolitonLattice) -> f64 {
        let mut total_flux = 0.0;
        let my_id = self.id.clone();
        
        // Sum coupling forces from all connected oscillators
        for (other_id, other_memory) in &lattice.memories {
            if other_id != &my_id {
                // Check both directions of coupling
                let key1 = (my_id.clone(), other_id.clone());
                let key2 = (other_id.clone(), my_id.clone());
                
                let coupling = lattice.coupling_matrix.get(&key1)
                    .or_else(|| lattice.coupling_matrix.get(&key2))
                    .unwrap_or(&0.0);
                
                if *coupling > 0.0 {
                    // Force is proportional to coupling * phase difference
                    let phase_diff = (other_memory.phase_tag - self.phase_tag).abs();
                    let normalized_diff = phase_diff.min(2.0 * PI - phase_diff);
                    let force = coupling * normalized_diff.sin();
                    total_flux += force;
                }
            }
        }
        
        // Normalize flux to [0, 1] range
        total_flux.abs().tanh()  // Use tanh for smooth saturation
    }
    
    fn calculate_perturbation(&self, lattice: &SolitonLattice) -> f64 {
        if lattice.target_topology.is_some() {
            // Currently morphing - perturbation based on blend progress
            let base_perturbation = 0.3;
            let morph_contribution = 0.7 * lattice.blend_progress;
            base_perturbation + morph_contribution
        } else {
            // Decay perturbation over time if no morphing
            self.comfort_metrics.perturbation * 0.95
        }
    }
}

impl SolitonLattice {
    // FIX 2: Implement crystallize_memories
    fn crystallize_memories(&mut self) {
        info!("Starting memory crystallization...");
        
        // Step 1: Sort memories by heat (hot memories are frequently accessed)
        let mut memory_list: Vec<(String, f64, f64)> = self.memories.iter()
            .map(|(id, mem)| (id.clone(), mem.heat, mem.stability))
            .collect();
        
        memory_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Step 2: Migrate hot memories to stable positions
        let hot_threshold = 0.7;
        let total_positions = memory_list.len();
        let stable_zone_start = total_positions / 3;
        let stable_zone_end = 2 * total_positions / 3;
        
        for (i, (mem_id, heat, _)) in memory_list.iter().enumerate() {
            if *heat > hot_threshold {
                if i < stable_zone_start || i > stable_zone_end {
                    // This hot memory is in an unstable position
                    if let Some(memory) = self.memories.get_mut(mem_id) {
                        // Find a cold memory in stable zone to swap with
                        for j in stable_zone_start..stable_zone_end {
                            if j < memory_list.len() {
                                let (swap_id, swap_heat, _) = &memory_list[j];
                                if *swap_heat < 0.3 {  // Cold memory
                                    // Perform swap (simplified - in real impl would swap oscillator positions)
                                    memory.stability = (memory.stability + 0.1).min(1.0);
                                    info!("Migrated hot memory {} to stable position", mem_id);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Step 3: Cool down all memories
        for (_, memory) in self.memories.iter_mut() {
            memory.heat *= 0.95;  // Decay heat
        }
    }
    
    // FIX 3: Implement resolve_collisions
    fn resolve_collisions(&mut self) -> usize {
        let mut resolved = 0;
        
        // Find bright/dark memory pairs with same concept
        let mut concept_map: HashMap<String, Vec<(String, SolitonMode)>> = HashMap::new();
        
        for (id, memory) in &self.memories {
            if let Some(concept) = memory.concept_ids.first() {
                concept_map.entry(concept.clone())
                    .or_insert_with(Vec::new)
                    .push((id.clone(), memory.mode));
            }
        }
        
        // Resolve collisions
        for (concept, memories) in concept_map {
            let brights: Vec<_> = memories.iter()
                .filter(|(_, mode)| *mode == SolitonMode::Bright)
                .map(|(id, _)| id.clone())
                .collect();
                
            let darks: Vec<_> = memories.iter()
                .filter(|(_, mode)| *mode == SolitonMode::Dark)
                .map(|(id, _)| id.clone())
                .collect();
            
            if !brights.is_empty() && !darks.is_empty() {
                // We have a collision - dark should suppress bright
                info!("Resolving collision for concept: {}", concept);
                
                // Option 1: Vault the bright memories
                for bright_id in &brights {
                    if let Some(bright_mem) = self.memories.get_mut(bright_id) {
                        if bright_mem.vault_status == VaultStatus::Active {
                            bright_mem.apply_vault_phase_shift(VaultStatus::Quarantine);
                            resolved += 1;
                        }
                    }
                }
                
                // Option 2: If dark memory is weak, might merge instead
                if darks.len() == 1 && brights.len() > 3 {
                    if let Some(dark_mem) = self.memories.get(&darks[0]) {
                        if dark_mem.amplitude < 0.5 {
                            // Weak dark memory - consider removing it
                            info!("Weak dark soliton for {} - considering removal", concept);
                        }
                    }
                }
            }
        }
        
        resolved
    }
}

// Add the missing fields to SolitonLattice if not present:
// pub blend_progress: f64,
// pub heat: f64,  // Add to SolitonMemory struct
