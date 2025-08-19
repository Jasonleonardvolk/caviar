// Fix for soliton_memory.rs - Complete comfort metrics calculation
// Add this to the SolitonMemory implementation

impl SolitonMemory {
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
        // Calculate net coupling forces on this memory
        let mut total_flux = 0.0;
        
        if let Some(osc_idx) = self.metadata.get("oscillator_idx") {
            let my_id = self.id.clone();
            
            // Sum coupling forces from all connected oscillators
            for (other_id, other_memory) in &lattice.memories {
                if other_id != &my_id {
                    // Get coupling strength between these two memories
                    let key = (my_id.clone(), other_id.clone());
                    if let Some(&coupling) = lattice.coupling_matrix.get(&key) {
                        // Force is proportional to coupling * phase difference
                        let phase_diff = (other_memory.phase_tag - self.phase_tag).abs();
                        let force = coupling * phase_diff.sin();
                        total_flux += force;
                    }
                }
            }
        }
        
        // Normalize flux to [0, 1] range
        total_flux.abs().min(1.0)
    }
    
    fn calculate_perturbation(&self, lattice: &SolitonLattice) -> f64 {
        // Perturbation based on topology morphing state
        if lattice.target_topology.is_some() {
            // Currently morphing - high perturbation
            0.5 + 0.5 * lattice.blend_progress.unwrap_or(0.0)
        } else {
            // Decay perturbation over time if no morphing
            self.comfort_metrics.perturbation * 0.95
        }
    }
}
