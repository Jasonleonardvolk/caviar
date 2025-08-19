// Dark Soliton Support - Add SolitonMode enum to soliton_memory.rs
// This patch adds dark soliton encoding capabilities

use serde::{Deserialize, Serialize};

// Add to soliton_memory.rs after other enums
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SolitonMode {
    Bright,  // Standard positive amplitude soliton (peak)
    Dark,    // Phase-inverted soliton (dip on continuous background)
}

impl Default for SolitonMode {
    fn default() -> Self {
        SolitonMode::Bright
    }
}

// Add to SolitonMemory struct
pub struct SolitonMemory {
    // ... existing fields ...
    pub mode: SolitonMode,  // NEW: Type of soliton (bright or dark)
    pub baseline_amplitude: Option<f64>,  // NEW: For dark solitons, the background level
}

// Modify SolitonMemory::new to include mode
impl SolitonMemory {
    pub fn new_with_mode(
        concept_id: String,
        content: String,
        importance: f64,
        mode: SolitonMode,
    ) -> Self {
        let mut memory = Self::new(concept_id, content, importance);
        memory.mode = mode;
        
        // For dark solitons, set baseline amplitude
        if mode == SolitonMode::Dark {
            memory.baseline_amplitude = Some(1.0);  // Normalized background
        }
        
        memory
    }
    
    // Add method to evaluate dark soliton waveform
    pub fn evaluate_dark_soliton_waveform(&self, t: f64) -> Complex64 {
        let t0 = self.position;
        let width = self.width;
        let omega = self.frequency;
        let phase = self.phase_tag;
        
        match self.mode {
            SolitonMode::Bright => {
                // Existing bright soliton formula
                self.evaluate_waveform(t)
            }
            SolitonMode::Dark => {
                // Dark soliton: dip on continuous background
                let baseline = self.baseline_amplitude.unwrap_or(1.0);
                let depth = self.amplitude;  // How deep the dip is
                
                // Dark soliton profile: A₀[1 - d²sech²((t-t₀)/T)]^(1/2) * exp(iφ)
                let sech_term = 1.0 / ((t - t0) / width).cosh();
                let amplitude = baseline * (1.0 - depth * depth * sech_term * sech_term).sqrt();
                
                // Phase jump at center (π phase shift)
                let phase_profile = if (t - t0).abs() < width {
                    phase + std::f64::consts::PI * ((t - t0) / width).tanh()
                } else {
                    phase
                };
                
                Complex64::from_polar(amplitude, omega * t + phase_profile)
            }
        }
    }
    
    // Modify correlate_with_signal to handle dark solitons
    pub fn correlate_with_signal(&self, target_phase: f64, tolerance: f64) -> f64 {
        let phase_diff = (self.phase_tag - target_phase).abs();
        let normalized_diff = phase_diff.min(2.0 * PI - phase_diff) / PI;
        
        if normalized_diff <= tolerance {
            let correlation = (1.0 - normalized_diff / tolerance) * self.amplitude;
            
            // Dark solitons anti-correlate (negative match)
            match self.mode {
                SolitonMode::Bright => correlation,
                SolitonMode::Dark => -correlation,  // Negative for suppression
            }
        } else {
            0.0
        }
    }
}

// Add to SolitonLattice for dark soliton recall filtering
impl SolitonLattice {
    pub fn recall_by_phase_with_dark_suppression(
        &self,
        target_phase: f64,
        tolerance: f64,
    ) -> Vec<&SolitonMemory> {
        let mut results = Vec::new();
        let mut dark_phases = Vec::new();
        
        // First pass: identify dark soliton phases
        for memory in self.memories.values() {
            if memory.mode == SolitonMode::Dark 
                && memory.correlate_with_signal(target_phase, tolerance).abs() > 0.0 {
                dark_phases.push(memory.phase_tag);
            }
        }
        
        // Second pass: collect bright solitons not suppressed by dark ones
        for memory in self.memories.values() {
            if memory.mode == SolitonMode::Bright {
                let correlation = memory.correlate_with_signal(target_phase, tolerance);
                if correlation > 0.0 {
                    // Check if this bright soliton is suppressed by any dark soliton
                    let mut suppressed = false;
                    for &dark_phase in &dark_phases {
                        if (memory.phase_tag - dark_phase).abs() < tolerance {
                            suppressed = true;
                            break;
                        }
                    }
                    
                    if !suppressed {
                        results.push(memory);
                    }
                }
            }
        }
        
        results
    }
    
    // Store a dark soliton memory
    pub fn store_dark_memory(
        &mut self,
        concept_id: String,
        content: String,
        importance: f64,
        reason: &str,
    ) -> String {
        info!("Storing dark soliton memory for concept '{}': {}", concept_id, reason);
        
        let memory = SolitonMemory::new_with_mode(
            concept_id.clone(),
            content,
            importance,
            SolitonMode::Dark,
        );
        
        let memory_id = memory.id.clone();
        self.memories.insert(memory_id.clone(), memory);
        
        // Update concept registry to note suppression
        if let Some(phase) = self.concept_phase_map.get(&concept_id) {
            self.phase_registry.insert(
                *phase,
                format!("SUPPRESSED:{}", concept_id),
            );
        }
        
        memory_id
    }
}

// Vault status extension for dark memories
impl SolitonMemory {
    pub fn apply_dark_suppression(&mut self) {
        // Convert a bright memory to effectively dark by vaulting with special status
        self.vault_status = VaultStatus::Quarantine;
        self.mode = SolitonMode::Dark;
        info!("Memory {} converted to dark soliton for suppression", self.id);
    }
}
