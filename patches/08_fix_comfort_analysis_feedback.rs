// Fix for comfort_analysis.rs - Implement automated feedback path
use std::collections::HashMap;
use crate::soliton_memory::{SolitonMemory, ComfortVector, SolitonLattice};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortFeedback {
    pub timestamp: DateTime<Utc>,
    pub memory_id: String,
    pub comfort_state: ComfortVector,
    pub suggested_actions: Vec<ComfortAction>,
    pub severity: FeedbackSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComfortAction {
    ReduceCoupling { target_reduction: f64 },
    BoostAmplitude { target_boost: f64 },
    MigrateToStable { suggested_position: f64 },
    PauseMorphing,
    SwitchTopology { target: String },
    VaultMemory { reason: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FeedbackSeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ComfortAnalyzer {
    // Thresholds from config
    high_stress_threshold: f64,
    low_energy_threshold: f64,
    high_flux_threshold: f64,
    high_perturbation_threshold: f64,
    
    // Automated action flags
    auto_reduce_coupling: bool,
    auto_boost_amplitude: bool,
    auto_migrate: bool,
    auto_pause_morphing: bool,
    
    // History for trend analysis
    feedback_history: Vec<ComfortFeedback>,
    max_history_size: usize,
}

impl ComfortAnalyzer {
    pub fn new() -> Self {
        Self {
            high_stress_threshold: 0.8,
            low_energy_threshold: 0.2,
            high_flux_threshold: 0.9,
            high_perturbation_threshold: 0.8,
            auto_reduce_coupling: true,
            auto_boost_amplitude: true,
            auto_migrate: true,
            auto_pause_morphing: true,
            feedback_history: Vec::new(),
            max_history_size: 1000,
        }
    }
    
    pub fn with_config(config: &HashMap<String, f64>, auto_actions: &HashMap<String, bool>) -> Self {
        Self {
            high_stress_threshold: *config.get("high_stress_action_threshold").unwrap_or(&0.8),
            low_energy_threshold: *config.get("low_energy_action_threshold").unwrap_or(&0.2),
            high_flux_threshold: *config.get("high_flux_action_threshold").unwrap_or(&0.9),
            high_perturbation_threshold: *config.get("high_perturbation_action_threshold").unwrap_or(&0.8),
            auto_reduce_coupling: *auto_actions.get("reduce_coupling_on_high_stress").unwrap_or(&true),
            auto_boost_amplitude: *auto_actions.get("boost_amplitude_on_low_energy").unwrap_or(&true),
            auto_migrate: *auto_actions.get("migrate_on_high_flux").unwrap_or(&true),
            auto_pause_morphing: *auto_actions.get("pause_morphing_on_high_perturbation").unwrap_or(&true),
            feedback_history: Vec::new(),
            max_history_size: 1000,
        }
    }
    
    pub fn analyze_memory(&mut self, memory: &SolitonMemory, lattice: &SolitonLattice) -> Option<ComfortFeedback> {
        let comfort = &memory.comfort_metrics;
        let mut actions = Vec::new();
        let mut max_severity = FeedbackSeverity::Low;
        
        // Analyze stress levels
        if comfort.stress > self.high_stress_threshold {
            max_severity = FeedbackSeverity::High;
            
            if self.auto_reduce_coupling {
                actions.push(ComfortAction::ReduceCoupling {
                    target_reduction: 0.2, // Reduce couplings by 20%
                });
            }
            
            // If critically stressed, consider vaulting
            if comfort.stress > 0.95 {
                max_severity = FeedbackSeverity::Critical;
                actions.push(ComfortAction::VaultMemory {
                    reason: "Critical stress level".to_string(),
                });
            }
        }
        
        // Analyze energy levels
        if comfort.energy < self.low_energy_threshold {
            if max_severity as u8 < FeedbackSeverity::Medium as u8 {
                max_severity = FeedbackSeverity::Medium;
            }
            
            if self.auto_boost_amplitude {
                actions.push(ComfortAction::BoostAmplitude {
                    target_boost: 1.5, // 50% boost
                });
            }
        }
        
        // Analyze flux (coupling forces)
        if comfort.flux > self.high_flux_threshold {
            if max_severity as u8 < FeedbackSeverity::High as u8 {
                max_severity = FeedbackSeverity::High;
            }
            
            if self.auto_migrate {
                // Suggest migration to a more stable position
                let suggested_pos = self.calculate_stable_position(memory, lattice);
                actions.push(ComfortAction::MigrateToStable {
                    suggested_position: suggested_pos,
                });
            }
        }
        
        // Analyze perturbation (topology changes)
        if comfort.perturbation > self.high_perturbation_threshold {
            if max_severity as u8 < FeedbackSeverity::Medium as u8 {
                max_severity = FeedbackSeverity::Medium;
            }
            
            if self.auto_pause_morphing && lattice.target_topology.is_some() {
                actions.push(ComfortAction::PauseMorphing);
            }
        }
        
        // Check for combined critical conditions
        if comfort.stress > 0.7 && comfort.flux > 0.7 && comfort.perturbation > 0.5 {
            max_severity = FeedbackSeverity::Critical;
            actions.push(ComfortAction::SwitchTopology {
                target: "kagome".to_string(), // Most stable
            });
        }
        
        // Only create feedback if there are actions to take
        if !actions.is_empty() {
            let feedback = ComfortFeedback {
                timestamp: Utc::now(),
                memory_id: memory.id.clone(),
                comfort_state: comfort.clone(),
                suggested_actions: actions,
                severity: max_severity,
            };
            
            // Add to history
            self.feedback_history.push(feedback.clone());
            if self.feedback_history.len() > self.max_history_size {
                self.feedback_history.remove(0);
            }
            
            Some(feedback)
        } else {
            None
        }
    }
    
    fn calculate_stable_position(&self, memory: &SolitonMemory, lattice: &SolitonLattice) -> f64 {
        // Simple heuristic: move toward center of lattice where it's more stable
        let current_pos = memory.position;
        let center = 50.0; // Assuming 0-100 range
        
        // Move 20% closer to center
        current_pos + (center - current_pos) * 0.2
    }
    
    pub fn apply_feedback(&self, feedback: &ComfortFeedback, 
                         memory: &mut SolitonMemory, 
                         lattice: &mut SolitonLattice) -> Vec<String> {
        let mut applied_actions = Vec::new();
        
        for action in &feedback.suggested_actions {
            match action {
                ComfortAction::ReduceCoupling { target_reduction } => {
                    // Reduce all couplings involving this memory
                    let memory_id = &memory.id;
                    let keys_to_modify: Vec<_> = lattice.coupling_matrix.keys()
                        .filter(|(from, to)| from == memory_id || to == memory_id)
                        .cloned()
                        .collect();
                    
                    for key in keys_to_modify {
                        if let Some(coupling) = lattice.coupling_matrix.get_mut(&key) {
                            *coupling *= (1.0 - target_reduction);
                        }
                    }
                    
                    applied_actions.push(format!("Reduced couplings by {:.0}%", target_reduction * 100.0));
                }
                
                ComfortAction::BoostAmplitude { target_boost } => {
                    memory.amplitude = (memory.amplitude * target_boost).min(2.0);
                    applied_actions.push(format!("Boosted amplitude to {:.2}", memory.amplitude));
                }
                
                ComfortAction::MigrateToStable { suggested_position } => {
                    memory.position = *suggested_position;
                    memory.stability = (memory.stability + 0.1).min(1.0);
                    applied_actions.push(format!("Migrated to position {:.1}", suggested_position));
                }
                
                ComfortAction::PauseMorphing => {
                    // This would need to signal the morphing system
                    applied_actions.push("Morphing pause requested".to_string());
                }
                
                ComfortAction::SwitchTopology { target } => {
                    // This would need to signal the topology policy
                    applied_actions.push(format!("Topology switch to {} requested", target));
                }
                
                ComfortAction::VaultMemory { reason } => {
                    memory.apply_vault_phase_shift(crate::soliton_memory::VaultStatus::UserSealed);
                    applied_actions.push(format!("Memory vaulted: {}", reason));
                }
            }
        }
        
        // Update comfort metrics after actions
        memory.update_comfort(lattice);
        
        applied_actions
    }
    
    pub fn get_system_comfort_report(&self, lattice: &SolitonLattice) -> ComfortSystemReport {
        let mut total_stress = 0.0;
        let mut total_energy = 0.0;
        let mut total_flux = 0.0;
        let mut total_perturbation = 0.0;
        let mut count = 0;
        
        let mut stressed_memories = 0;
        let mut low_energy_memories = 0;
        let mut high_flux_memories = 0;
        
        for (_, memory) in &lattice.memories {
            let comfort = &memory.comfort_metrics;
            total_stress += comfort.stress;
            total_energy += comfort.energy;
            total_flux += comfort.flux;
            total_perturbation += comfort.perturbation;
            count += 1;
            
            if comfort.stress > self.high_stress_threshold {
                stressed_memories += 1;
            }
            if comfort.energy < self.low_energy_threshold {
                low_energy_memories += 1;
            }
            if comfort.flux > self.high_flux_threshold {
                high_flux_memories += 1;
            }
        }
        
        let count_f = count as f64;
        
        ComfortSystemReport {
            timestamp: Utc::now(),
            avg_stress: total_stress / count_f,
            avg_energy: total_energy / count_f,
            avg_flux: total_flux / count_f,
            avg_perturbation: total_perturbation / count_f,
            stressed_count: stressed_memories,
            low_energy_count: low_energy_memories,
            high_flux_count: high_flux_memories,
            total_memories: count,
            recent_feedback_count: self.feedback_history.len(),
            system_health: self.calculate_system_health(
                total_stress / count_f,
                total_energy / count_f,
                total_flux / count_f,
                total_perturbation / count_f
            ),
        }
    }
    
    fn calculate_system_health(&self, avg_stress: f64, avg_energy: f64, 
                              avg_flux: f64, avg_perturbation: f64) -> SystemHealth {
        let stress_score = 1.0 - avg_stress;
        let energy_score = avg_energy;
        let flux_score = 1.0 - avg_flux;
        let perturbation_score = 1.0 - avg_perturbation;
        
        let total_score = (stress_score + energy_score + flux_score + perturbation_score) / 4.0;
        
        if total_score > 0.8 {
            SystemHealth::Excellent
        } else if total_score > 0.6 {
            SystemHealth::Good
        } else if total_score > 0.4 {
            SystemHealth::Fair
        } else if total_score > 0.2 {
            SystemHealth::Poor
        } else {
            SystemHealth::Critical
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortSystemReport {
    pub timestamp: DateTime<Utc>,
    pub avg_stress: f64,
    pub avg_energy: f64,
    pub avg_flux: f64,
    pub avg_perturbation: f64,
    pub stressed_count: usize,
    pub low_energy_count: usize,
    pub high_flux_count: usize,
    pub total_memories: usize,
    pub recent_feedback_count: usize,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SystemHealth {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

// Integration with the main system
impl SolitonLattice {
    pub fn run_comfort_analysis(&mut self, analyzer: &mut ComfortAnalyzer) -> Vec<ComfortFeedback> {
        let mut all_feedback = Vec::new();
        let memory_ids: Vec<String> = self.memories.keys().cloned().collect();
        
        for memory_id in memory_ids {
            if let Some(memory) = self.memories.get(&memory_id) {
                if let Some(feedback) = analyzer.analyze_memory(memory, self) {
                    all_feedback.push(feedback);
                }
            }
        }
        
        // Apply critical feedback immediately
        for feedback in &all_feedback {
            if matches!(feedback.severity, FeedbackSeverity::Critical) {
                if let Some(memory) = self.memories.get_mut(&feedback.memory_id) {
                    let actions = analyzer.apply_feedback(&feedback, memory, self);
                    info!("Applied critical comfort feedback for {}: {:?}", 
                          feedback.memory_id, actions);
                }
            }
        }
        
        all_feedback
    }
}
