// concept-mesh/src/comfort_analysis.rs

use crate::soliton_memory::{ComfortVector, SolitonLattice, SolitonMemory};
use std::collections::HashMap;

/// Analyze comfort metrics for all solitons and suggest adjustments
pub struct ComfortAnalyzer {
    stress_threshold: f64,
    flux_threshold: f64,
    adjustment_rate: f64,
}

impl ComfortAnalyzer {
    pub fn new() -> Self {
        Self {
            stress_threshold: 0.7,
            flux_threshold: 0.8,
            adjustment_rate: 0.05,
        }
    }

    pub fn analyze_lattice(&self, lattice: &SolitonLattice) -> ComfortReport {
        let mut report = ComfortReport::default();

        // Compute comfort for each soliton
        for (id, memory) in &lattice.memories {
            let comfort = self.compute_comfort(memory, lattice);

            if comfort.stress > self.stress_threshold {
                report.stressed_solitons.push(id.clone());
            }

            if comfort.flux > self.flux_threshold {
                report.high_flux_solitons.push(id.clone());
            }

            report.average_stress += comfort.stress;
            report.average_flux += comfort.flux;
        }

        let n = lattice.memories.len() as f64;
        report.average_stress /= n;
        report.average_flux /= n;

        report
    }

    pub fn compute_comfort(
        &self,
        memory: &SolitonMemory,
        lattice: &SolitonLattice,
    ) -> ComfortVector {
        let mut comfort = memory.comfort_metrics.clone();

        // Update energy based on current amplitude
        comfort.energy = memory.amplitude;

        // Compute stress from stability
        comfort.stress = 1.0 - memory.stability;

        // Compute flux from coupling forces
        comfort.flux = self.compute_flux(memory, lattice);

        // Check for recent perturbations
        comfort.perturbation =
            if memory.last_accessed > chrono::Utc::now() - chrono::Duration::minutes(5) {
                1.0
            } else {
                comfort.perturbation * 0.9 // Decay
            };

        comfort
    }

    fn compute_flux(&self, memory: &SolitonMemory, lattice: &SolitonLattice) -> f64 {
        let mut total_flux = 0.0;
        let phase = memory.phase_tag;

        // Sum coupling forces from neighbors
        for (edge, weight) in &lattice.coupling_matrix {
            if edge.0 == memory.id || edge.1 == memory.id {
                let other_id = if edge.0 == memory.id {
                    &edge.1
                } else {
                    &edge.0
                };

                if let Some(other) = lattice.memories.get(other_id) {
                    // Kuramoto-like coupling force
                    let phase_diff = other.phase_tag - phase;
                    let force = weight * phase_diff.sin();
                    total_flux += force.abs();
                }
            }
        }

        total_flux
    }

    pub fn suggest_adjustments(
        &self,
        report: &ComfortReport,
        lattice: &mut SolitonLattice,
    ) -> Vec<LatticeAdjustment> {
        let mut adjustments = vec![];

        // Reduce coupling for high-stress solitons
        for id in &report.stressed_solitons {
            if let Some(memory) = lattice.memories.get(id) {
                let edges_to_weaken = self.find_stressful_edges(id, lattice);

                for edge in edges_to_weaken {
                    adjustments.push(LatticeAdjustment::WeakenCoupling { edge, factor: 0.8 });
                }
            }
        }

        // Add connections for isolated solitons
        for id in &report.isolated_solitons {
            if let Some(memory) = lattice.memories.get(id) {
                let candidates = self.find_connection_candidates(id, lattice);

                for other_id in candidates.iter().take(2) {
                    adjustments.push(LatticeAdjustment::AddCoupling {
                        edge: (id.clone(), other_id.clone()),
                        weight: 0.1,
                    });
                }
            }
        }

        adjustments
    }

    fn find_stressful_edges(&self, id: &str, lattice: &SolitonLattice) -> Vec<(String, String)> {
        let mut stressful = vec![];

        for (edge, weight) in &lattice.coupling_matrix {
            if (edge.0 == id || edge.1 == id) && *weight > 0.5 {
                stressful.push(edge.clone());
            }
        }

        stressful
    }

    fn find_connection_candidates(&self, id: &str, lattice: &SolitonLattice) -> Vec<String> {
        // Find nearby solitons with similar phase
        let mut candidates = vec![];

        if let Some(memory) = lattice.memories.get(id) {
            let phase = memory.phase_tag;

            for (other_id, other) in &lattice.memories {
                if other_id != id {
                    let phase_diff = (other.phase_tag - phase).abs();
                    if phase_diff < 0.5 {
                        // Similar phase
                        candidates.push(other_id.clone());
                    }
                }
            }
        }

        candidates
    }
}

#[derive(Default)]
pub struct ComfortReport {
    pub average_stress: f64,
    pub average_flux: f64,
    pub stressed_solitons: Vec<String>,
    pub high_flux_solitons: Vec<String>,
    pub isolated_solitons: Vec<String>,
}

pub enum LatticeAdjustment {
    WeakenCoupling { edge: (String, String), factor: f64 },
    StrengthenCoupling { edge: (String, String), factor: f64 },
    AddCoupling { edge: (String, String), weight: f64 },
    RemoveCoupling { edge: (String, String) },
}
