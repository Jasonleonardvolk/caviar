// TORI Soliton Memory Engine - Core Implementation
// File: concept-mesh/src/soliton_memory.rs

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use anyhow::Result;
use tracing::{info, debug};

use crate::wal::{WalEntry, WalWriter};
use crate::lattice_topology::{LatticeTopology, KagomeTopology};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SolitonMode {
    Bright,  // Standard memory (additive)
    Dark,    // Suppressive memory (subtractive)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortVector {
    pub energy: f64,      // Current energy level (amplitude-based)
    pub stress: f64,      // Stability measure (0=stable, 1=stressed)
    pub flux: f64,        // Net coupling forces from neighbors
    pub perturbation: f64, // Recent topology changes affecting this soliton
}

impl Default for ComfortVector {
    fn default() -> Self {
        Self {
            energy: 1.0, stress: 0.0, flux: 0.0, perturbation: 0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolitonMemory {
    pub id: String,
    pub concept_id: String,
    pub phase_tag: f64,           // ψᵢ - unique phase signature
    pub amplitude: f64,           // A - memory strength/importance
    pub frequency: f64,           // ω₀ - carrier frequency
    pub width: f64,              // T - temporal focus
    pub position: f64,           // x₀ - spatial position in lattice
    pub stability: f64,          // attractor depth (0.0-1.0)
    pub creation_time: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub content: String,         // the actual memory content
    pub content_type: ContentType,
    pub emotional_signature: EmotionalSignature,
    pub vault_status: VaultStatus,
    pub mode: SolitonMode,
    pub comfort_metrics: ComfortVector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Conversation,
    UploadedDocument,
    GeneratedInsight,
    UserReflection,
    AssociativeMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSignature {
    pub valence: f64,        // -1.0 (negative) to 1.0 (positive)
    pub arousal: f64,        // 0.0 (calm) to 1.0 (excited)
    pub dominance: f64,      // 0.0 (submissive) to 1.0 (dominant)
    pub trauma_indicators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaultStatus {
    Active,                  // Normally accessible
    UserSealed,             // User chose to seal (45° phase shift)
    TimeLocked,             // Temporarily protected (90° phase shift)  
    DeepVault,              // Maximum protection (180° phase shift)
    Quarantine,             // Isolated (for dark suppression)
}

impl SolitonMemory {
    pub fn new(concept_id: String, content: String, importance: f64) -> Self {
        let now = Utc::now();
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            concept_id: concept_id.clone(),
            phase_tag: Self::calculate_phase_tag(&concept_id),
            amplitude: importance.sqrt(), // √importance for stability
            frequency: 1.0, // Base frequency - can be adjusted
            width: 1.0 / (content.len() as f64).sqrt(), // Focus inversely related to complexity
            position: 0.0,
            stability: 0.8, // Default high stability
            creation_time: now,
            last_accessed: now,
            access_count: 0,
            content,
            content_type: ContentType::Conversation,
            emotional_signature: EmotionalSignature::neutral(),
            vault_status: VaultStatus::Active,
            mode: SolitonMode::Bright,
            comfort_metrics: ComfortVector::default(),
        }
    }
    
    pub fn new_dark(concept_id: String, content: String, importance: f64) -> Self {
        let mut memory = Self::new(concept_id, content, importance);
        memory.mode = SolitonMode::Dark;
        // Dark solitons have inverted phase
        memory.phase_tag = (memory.phase_tag + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
        memory
    }
    
    pub fn update_comfort(&mut self, lattice: &SolitonLattice) {
        // Update comfort metrics based on local lattice conditions
        self.comfort_metrics.energy = self.amplitude;
        self.comfort_metrics.stress = 1.0 - self.stability;
        // flux and perturbation computed from lattice topology
    }
    
    // Core soliton equation: Si(t) = A·sech((t-t₀)/T)·exp[j(ω₀t + ψᵢ)]
    pub fn evaluate_waveform(&self, t: f64) -> (f64, f64) {
        let envelope = self.amplitude * ((t - self.position) / self.width).tanh().sech();
        let phase = self.frequency * t + self.phase_tag;
        (envelope * phase.cos(), envelope * phase.sin())
    }
    
    // Calculate unique phase tag from concept ID
    fn calculate_phase_tag(concept_id: &str) -> f64 {
        let hash = md5::compute(concept_id.as_bytes());
        let hash_num = u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]);
        (hash_num as f64 / u32::MAX as f64) * 2.0 * PI
    }
    
    // Matched filter correlation for retrieval
    pub fn correlate_with_signal(&self, target_phase: f64, tolerance: f64) -> f64 {
        let phase_diff = (self.phase_tag - target_phase).abs();
        let normalized_diff = phase_diff.min(2.0 * PI - phase_diff); // Handle wraparound
        
        if normalized_diff <= tolerance {
            let correlation = (1.0 - normalized_diff / tolerance) * self.amplitude;
            match self.mode {
                SolitonMode::Bright => correlation,
                // Dark solitons contribute negative correlation
                SolitonMode::Dark => -correlation,
            }
        } else {
            0.0
        }
    }
    
    pub fn access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
        
        // Strengthen memory through access (Hebbian principle)
        self.amplitude = (self.amplitude * 1.01).min(2.0);
        self.stability = (self.stability * 1.005).min(1.0);
    }
    
    pub fn apply_vault_phase_shift(&mut self, vault_status: VaultStatus) {
        let original_phase = self.phase_tag;
        
        self.phase_tag = match vault_status {
            VaultStatus::Active => original_phase,
            VaultStatus::UserSealed => (original_phase + PI/4.0) % (2.0 * PI),
            VaultStatus::TimeLocked => (original_phase + PI/2.0) % (2.0 * PI),
            VaultStatus::DeepVault => (original_phase + PI) % (2.0 * PI),
            VaultStatus::Quarantine => (original_phase + PI * 1.5) % (2.0 * PI),
        };
        
        self.vault_status = vault_status;
    }
}

impl EmotionalSignature {
    pub fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            trauma_indicators: Vec::new(),
        }
    }
    
    pub fn analyze_content(content: &str) -> Self {
        // Simple emotional analysis - in production would use more sophisticated NLP
        let mut valence = 0.0;
        let mut arousal = 0.0;
        let mut dominance = 0.0;
        let mut trauma_indicators = Vec::new();
        
        let content_lower = content.to_lowercase();
        
        // Positive words
        if content_lower.contains("happy") || content_lower.contains("joy") || content_lower.contains("love") {
            valence += 0.3;
        }
        
        // Negative words  
        if content_lower.contains("sad") || content_lower.contains("pain") || content_lower.contains("hurt") {
            valence -= 0.3;
        }
        
        // High arousal words
        if content_lower.contains("excited") || content_lower.contains("urgent") || content_lower.contains("panic") {
            arousal += 0.4;
        }
        
        // Trauma indicators
        if content_lower.contains("trauma") || content_lower.contains("abuse") || content_lower.contains("nightmare") {
            trauma_indicators.push("potential_trauma".to_string());
            valence -= 0.5;
            arousal += 0.3;
        }
        
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(0.0, 1.0),
            trauma_indicators,
        }
    }
    
    pub fn requires_protection(&self) -> bool {
        self.valence < -0.4 || !self.trauma_indicators.is_empty()
    }
}

#[derive(Debug)]
pub struct SolitonLattice {
    pub memories: HashMap<String, SolitonMemory>,
    pub user_id: String,
    pub global_frequency: f64,
    pub coupling_matrix: HashMap<(String, String), f64>, // Concept relationships
    pub phase_registry: HashMap<String, f64>, // concept_id -> phase
    pub creation_count: u64,
    pub total_charge: f64,
    pub current_topology: Box<dyn LatticeTopology>,
    pub target_topology: Option<Box<dyn LatticeTopology>>,
    pub blend_rate: f64,
}

impl SolitonLattice {
    pub fn new(user_id: String) -> Self {
        let mut lattice = Self {
            memories: HashMap::new(),
            user_id,
            global_frequency: 1.0,
            coupling_matrix: HashMap::new(),
            phase_registry: HashMap::new(),
            creation_count: 0,
            total_charge: 0.0,
            current_topology: Box::new(KagomeTopology::new()),
            target_topology: None,
            blend_rate: 0.05,
        };
        
        // Initialize with default Kagome topology
        lattice.initialize_lattice(100);
        lattice
    }
    
    fn initialize_lattice(&mut self, size: usize) {
        // Initialize lattice with topology-specific coupling
        let coupling = self.current_topology.generate_coupling(size);
        // Convert to string-based keys for compatibility
        for ((i, j), weight) in coupling {
            let key = (format!("node_{}", i), format!("node_{}", j));
            self.coupling_matrix.insert(key, weight);
        }
    }
    
    pub fn store_memory(&mut self, concept_id: String, content: String, importance: f64) -> Result<String, String> {
        let emotional_sig = EmotionalSignature::analyze_content(&content);
        let mut memory = SolitonMemory::new(concept_id.clone(), content, importance);
        memory.emotional_signature = emotional_sig.clone();
        
        // Auto-vault traumatic content for user protection
        if emotional_sig.requires_protection() {
            memory.apply_vault_phase_shift(VaultStatus::UserSealed);
            println!("🛡️ Memory auto-sealed for protection: {}", concept_id);
        }
        
        // Update coupling matrix
        self.update_couplings(&memory.id);
        
        // Apply topology-specific placement
        self.current_topology.place_soliton(&mut memory, &self);
        
        self.phase_registry.insert(concept_id.clone(), memory.phase_tag);
        let memory_id = memory.id.clone();
        
        // Handle dark soliton suppression
        if memory.mode == SolitonMode::Dark {
            self.suppress_bright_memories(&memory.concept_id);
        }
        
        self.memories.insert(memory_id.clone(), memory);
        self.creation_count += 1;
        
        println!("✨ Soliton memory created: {} (Phase: {:.3})", concept_id, self.phase_registry[&concept_id]);
        
        Ok(memory_id)
    }
    
    fn update_couplings(&mut self, memory_id: &str) {
        // Update coupling matrix based on new memory
        // This is a placeholder - actual implementation would be more sophisticated
        for (other_id, _) in &self.memories {
            let coupling_strength = 0.1; // Default weak coupling
            self.coupling_matrix.insert((memory_id.to_string(), other_id.clone()), coupling_strength);
            self.coupling_matrix.insert((other_id.clone(), memory_id.to_string()), coupling_strength);
        }
    }
    
    fn suppress_bright_memories(&mut self, concept_id: &str) {
        // Find and suppress bright memories with same concept
        let to_suppress: Vec<String> = self.memories
            .iter()
            .filter(|(_, m)| m.concept_id == concept_id && m.mode == SolitonMode::Bright)
            .map(|(id, _)| id.clone())
            .collect();
            
        for id in to_suppress {
            if let Some(memory) = self.memories.get_mut(&id) {
                memory.vault_status = VaultStatus::Quarantine;
                info!("Suppressed bright memory {} due to dark soliton", id);
            }
        }
    }
    
    pub fn morph_topology(&mut self, new_topology: Box<dyn LatticeTopology>, blend_rate: f64) {
        self.target_topology = Some(new_topology);
        self.blend_rate = blend_rate;
        info!("Initiating topology morph with blend rate {}", blend_rate);
    }
    
    pub fn step_topology_blend(&mut self) {
        // Implemented in lattice_topology.rs
        // This would gradually blend the coupling matrices
    }
    
    pub fn recall_by_concept(&mut self, concept_id: &str) -> Option<&mut SolitonMemory> {
        // Find memory by concept_id
        let memory_id = self.memories.iter()
            .find(|(_, memory)| memory.concept_id == concept_id)
            .map(|(id, _)| id.clone())?;
            
        if let Some(memory) = self.memories.get_mut(&memory_id) {
            memory.access();
            Some(memory)
        } else {
            None
        }
    }
    
    pub fn recall_by_phase(&mut self, target_phase: f64, tolerance: f64) -> Vec<&mut SolitonMemory> {
        let mut results = Vec::new();
        
        for memory in self.memories.values_mut() {
            let correlation = memory.correlate_with_signal(target_phase, tolerance);
            if correlation > 0.0 {
                // Skip dark memories and suppressed bright memories
                if memory.mode == SolitonMode::Bright && 
                   memory.vault_status == VaultStatus::Active {
                    memory.access();
                    results.push(memory);
                }
                // Dark memories are used for suppression, not direct recall
            }
        }
        
        // Sort by correlation strength
        results.sort_by(|a, b| {
            b.correlate_with_signal(target_phase, tolerance)
                .partial_cmp(&a.correlate_with_signal(target_phase, tolerance))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        results
    }
    
    pub fn find_related_memories(&mut self, concept_id: &str, max_results: usize) -> Vec<&SolitonMemory> {
        if let Some(&target_phase) = self.phase_registry.get(concept_id) {
            let tolerance = PI / 4.0; // 45 degree tolerance for associations
            
            let mut related: Vec<_> = self.memories.values()
                .filter(|memory| {
                    memory.concept_id != concept_id && 
                    memory.correlate_with_signal(target_phase, tolerance) > 0.0
                })
                .collect();
            
            related.sort_by(|a, b| {
                b.correlate_with_signal(target_phase, tolerance)
                    .partial_cmp(&a.correlate_with_signal(target_phase, tolerance))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
            related.into_iter().take(max_results).collect()
        } else {
            Vec::new()
        }
    }
    
    pub fn nightly_adapt(&mut self) {
        info!("=== Starting Nightly Memory Adaptation Cycle ===");
        
        // 1. Switch to diffusive topology for interaction
        let temp_topology = Box::new(crate::lattice_topology::AllToAllTopology::new(0.05));
        self.morph_topology(temp_topology, 0.1);
        
        // 2. Let topology settle
        for _ in 0..10 {
            self.step_topology_blend();
        }
        
        // 3. Perform memory operations
        let fusion_count = self.perform_fusion();
        let fission_count = self.perform_fission();
        let collision_results = self.resolve_collisions();
        
        info!("Fusion: {} pairs merged", fusion_count);
        info!("Fission: {} memories split", fission_count);
        info!("Collisions: {} resolved", collision_results);
        
        // 4. Crystallization - migrate hot memories to stable positions
        self.crystallize_memories();
        
        // 5. Return to stable Kagome topology
        let stable_topology = Box::new(KagomeTopology::new());
        self.morph_topology(stable_topology, 0.05);
        
        for _ in 0..20 {
            self.step_topology_blend();
        }
        
        info!("=== Nightly Adaptation Complete ===");
    }
    
    fn perform_fusion(&mut self) -> usize {
        // Group memories by concept
        let mut concept_groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for (id, memory) in &self.memories {
            concept_groups.entry(memory.concept_id.clone())
                .or_insert_with(Vec::new)
                .push(id.clone());
        }
        
        let mut fusion_count = 0;
        
        for (concept, ids) in concept_groups {
            if ids.len() > 1 {
                // Keep strongest memory, merge others into it
                let mut strongest_id = ids[0].clone();
                let mut max_amplitude = 0.0;
                
                for id in &ids {
                    if let Some(mem) = self.memories.get(id) {
                        if mem.amplitude > max_amplitude {
                            max_amplitude = mem.amplitude;
                            strongest_id = id.clone();
                        }
                    }
                }
                
                // Merge other memories into strongest
                for id in ids {
                    if id != strongest_id {
                        if let Some(mem) = self.memories.remove(&id) {
                            if let Some(strong_mem) = self.memories.get_mut(&strongest_id) {
                                // Combine amplitudes (bounded)
                                strong_mem.amplitude = (strong_mem.amplitude + mem.amplitude * 0.5).min(2.0);
                                strong_mem.access_count += mem.access_count;
                                fusion_count += 1;
                            }
                        }
                    }
                }
            }
        }
        
        fusion_count
    }
    
    fn perform_fission(&mut self) -> usize {
        // Placeholder - would split complex memories
        0
    }
    
    fn resolve_collisions(&mut self) -> usize {
        // Placeholder - would resolve bright/dark collisions
        0
    }
    
    fn crystallize_memories(&mut self) {
        // Placeholder - would migrate hot memories to stable positions
    }
    
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_memories = self.memories.len();
        let active_memories = self.memories.values()
            .filter(|m| matches!(m.vault_status, VaultStatus::Active))
            .count();
        let vaulted_memories = total_memories - active_memories;
        
        let average_stability = if total_memories > 0 {
            self.memories.values().map(|m| m.stability).sum::<f64>() / total_memories as f64
        } else {
            0.0
        };
        
        MemoryStats {
            total_memories,
            active_memories,
            vaulted_memories,
            average_stability,
            creation_count: self.creation_count,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub active_memories: usize,
    pub vaulted_memories: usize,
    pub average_stability: f64,
    pub creation_count: u64,
}

// Traits for mathematical operations
trait SechFunction {
    fn sech(self) -> Self;
}

impl SechFunction for f64 {
    fn sech(self) -> Self {
        1.0 / self.cosh()
    }
}

/// SolitonMemoryVault with WAL integration for persistence
pub struct SolitonMemoryVault {
    lattice: SolitonLattice,
    wal_writer: Option<Arc<WalWriter>>,
}

impl SolitonMemoryVault {
    pub fn new(user_id: String) -> Self {
        Self {
            lattice: SolitonLattice::new(user_id),
            wal_writer: None,
        }
    }
    
    /// Initialize with WAL support
    pub async fn with_wal(user_id: String, wal_writer: Arc<WalWriter>) -> Self {
        Self {
            lattice: SolitonLattice::new(user_id),
            wal_writer: Some(wal_writer),
        }
    }
    
    /// Store memory with WAL logging
    pub async fn store_memory(
        &mut self, 
        concept_id: String, 
        content: String, 
        importance: f64
    ) -> Result<String> {
        // Store in lattice
        let memory_id = self.lattice.store_memory(concept_id.clone(), content.clone(), importance)
            .map_err(|e| anyhow::anyhow!(e))?;
        
        // Log to WAL if available
        if let Some(wal) = &self.wal_writer {
            let entry = WalEntry::MemoryOp {
                operation: "store".to_string(),
                key: format!("soliton:{}", memory_id),
                value: Some(serde_json::to_vec(&self.lattice.memories[&memory_id])?),
                timestamp: Utc::now(),
            };
            
            wal.write(&entry).await?;
            debug!("Logged soliton memory to WAL: {}", memory_id);
        }
        
        Ok(memory_id)
    }
    
    /// Recall memory with WAL logging
    pub async fn recall_by_concept(&mut self, concept_id: &str) -> Option<&mut SolitonMemory> {
        let result = self.lattice.recall_by_concept(concept_id);
        
        // Log access to WAL
        if result.is_some() {
            if let Some(wal) = &self.wal_writer {
                let entry = WalEntry::MemoryOp {
                    operation: "access".to_string(),
                    key: format!("concept:{}", concept_id),
                    value: None,
                    timestamp: Utc::now(),
                };
                
                // Fire and forget - don't block on WAL write
                let wal_clone = wal.clone();
                tokio::spawn(async move {
                    if let Err(e) = wal_clone.write(&entry).await {
                        tracing::warn!("Failed to log memory access: {}", e);
                    }
                });
            }
        }
        
        result
    }
    
    /// Apply vault phase shift with WAL logging
    pub async fn vault_memory(&mut self, memory_id: &str, vault_status: VaultStatus) -> Result<()> {
        let memory = self.lattice.memories.get_mut(memory_id)
            .ok_or_else(|| anyhow::anyhow!("Memory not found"))?;
        
        let old_status = memory.vault_status.clone();
        memory.apply_vault_phase_shift(vault_status.clone());
        
        // Log vault operation
        if let Some(wal) = &self.wal_writer {
            let entry = WalEntry::PhaseChange {
                from_phase: format!("{:?}", old_status),
                to_phase: format!("{:?}", vault_status),
                timestamp: Utc::now(),
            };
            
            wal.write(&entry).await?;
            info!("Vaulted memory {} with status {:?}", memory_id, vault_status);
        }
        
        Ok(())
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.lattice.get_memory_stats()
    }
}
