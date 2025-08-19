// TORI Soliton Memory Engine - Core Implementation
// File: concept-mesh/src/soliton_memory.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use tracing::{debug, info};

use crate::wal::{writer::WalWriter, WalEntry};

pub type ComfortVector = Vec<f64>;

pub struct SolitonMemory {
    pub id: String,
    pub concept_id: String,
    pub phase_tag: f64, // Ïˆáµ¢ - unique phase signature
    pub amplitude: f64, // A - memory strength/importance
    pub frequency: f64, // Ï‰â‚€ - carrier frequency
    pub width: f64,     // T - temporal focus
    pub position: f64,  // xâ‚€ - spatial position in lattice
    pub stability: f64, // attractor depth (0.0-1.0)
    pub creation_time: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub content: String, // the actual memory content
    pub content_type: ContentType,
    pub emotional_signature: EmotionalSignature,
    pub vault_status: VaultStatus,
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
    pub valence: f64,   // -1.0 (negative) to 1.0 (positive)
    pub arousal: f64,   // 0.0 (calm) to 1.0 (excited)
    pub dominance: f64, // 0.0 (submissive) to 1.0 (dominant)
    pub trauma_indicators: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaultStatus {
    Active,     // Normally accessible
    UserSealed, // User chose to seal (45Â° phase shift)
    TimeLocked, // Temporarily protected (90Â° phase shift)
    DeepVault,  // Maximum protection (180Â° phase shift)
}

impl SolitonMemory {
    pub fn new(concept_id: String, content: String, importance: f64) -> Self {
        let now = Utc::now();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            concept_id: concept_id.clone(),
            phase_tag: Self::calculate_phase_tag(&concept_id),
            amplitude: importance.sqrt(), // âˆšimportance for stability
            frequency: 1.0,               // Base frequency - can be adjusted
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
        }
    }

    // Core soliton equation: Si(t) = AÂ·sech((t-tâ‚€)/T)Â·exp[j(Ï‰â‚€t + Ïˆáµ¢)]
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
            (1.0 - normalized_diff / tolerance) * self.amplitude
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
            VaultStatus::UserSealed => (original_phase + PI / 4.0) % (2.0 * PI),
            VaultStatus::TimeLocked => (original_phase + PI / 2.0) % (2.0 * PI),
            VaultStatus::DeepVault => (original_phase + PI) % (2.0 * PI),
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
        if content_lower.contains("happy")
            || content_lower.contains("joy")
            || content_lower.contains("love")
        {
            valence += 0.3;
        }

        // Negative words
        if content_lower.contains("sad")
            || content_lower.contains("pain")
            || content_lower.contains("hurt")
        {
            valence -= 0.3;
        }

        // High arousal words
        if content_lower.contains("excited")
            || content_lower.contains("urgent")
            || content_lower.contains("panic")
        {
            arousal += 0.4;
        }

        // Trauma indicators
        if content_lower.contains("trauma")
            || content_lower.contains("abuse")
            || content_lower.contains("nightmare")
        {
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
    pub phase_registry: HashMap<String, f64>,            // concept_id -> phase
    pub creation_count: u64,
}

impl SolitonLattice {
    pub fn new(user_id: String) -> Self {
        Self {
            memories: HashMap::new(),
            user_id,
            global_frequency: 1.0,
            coupling_matrix: HashMap::new(),
            phase_registry: HashMap::new(),
            creation_count: 0,
        }
    }

    pub fn store_memory(
        &mut self,
        concept_id: String,
        content: String,
        importance: f64,
    ) -> Result<String, String> {
        let emotional_sig = EmotionalSignature::analyze_content(&content);
        let mut memory = SolitonMemory::new(concept_id.clone(), content, importance);
        memory.emotional_signature = emotional_sig.clone();

        // Auto-vault traumatic content for user protection
        if emotional_sig.requires_protection() {
            memory.apply_vault_phase_shift(VaultStatus::UserSealed);
            println!("ðŸ›¡ï¸ Memory auto-sealed for protection: {}", concept_id);
        }

        self.phase_registry
            .insert(concept_id.clone(), memory.phase_tag);
        let memory_id = memory.id.clone();
        self.memories.insert(memory_id.clone(), memory);
        self.creation_count += 1;

        println!(
            "âœ¨ Soliton memory created: {} (Phase: {:.3})",
            concept_id, self.phase_registry[&concept_id]
        );

        Ok(memory_id)
    }

    pub fn recall_by_concept(&mut self, concept_id: &str) -> Option<&mut SolitonMemory> {
        // Find memory by concept_id
        let memory_id = self
            .memories
            .iter()
            .find(|(_, memory)| memory.concept_id == concept_id)
            .map(|(id, _)| id.clone())?;

        if let Some(memory) = self.memories.get_mut(&memory_id) {
            memory.access();
            Some(memory)
        } else {
            None
        }
    }

    pub fn recall_by_phase(
        &mut self,
        target_phase: f64,
        tolerance: f64,
    ) -> Vec<&mut SolitonMemory> {
        let mut matches = Vec::new();

        for memory in self.memories.values_mut() {
            if memory.correlate_with_signal(target_phase, tolerance) > 0.0 {
                memory.access();
                matches.push(memory);
            }
        }

        // Sort by correlation strength
        matches.sort_by(|a, b| {
            b.correlate_with_signal(target_phase, tolerance)
                .partial_cmp(&a.correlate_with_signal(target_phase, tolerance))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    pub fn find_related_memories(
        &mut self,
        concept_id: &str,
        max_results: usize,
    ) -> Vec<&SolitonMemory> {
        if let Some(&target_phase) = self.phase_registry.get(concept_id) {
            let tolerance = PI / 4.0; // 45 degree tolerance for associations

            let mut related: Vec<_> = self
                .memories
                .values()
                .filter(|memory| {
                    memory.concept_id != concept_id
                        && memory.correlate_with_signal(target_phase, tolerance) > 0.0
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

    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_memories = self.memories.len();
        let active_memories = self
            .memories
            .values()
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
        importance: f64,
    ) -> Result<String> {
        // Store in lattice
        let memory_id = self
            .lattice
            .store_memory(concept_id.clone(), content.clone(), importance)
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
        let memory = self
            .lattice
            .memories
            .get_mut(memory_id)
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
            info!(
                "Vaulted memory {} with status {:?}",
                memory_id, vault_status
            );
        }

        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.lattice.get_memory_stats()
    }
}
