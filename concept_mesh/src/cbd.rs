//! Concept Boundary Detector (CBD)
//!
//! The CBD is responsible for segmenting content at semantic breakpoints rather than
//! arbitrary fixed-size chunks. It uses phase shifts, Koopman-mode inflections, and
//! eigen-entropy slopes to determine natural concept boundaries.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration parameters for the CBD algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CBDConfig {
    /// Threshold for semantic shift detection (concept overlap ratio)
    pub semantic_shift_threshold: f64,

    /// Threshold for Koopman mode inflection detection
    pub koopman_inflection_threshold: f64,

    /// Threshold for entropy slope change detection
    pub entropy_slope_threshold: f64,

    /// Minimum segment size in tokens/characters
    pub min_segment_size: usize,

    /// Maximum segment size in tokens/characters
    pub max_segment_size: usize,

    /// Window size for computing metrics
    pub metrics_window_size: usize,

    /// Whether to use simplified heuristics (Day 1) or full spectral analysis
    pub use_simplified_detection: bool,
}

impl Default for CBDConfig {
    fn default() -> Self {
        Self {
            semantic_shift_threshold: 0.3,
            koopman_inflection_threshold: 0.25,
            entropy_slope_threshold: 0.15,
            min_segment_size: 100,
            max_segment_size: 2000,
            metrics_window_size: 50,
            use_simplified_detection: true,
        }
    }
}

/// Reason for a concept boundary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryReason {
    /// Semantic shift (topic change)
    SemanticShift,

    /// Koopman mode inflection
    KoopmanInflection,

    /// Entropy slope change
    EntropySlope,

    /// Maximum segment size reached
    MaxSegmentSize,

    /// Explicit marker (e.g., section heading)
    ExplicitMarker,

    /// End of document
    EndOfDocument,
}

/// Phase vector representing a concept embedding
pub type PhaseVector = Array1<f32>;

/// Map of concept activations and strengths
pub type ConceptActivation = HashMap<String, f32>;

/// A ConceptPack represents a coherent segment of content aligned to a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPack {
    /// Unique identifier for this ConceptPack
    pub id: String,

    /// The raw content of this segment
    pub content: String,

    /// The start position of this segment in the original content
    pub start_pos: usize,

    /// The end position of this segment in the original content
    pub end_pos: usize,

    /// Reason this boundary was detected
    pub boundary_reason: BoundaryReason,

    /// Phase seed vector for this concept segment
    #[serde(skip)]
    pub phase_seed: Option<PhaseVector>,

    /// Concept activations for this segment
    pub concepts: ConceptActivation,

    /// Metadata about this segment
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ConceptPack {
    /// Create a new ConceptPack
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        start_pos: usize,
        end_pos: usize,
        boundary_reason: BoundaryReason,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            start_pos,
            end_pos,
            boundary_reason,
            phase_seed: None,
            concepts: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the phase seed vector for this ConceptPack
    pub fn with_phase_seed(mut self, phase_seed: PhaseVector) -> Self {
        self.phase_seed = Some(phase_seed);
        self
    }

    /// Add a concept activation to this ConceptPack
    pub fn with_concept(mut self, concept: impl Into<String>, strength: f32) -> Self {
        self.concepts.insert(concept.into(), strength);
        self
    }

    /// Add metadata to this ConceptPack
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), value);
        }
        self
    }

    /// Get the length of this ConceptPack in characters
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if this ConceptPack is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Reference-counted ConceptPack
pub type ConceptPackRef = Arc<ConceptPack>;

/// Callback function for when a ConceptPack is detected
pub type ConceptPackHandler = Box<dyn Fn(ConceptPackRef) + Send + Sync>;

/// The Concept Boundary Detector detects natural boundaries in content
pub struct ConceptBoundaryDetector {
    /// Configuration for the CBD algorithm
    config: CBDConfig,

    /// Current accumulated content buffer
    buffer: String,

    /// Current position in the original content
    current_pos: usize,

    /// History of recent concept activations
    concept_history: Vec<ConceptActivation>,

    /// History of entropy values
    entropy_history: Vec<f64>,

    /// Handler for detected ConceptPacks
    handler: Option<ConceptPackHandler>,

    /// Next segment ID to assign
    next_segment_id: usize,

    /// Document ID this detector is processing
    document_id: String,
}

impl ConceptBoundaryDetector {
    /// Create a new CBD with default configuration
    pub fn new(document_id: impl Into<String>) -> Self {
        Self::with_config(document_id, CBDConfig::default())
    }

    /// Create a new CBD with custom configuration
    pub fn with_config(document_id: impl Into<String>, config: CBDConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            current_pos: 0,
            concept_history: Vec::new(),
            entropy_history: Vec::new(),
            handler: None,
            next_segment_id: 1,
            document_id: document_id.into(),
        }
    }

    /// Set the handler for detected ConceptPacks
    pub fn set_handler<F>(&mut self, handler: F)
    where
        F: Fn(ConceptPackRef) + Send + Sync + 'static,
    {
        self.handler = Some(Box::new(handler));
    }

    /// Process a chunk of content
    pub fn process_chunk(&mut self, chunk: &str) {
        // Append to buffer
        self.buffer.push_str(chunk);

        // If we've accumulated enough content, check for boundaries
        if self.buffer.len() >= self.config.min_segment_size {
            self.detect_boundaries();
        }
    }

    /// Finalize processing and emit any remaining content
    pub fn finalize(&mut self) {
        if !self.buffer.is_empty() {
            let concept_pack = self.create_concept_pack(BoundaryReason::EndOfDocument);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
        }
    }

    /// Detect concept boundaries in the current buffer
    fn detect_boundaries(&mut self) {
        // In a real implementation, this would perform sophisticated
        // analysis based on phase shifts, Koopman modes, and entropy.
        // For Day 1, we use simplified heuristics.

        if self.config.use_simplified_detection {
            self.detect_boundaries_simplified();
        } else {
            self.detect_boundaries_spectral();
        }
    }

    /// Simplified boundary detection using heuristics
    fn detect_boundaries_simplified(&mut self) {
        // Look for explicit markers first (headings, paragraph breaks, etc.)
        if let Some(pos) = self.find_explicit_markers() {
            let concept_pack = self.extract_concept_pack(pos, BoundaryReason::ExplicitMarker);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            return;
        }

        // Check if we've reached max segment size
        if self.buffer.len() >= self.config.max_segment_size {
            let concept_pack = self.create_concept_pack(BoundaryReason::MaxSegmentSize);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
        }
    }

    /// Advanced boundary detection using spectral analysis
    fn detect_boundaries_spectral(&mut self) {
        // Compute current concept activations
        let activations = self.compute_concept_activations();

        // Add to history
        self.concept_history.push(activations.clone());
        if self.concept_history.len() > self.config.metrics_window_size {
            self.concept_history.remove(0);
        }

        // Compute entropy
        let entropy = self.compute_entropy(&activations);
        self.entropy_history.push(entropy);
        if self.entropy_history.len() > self.config.metrics_window_size {
            self.entropy_history.remove(0);
        }

        // Check for boundary conditions

        // 1. Semantic shift
        if self.detect_semantic_shift() {
            let concept_pack = self.create_concept_pack(BoundaryReason::SemanticShift);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
            return;
        }

        // 2. Koopman mode inflection
        if self.detect_koopman_inflection() {
            let concept_pack = self.create_concept_pack(BoundaryReason::KoopmanInflection);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
            return;
        }

        // 3. Entropy slope change
        if self.detect_entropy_slope_change() {
            let concept_pack = self.create_concept_pack(BoundaryReason::EntropySlope);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
            return;
        }

        // 4. Max segment size (fallback)
        if self.buffer.len() >= self.config.max_segment_size {
            let concept_pack = self.create_concept_pack(BoundaryReason::MaxSegmentSize);
            if let Some(handler) = &self.handler {
                handler(Arc::new(concept_pack));
            }
            self.buffer.clear();
        }
    }

    /// Look for explicit section markers (headings, etc.)
    fn find_explicit_markers(&self) -> Option<usize> {
        // Simple heuristic - look for Markdown-style headings or line breaks
        let marker_patterns = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            "\n\n",
            "\r\n\r\n",
        ];

        for pattern in marker_patterns {
            if let Some(pos) = self.buffer.find(pattern) {
                // Ensure we have at least min_segment_size before the marker
                if pos >= self.config.min_segment_size {
                    return Some(pos);
                }
            }
        }

        None
    }

    /// Extract a ConceptPack up to the given position
    fn extract_concept_pack(&mut self, pos: usize, reason: BoundaryReason) -> ConceptPack {
        let content = self.buffer[..pos].to_string();
        let end_pos = self.current_pos + pos;

        // Update state
        self.buffer = self.buffer[pos..].to_string();
        self.current_pos = end_pos;

        // Create ConceptPack
        let id = format!("{}_{}", self.document_id, self.next_segment_id);
        self.next_segment_id += 1;

        let mut pack = ConceptPack::new(id, content, self.current_pos - pos, end_pos, reason);

        // In a real implementation, we would compute proper concept activations and phase seed
        // For Day 1, we just add some placeholder metadata
        pack.metadata.insert(
            "word_count".to_string(),
            serde_json::json!(content.split_whitespace().count()),
        );

        pack
    }

    /// Create a ConceptPack from the current buffer
    fn create_concept_pack(&mut self, reason: BoundaryReason) -> ConceptPack {
        let content = self.buffer.clone();
        let start_pos = self.current_pos;
        let end_pos = self.current_pos + content.len();

        // Update state
        self.current_pos = end_pos;

        // Create ConceptPack
        let id = format!("{}_{}", self.document_id, self.next_segment_id);
        self.next_segment_id += 1;

        let mut pack = ConceptPack::new(id, content, start_pos, end_pos, reason);

        // Compute concept activations (simplified for Day 1)
        let activations = self.compute_concept_activations();
        pack.concepts = activations;

        // Add metadata
        pack.metadata.insert(
            "word_count".to_string(),
            serde_json::json!(pack.content.split_whitespace().count()),
        );

        pack
    }

    /// Compute concept activations for the current buffer (simplified)
    fn compute_concept_activations(&self) -> ConceptActivation {
        // In a real implementation, this would use a sophisticated algorithm
        // to determine which concepts are active in the current buffer.
        // For Day 1, we use a simple keyword-based approach.

        let mut activations = HashMap::new();
        let keywords = [
            ("mathematics", "MATH"),
            ("algorithm", "ALGORITHM"),
            ("function", "FUNCTION"),
            ("data", "DATA"),
            ("structure", "DATA_STRUCTURE"),
            ("system", "SYSTEM"),
            ("concept", "CONCEPT"),
        ];

        let buffer_lower = self.buffer.to_lowercase();

        for (keyword, concept) in keywords {
            let count = buffer_lower.matches(keyword).count() as f32;
            if count > 0.0 {
                let normalized = (count / (self.buffer.len() as f32 / 100.0)).min(1.0);
                activations.insert(concept.to_string(), normalized);
            }
        }

        activations
    }

    /// Compute entropy for the given concept activations
    fn compute_entropy(&self, activations: &ConceptActivation) -> f64 {
        // Shannon entropy: -Σ p_i * log(p_i)
        if activations.is_empty() {
            return 0.0;
        }

        let total: f32 = activations.values().sum();
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &strength in activations.values() {
            let p = strength / total;
            if p > 0.0 {
                entropy -= (p as f64) * (p as f64).ln();
            }
        }

        entropy
    }

    /// Detect semantic shift by comparing current and historical concept activations
    fn detect_semantic_shift(&self) -> bool {
        if self.concept_history.len() < 2 {
            return false;
        }

        let current = self.concept_history.last().unwrap();
        let historical_avg = self.compute_historical_avg();

        let overlap = self.concept_overlap(current, &historical_avg);
        overlap < self.config.semantic_shift_threshold
    }

    /// Compute historical average of concept activations
    fn compute_historical_avg(&self) -> ConceptActivation {
        let mut avg = HashMap::new();
        let len = self.concept_history.len().saturating_sub(1); // Exclude current

        if len == 0 {
            return avg;
        }

        for i in 0..len {
            for (concept, &strength) in &self.concept_history[i] {
                *avg.entry(concept.clone()).or_insert(0.0) += strength / (len as f32);
            }
        }

        avg
    }

    /// Compute concept overlap between two activation sets
    fn concept_overlap(&self, a: &ConceptActivation, b: &ConceptActivation) -> f64 {
        let mut overlap = 0.0;
        let mut total = 0.0;

        // For each concept in a, find its overlap with b
        for (concept, &strength_a) in a {
            if let Some(&strength_b) = b.get(concept) {
                overlap += (strength_a.min(strength_b)) as f64;
            }
            total += strength_a as f64;
        }

        // Add strength from concepts only in b
        for (concept, &strength_b) in b {
            if !a.contains_key(concept) {
                total += strength_b as f64;
            }
        }

        if total > 0.0 {
            overlap / total
        } else {
            1.0 // If no concepts are active, consider full overlap
        }
    }

    /// Detect Koopman mode inflection (simplified)
    fn detect_koopman_inflection(&self) -> bool {
        // In a real implementation, this would use EDMD to detect eigenmode changes
        // For Day 1, we use a simple approximation based on sudden activation changes

        if self.concept_history.len() < 3 {
            return false;
        }

        let current = self.concept_history.last().unwrap();
        let previous = &self.concept_history[self.concept_history.len() - 2];

        // Compute "velocity" of concept changes
        let mut velocity_sq = 0.0;
        let mut total_concepts = 0;

        let all_concepts = current
            .keys()
            .chain(previous.keys())
            .collect::<std::collections::HashSet<_>>();

        for concept in all_concepts {
            let current_val = current.get(concept).copied().unwrap_or(0.0);
            let prev_val = previous.get(concept).copied().unwrap_or(0.0);

            let diff = current_val - prev_val;
            velocity_sq += (diff * diff) as f64;
            total_concepts += 1;
        }

        if total_concepts == 0 {
            return false;
        }

        let avg_velocity = (velocity_sq / total_concepts as f64).sqrt();
        avg_velocity > self.config.koopman_inflection_threshold
    }

    /// Detect entropy slope change
    fn detect_entropy_slope_change(&self) -> bool {
        if self.entropy_history.len() < 3 {
            return false;
        }

        let len = self.entropy_history.len();
        let current_slope = self.entropy_history[len - 1] - self.entropy_history[len - 2];
        let previous_slope = self.entropy_history[len - 2] - self.entropy_history[len - 3];

        let slope_change = (current_slope - previous_slope).abs();
        slope_change > self.config.entropy_slope_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_concept_pack_creation() {
        let pack = ConceptPack::new(
            "test_1",
            "This is a test",
            0,
            14,
            BoundaryReason::ExplicitMarker,
        );
        assert_eq!(pack.id, "test_1");
        assert_eq!(pack.content, "This is a test");
        assert_eq!(pack.boundary_reason, BoundaryReason::ExplicitMarker);
    }

    #[test]
    fn test_simplified_boundary_detection() {
        let mut cbd = ConceptBoundaryDetector::new("doc1");
        let detected = Arc::new(Mutex::new(Vec::new()));

        let detected_clone = detected.clone();
        cbd.set_handler(move |pack| {
            detected_clone.lock().unwrap().push(pack);
        });

        // Process a text with explicit markers
        cbd.process_chunk("This is the first section.\n\n");
        cbd.process_chunk("This is the second section.\n## Heading\n");
        cbd.process_chunk("This is the third section.");
        cbd.finalize();

        let detected = detected.lock().unwrap();
        assert_eq!(detected.len(), 3);
        assert_eq!(detected[0].boundary_reason, BoundaryReason::ExplicitMarker);
        assert_eq!(detected[1].boundary_reason, BoundaryReason::ExplicitMarker);
        assert_eq!(detected[2].boundary_reason, BoundaryReason::EndOfDocument);
    }
}
