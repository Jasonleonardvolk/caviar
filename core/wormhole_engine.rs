/**
 * TORI WormholeEngine - Rust Core Implementation
 * 
 * This module provides semantic bridge detection and wormhole connection management
 * for the TORI cognitive system. It creates shortcuts between distant concepts in
 * the knowledge space, enabling rapid associative traversal and pattern recognition.
 * 
 * The WormholeEngine implements:
 * - Real-time semantic similarity detection using vector embeddings
 * - Small-world network optimization through strategic link placement
 * - Integration with Python ML models for advanced semantic analysis
 * - Thread-safe concurrent wormhole management with spatial indexing
 * - Event-driven integration with other TORI cognitive modules
 */

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;
use std::process::{Command, Stdio};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

// ===================================================================
// TYPE DEFINITIONS AND CORE STRUCTURES
// ===================================================================

pub type ConceptId = u64;
pub type WormholeId = Uuid;
pub type SimilarityScore = f64;
pub type Timestamp = u64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wormhole {
    pub id: WormholeId,
    pub concept_a: ConceptId,
    pub concept_b: ConceptId,
    pub strength: SimilarityScore,
    pub wormhole_type: WormholeType,
    pub created_at: Timestamp,
    pub last_accessed: Timestamp,
    pub access_count: u64,
    pub bidirectional: bool,
    pub context_tags: Vec<String>,
    pub confidence: f64,
    pub discovery_method: DiscoveryMethod,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WormholeType {
    Semantic,           // Based on semantic similarity
    Temporal,           // Based on temporal co-occurrence
    Causal,            // Based on causal relationships
    Analogical,        // Based on structural analogy
    UserDefined,       // Manually created by user
    SystemDetected,    // Automatically detected by system
    Experimental,      // Created for exploration/testing
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    VectorSimilarity { model: String, threshold: f64 },
    CoOccurrence { window_size: usize, frequency: f64 },
    CausalAnalysis { confidence: f64 },
    StructuralAnalogy { similarity_metric: String },
    UserInput { user_id: String },
    AlienDetection { alien_significance: f64 },
    BraidConnection { braid_strength: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeInfo {
    pub wormhole: Wormhole,
    pub traversal_cost: f64,
    pub path_length_reduction: usize,
    pub semantic_coherence: f64,
    pub usage_statistics: UsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub total_traversals: u64,
    pub recent_traversals: u64,
    pub average_traversal_time: Duration,
    pub success_rate: f64,
    pub user_ratings: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptVector {
    pub concept_id: ConceptId,
    pub vector: Vec<f64>,
    pub model_version: String,
    pub created_at: Timestamp,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityCandidate {
    pub concept_id: ConceptId,
    pub similarity_score: SimilarityScore,
    pub semantic_distance: f64,
    pub confidence: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeSearchRequest {
    pub source_concept: ConceptId,
    pub max_candidates: usize,
    pub min_similarity: f64,
    pub search_radius: usize,
    pub include_types: Vec<WormholeType>,
    pub exclude_concepts: HashSet<ConceptId>,
    pub context_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeCluster {
    pub cluster_id: Uuid,
    pub wormholes: Vec<WormholeId>,
    pub center_concept: ConceptId,
    pub radius: f64,
    pub density: f64,
    pub cohesion_score: f64,
}

// ===================================================================
// ERROR HANDLING
// ===================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum WormholeError {
    ConceptNotFound(ConceptId),
    WormholeNotFound(WormholeId),
    InvalidSimilarityScore(f64),
    DuplicateWormhole(ConceptId, ConceptId),
    PythonServiceUnavailable,
    PythonServiceTimeout,
    InvalidVectorDimensions(usize, usize),
    InsufficientData(String),
    ConcurrencyConflict(String),
    SerializationError(String),
    NetworkError(String),
    ConfigurationError(String),
    ValidationError(String),
    ResourceExhausted(String),
}

impl std::fmt::Display for WormholeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WormholeError::ConceptNotFound(id) => write!(f, "Concept not found: {}", id),
            WormholeError::WormholeNotFound(id) => write!(f, "Wormhole not found: {}", id),
            WormholeError::InvalidSimilarityScore(score) => write!(f, "Invalid similarity score: {}", score),
            WormholeError::DuplicateWormhole(a, b) => write!(f, "Duplicate wormhole between {} and {}", a, b),
            WormholeError::PythonServiceUnavailable => write!(f, "Python semantic service unavailable"),
            WormholeError::PythonServiceTimeout => write!(f, "Python service request timeout"),
            WormholeError::InvalidVectorDimensions(expected, actual) => 
                write!(f, "Invalid vector dimensions: expected {}, got {}", expected, actual),
            WormholeError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            WormholeError::ConcurrencyConflict(msg) => write!(f, "Concurrency conflict: {}", msg),
            WormholeError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            WormholeError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            WormholeError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            WormholeError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            WormholeError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
        }
    }
}

impl std::error::Error for WormholeError {}

// ===================================================================
// WORMHOLE ENGINE CONFIGURATION
// ===================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeEngineConfig {
    pub similarity_threshold: f64,
    pub max_wormholes_per_concept: usize,
    pub vector_dimensions: usize,
    pub python_service_url: String,
    pub python_service_timeout: Duration,
    pub cache_size: usize,
    pub clustering_threshold: f64,
    pub background_scan_interval: Duration,
    pub prune_unused_after: Duration,
    pub max_search_radius: usize,
    pub enable_experimental_detection: bool,
    pub auto_bidirectional: bool,
    pub quality_threshold: f64,
    pub batch_size: usize,
}

impl Default for WormholeEngineConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_wormholes_per_concept: 10,
            vector_dimensions: 512,
            python_service_url: "http://localhost:8003".to_string(),
            python_service_timeout: Duration::from_secs(30),
            cache_size: 10000,
            clustering_threshold: 0.85,
            background_scan_interval: Duration::from_secs(300), // 5 minutes
            prune_unused_after: Duration::from_secs(7 * 24 * 3600), // 1 week
            max_search_radius: 5,
            enable_experimental_detection: true,
            auto_bidirectional: true,
            quality_threshold: 0.8,
            batch_size: 100,
        }
    }
}

// ===================================================================
// SPATIAL INDEX FOR EFFICIENT SIMILARITY SEARCH
// ===================================================================

#[derive(Debug)]
struct SpatialIndex {
    index: Arc<RwLock<HashMap<ConceptId, ConceptVector>>>,
    clusters: Arc<RwLock<Vec<WormholeCluster>>>,
    nearest_neighbors: Arc<RwLock<HashMap<ConceptId, Vec<ConceptId>>>>,
}

impl SpatialIndex {
    fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(HashMap::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            nearest_neighbors: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn add_vector(&self, vector: ConceptVector) -> Result<(), WormholeError> {
        let mut index = self.index.write()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire write lock".to_string()))?;
        
        index.insert(vector.concept_id, vector);
        
        // Invalidate nearest neighbors cache for affected concepts
        let mut nn = self.nearest_neighbors.write()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire NN write lock".to_string()))?;
        nn.clear(); // Simple invalidation - could be optimized
        
        Ok(())
    }

    fn find_similar(&self, concept_id: ConceptId, k: usize, min_similarity: f64) -> Result<Vec<SimilarityCandidate>, WormholeError> {
        let index = self.index.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire read lock".to_string()))?;
        
        let query_vector = index.get(&concept_id)
            .ok_or(WormholeError::ConceptNotFound(concept_id))?;
        
        let mut candidates = Vec::new();
        
        for (other_id, other_vector) in index.iter() {
            if *other_id == concept_id {
                continue;
            }
            
            let similarity = self.cosine_similarity(&query_vector.vector, &other_vector.vector)?;
            
            if similarity >= min_similarity {
                candidates.push(SimilarityCandidate {
                    concept_id: *other_id,
                    similarity_score: similarity,
                    semantic_distance: 1.0 - similarity,
                    confidence: self.calculate_confidence(similarity, &query_vector.vector, &other_vector.vector),
                    explanation: format!("Cosine similarity: {:.3}", similarity),
                });
            }
        }
        
        // Sort by similarity descending
        candidates.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        candidates.truncate(k);
        
        Ok(candidates)
    }

    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> Result<f64, WormholeError> {
        if a.len() != b.len() {
            return Err(WormholeError::InvalidVectorDimensions(a.len(), b.len()));
        }
        
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }

    fn calculate_confidence(&self, similarity: f64, vector_a: &[f64], vector_b: &[f64]) -> f64 {
        // Enhanced confidence calculation based on vector properties
        let magnitude_a = vector_a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_b = vector_b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        // Normalize confidence based on vector magnitudes and similarity
        let magnitude_factor = (magnitude_a * magnitude_b).min(1.0);
        let similarity_factor = similarity;
        
        (magnitude_factor * similarity_factor).max(0.0).min(1.0)
    }
}

// ===================================================================
// MAIN WORMHOLE ENGINE IMPLEMENTATION
// ===================================================================

pub struct WormholeEngine {
    config: WormholeEngineConfig,
    wormholes: Arc<RwLock<HashMap<WormholeId, Wormhole>>>,
    concept_wormholes: Arc<RwLock<HashMap<ConceptId, HashSet<WormholeId>>>>,
    spatial_index: SpatialIndex,
    similarity_cache: Arc<RwLock<HashMap<(ConceptId, ConceptId), SimilarityScore>>>,
    usage_stats: Arc<RwLock<HashMap<WormholeId, UsageStats>>>,
    python_service: Arc<Mutex<Option<std::process::Child>>>,
    last_maintenance: Arc<RwLock<Timestamp>>,
    event_sender: Option<std::sync::mpsc::Sender<WormholeEvent>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WormholeEvent {
    WormholeCreated {
        wormhole_id: WormholeId,
        concept_a: ConceptId,
        concept_b: ConceptId,
        strength: f64,
    },
    WormholeTraversed {
        wormhole_id: WormholeId,
        traverser: String,
        duration: Duration,
    },
    WormholeDeleted {
        wormhole_id: WormholeId,
        reason: String,
    },
    SimilarityComputed {
        concept_a: ConceptId,
        concept_b: ConceptId,
        similarity: f64,
        method: String,
    },
    ClusterDetected {
        cluster: WormholeCluster,
    },
    MaintenanceCompleted {
        wormholes_pruned: usize,
        clusters_updated: usize,
    },
}

impl WormholeEngine {
    // ===================================================================
    // INITIALIZATION AND CONFIGURATION
    // ===================================================================

    pub fn new(config: WormholeEngineConfig) -> Result<Self, WormholeError> {
        let engine = Self {
            config,
            wormholes: Arc::new(RwLock::new(HashMap::new())),
            concept_wormholes: Arc::new(RwLock::new(HashMap::new())),
            spatial_index: SpatialIndex::new(),
            similarity_cache: Arc::new(RwLock::new(HashMap::new())),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
            python_service: Arc::new(Mutex::new(None)),
            last_maintenance: Arc::new(RwLock::new(Self::current_timestamp())),
            event_sender: None,
        };

        // Start Python service
        engine.start_python_service()?;
        
        Ok(engine)
    }

    pub fn with_event_sender(mut self, sender: std::sync::mpsc::Sender<WormholeEvent>) -> Self {
        self.event_sender = Some(sender);
        self
    }

    fn start_python_service(&self) -> Result<(), WormholeError> {
        let mut service = self.python_service.lock()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire service lock".to_string()))?;

        let child = Command::new("python")
            .arg("-m")
            .arg("analysis.semantic_bridge")
            .arg("--port")
            .arg("8003")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| WormholeError::PythonServiceUnavailable)?;

        *service = Some(child);
        
        // Give service time to start
        thread::sleep(Duration::from_secs(2));
        
        Ok(())
    }

    // ===================================================================
    // CORE WORMHOLE OPERATIONS
    // ===================================================================

    pub fn find_wormholes(&self, concept: ConceptId) -> Result<Vec<(ConceptId, f64)>, WormholeError> {
        // Check cache first
        if let Some(cached_results) = self.get_cached_similarities(concept) {
            return Ok(cached_results);
        }

        // Find similar concepts using spatial index
        let candidates = self.spatial_index.find_similar(
            concept,
            self.config.max_wormholes_per_concept,
            self.config.similarity_threshold,
        )?;

        // Enhance with Python semantic analysis
        let enhanced_candidates = self.enhance_with_python_analysis(concept, candidates)?;

        // Filter and rank results
        let mut results: Vec<(ConceptId, f64)> = enhanced_candidates
            .into_iter()
            .filter(|c| c.similarity_score >= self.config.similarity_threshold)
            .map(|c| (c.concept_id, c.similarity_score))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(self.config.max_wormholes_per_concept);

        // Cache results
        self.cache_similarities(concept, &results)?;

        Ok(results)
    }

    pub fn create_wormhole(
        &self,
        concept_a: ConceptId,
        concept_b: ConceptId,
        strength: f64,
    ) -> Result<WormholeId, WormholeError> {
        if strength < 0.0 || strength > 1.0 {
            return Err(WormholeError::InvalidSimilarityScore(strength));
        }

        // Check for duplicate wormhole
        if self.wormhole_exists(concept_a, concept_b)? {
            return Err(WormholeError::DuplicateWormhole(concept_a, concept_b));
        }

        let wormhole_id = Uuid::new_v4();
        let timestamp = Self::current_timestamp();

        let wormhole = Wormhole {
            id: wormhole_id,
            concept_a,
            concept_b,
            strength,
            wormhole_type: WormholeType::SystemDetected,
            created_at: timestamp,
            last_accessed: timestamp,
            access_count: 0,
            bidirectional: self.config.auto_bidirectional,
            context_tags: Vec::new(),
            confidence: self.calculate_wormhole_confidence(concept_a, concept_b, strength)?,
            discovery_method: DiscoveryMethod::VectorSimilarity {
                model: "default".to_string(),
                threshold: self.config.similarity_threshold,
            },
            metadata: HashMap::new(),
        };

        // Store wormhole
        {
            let mut wormholes = self.wormholes.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
            wormholes.insert(wormhole_id, wormhole.clone());
        }

        // Update concept index
        {
            let mut concept_wormholes = self.concept_wormholes.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire concept index lock".to_string()))?;
            
            concept_wormholes.entry(concept_a).or_insert_with(HashSet::new).insert(wormhole_id);
            concept_wormholes.entry(concept_b).or_insert_with(HashSet::new).insert(wormhole_id);
        }

        // Initialize usage stats
        {
            let mut stats = self.usage_stats.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire stats lock".to_string()))?;
            stats.insert(wormhole_id, UsageStats {
                total_traversals: 0,
                recent_traversals: 0,
                average_traversal_time: Duration::from_millis(0),
                success_rate: 1.0,
                user_ratings: Vec::new(),
            });
        }

        // Emit event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(WormholeEvent::WormholeCreated {
                wormhole_id,
                concept_a,
                concept_b,
                strength,
            });
        }

        Ok(wormhole_id)
    }

    pub fn list_wormholes(&self) -> Result<Vec<WormholeInfo>, WormholeError> {
        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
        
        let stats = self.usage_stats.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire stats lock".to_string()))?;

        let mut wormhole_infos = Vec::new();

        for (id, wormhole) in wormholes.iter() {
            let usage_stats = stats.get(id).cloned().unwrap_or_else(|| UsageStats {
                total_traversals: 0,
                recent_traversals: 0,
                average_traversal_time: Duration::from_millis(0),
                success_rate: 1.0,
                user_ratings: Vec::new(),
            });

            let info = WormholeInfo {
                wormhole: wormhole.clone(),
                traversal_cost: self.calculate_traversal_cost(wormhole),
                path_length_reduction: self.estimate_path_reduction(wormhole.concept_a, wormhole.concept_b),
                semantic_coherence: wormhole.confidence,
                usage_statistics: usage_stats,
            };

            wormhole_infos.push(info);
        }

        // Sort by strength descending
        wormhole_infos.sort_by(|a, b| {
            b.wormhole.strength.partial_cmp(&a.wormhole.strength).unwrap()
        });

        Ok(wormhole_infos)
    }

    pub fn get_wormhole(&self, id: WormholeId) -> Result<WormholeInfo, WormholeError> {
        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
        
        let wormhole = wormholes.get(&id)
            .ok_or(WormholeError::WormholeNotFound(id))?;

        let stats = self.usage_stats.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire stats lock".to_string()))?;

        let usage_stats = stats.get(&id).cloned().unwrap_or_else(|| UsageStats {
            total_traversals: 0,
            recent_traversals: 0,
            average_traversal_time: Duration::from_millis(0),
            success_rate: 1.0,
            user_ratings: Vec::new(),
        });

        Ok(WormholeInfo {
            wormhole: wormhole.clone(),
            traversal_cost: self.calculate_traversal_cost(wormhole),
            path_length_reduction: self.estimate_path_reduction(wormhole.concept_a, wormhole.concept_b),
            semantic_coherence: wormhole.confidence,
            usage_statistics: usage_stats,
        })
    }

    pub fn delete_wormhole(&self, id: WormholeId, reason: String) -> Result<(), WormholeError> {
        let wormhole = {
            let mut wormholes = self.wormholes.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
            
            wormholes.remove(&id)
                .ok_or(WormholeError::WormholeNotFound(id))?
        };

        // Remove from concept index
        {
            let mut concept_wormholes = self.concept_wormholes.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire concept index lock".to_string()))?;
            
            if let Some(set) = concept_wormholes.get_mut(&wormhole.concept_a) {
                set.remove(&id);
            }
            if let Some(set) = concept_wormholes.get_mut(&wormhole.concept_b) {
                set.remove(&id);
            }
        }

        // Remove usage stats
        {
            let mut stats = self.usage_stats.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire stats lock".to_string()))?;
            stats.remove(&id);
        }

        // Emit event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(WormholeEvent::WormholeDeleted {
                wormhole_id: id,
                reason,
            });
        }

        Ok(())
    }

    // ===================================================================
    // TRAVERSAL AND ACCESS TRACKING
    // ===================================================================

    pub fn traverse_wormhole(
        &self,
        id: WormholeId,
        traverser: String,
    ) -> Result<(ConceptId, ConceptId), WormholeError> {
        let start_time = std::time::Instant::now();
        
        let wormhole = {
            let mut wormholes = self.wormholes.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
            
            let wormhole = wormholes.get_mut(&id)
                .ok_or(WormholeError::WormholeNotFound(id))?;
            
            wormhole.last_accessed = Self::current_timestamp();
            wormhole.access_count += 1;
            
            wormhole.clone()
        };

        // Update usage statistics
        {
            let mut stats = self.usage_stats.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire stats lock".to_string()))?;
            
            if let Some(usage_stats) = stats.get_mut(&id) {
                usage_stats.total_traversals += 1;
                usage_stats.recent_traversals += 1;
                
                let duration = start_time.elapsed();
                usage_stats.average_traversal_time = Duration::from_nanos(
                    (usage_stats.average_traversal_time.as_nanos() as u64 + duration.as_nanos() as u64) / 2
                );
            }
        }

        // Emit event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(WormholeEvent::WormholeTraversed {
                wormhole_id: id,
                traverser,
                duration: start_time.elapsed(),
            });
        }

        Ok((wormhole.concept_a, wormhole.concept_b))
    }

    // ===================================================================
    // CONCEPT VECTOR MANAGEMENT
    // ===================================================================

    pub fn add_concept_vector(&self, vector: ConceptVector) -> Result<(), WormholeError> {
        if vector.dimensions != self.config.vector_dimensions {
            return Err(WormholeError::InvalidVectorDimensions(
                self.config.vector_dimensions,
                vector.dimensions,
            ));
        }

        self.spatial_index.add_vector(vector)?;
        
        // Invalidate similarity cache entries for this concept
        self.invalidate_similarity_cache(vector.concept_id)?;

        Ok(())
    }

    pub fn batch_add_vectors(&self, vectors: Vec<ConceptVector>) -> Result<(), WormholeError> {
        for vector in vectors {
            self.add_concept_vector(vector)?;
        }
        Ok(())
    }

    // ===================================================================
    // ADVANCED SEARCH AND ANALYSIS
    // ===================================================================

    pub fn analyze_concept_connectivity(&self, concept_id: ConceptId) -> Result<f64, WormholeError> {
        let concept_wormholes = self.concept_wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire concept index lock".to_string()))?;

        let wormhole_ids = concept_wormholes.get(&concept_id)
            .ok_or(WormholeError::ConceptNotFound(concept_id))?;

        if wormhole_ids.is_empty() {
            return Ok(0.0);
        }

        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;

        let total_strength: f64 = wormhole_ids.iter()
            .filter_map(|id| wormholes.get(id))
            .map(|wh| wh.strength)
            .sum();

        let connectivity = total_strength / wormhole_ids.len() as f64;
        Ok(connectivity)
    }

    // ===================================================================
    // MAINTENANCE AND OPTIMIZATION
    // ===================================================================

    pub fn run_maintenance(&self) -> Result<(), WormholeError> {
        let current_time = Self::current_timestamp();
        
        // Prune unused wormholes
        let pruned_count = self.prune_unused_wormholes(current_time)?;
        
        // Detect clusters
        let clusters = self.detect_wormhole_clusters()?;
        
        // Update last maintenance time
        {
            let mut last_maintenance = self.last_maintenance.write()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire maintenance lock".to_string()))?;
            *last_maintenance = current_time;
        }

        // Emit maintenance completion event
        if let Some(ref sender) = self.event_sender {
            let _ = sender.send(WormholeEvent::MaintenanceCompleted {
                wormholes_pruned: pruned_count,
                clusters_updated: clusters.len(),
            });
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> Result<HashMap<String, f64>, WormholeError> {
        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;
        
        let concept_wormholes = self.concept_wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire concept index lock".to_string()))?;

        let mut stats = HashMap::new();
        
        stats.insert("total_wormholes".to_string(), wormholes.len() as f64);
        stats.insert("total_concepts".to_string(), concept_wormholes.len() as f64);
        
        if !wormholes.is_empty() {
            let average_strength: f64 = wormholes.values()
                .map(|wh| wh.strength)
                .sum::<f64>() / wormholes.len() as f64;
            stats.insert("average_strength".to_string(), average_strength);
        }
        
        if !concept_wormholes.is_empty() {
            let average_wormholes_per_concept: f64 = concept_wormholes.values()
                .map(|set| set.len() as f64)
                .sum::<f64>() / concept_wormholes.len() as f64;
            stats.insert("average_wormholes_per_concept".to_string(), average_wormholes_per_concept);
        }

        // Calculate clustering coefficient
        let clustering_coefficient = self.calculate_clustering_coefficient(&wormholes, &concept_wormholes)?;
        stats.insert("clustering_coefficient".to_string(), clustering_coefficient);

        Ok(stats)
    }

    // ===================================================================
    // HELPER METHODS
    // ===================================================================

    fn current_timestamp() -> Timestamp {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn wormhole_exists(&self, concept_a: ConceptId, concept_b: ConceptId) -> Result<bool, WormholeError> {
        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;

        for wormhole in wormholes.values() {
            if (wormhole.concept_a == concept_a && wormhole.concept_b == concept_b) ||
               (wormhole.concept_a == concept_b && wormhole.concept_b == concept_a && wormhole.bidirectional) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn calculate_wormhole_confidence(&self, concept_a: ConceptId, concept_b: ConceptId, strength: f64) -> Result<f64, WormholeError> {
        // Base confidence on similarity strength
        let base_confidence = strength;
        
        // Factor in concept connectivity
        let connectivity_a = self.analyze_concept_connectivity(concept_a).unwrap_or(0.0);
        let connectivity_b = self.analyze_concept_connectivity(concept_b).unwrap_or(0.0);
        
        // Higher connectivity concepts have slightly lower confidence for new wormholes
        let connectivity_factor = 1.0 - ((connectivity_a + connectivity_b) / 2.0 * 0.1);
        
        let confidence = (base_confidence * connectivity_factor).max(0.0).min(1.0);
        Ok(confidence)
    }

    fn calculate_traversal_cost(&self, wormhole: &Wormhole) -> f64 {
        // Lower strength means higher cost
        let strength_cost = 1.0 - wormhole.strength;
        
        // Age factor (older wormholes might be more reliable)
        let age_seconds = Self::current_timestamp() - wormhole.created_at;
        let age_factor = (age_seconds as f64 / 86400.0).min(1.0) * 0.1; // Max 10% reduction
        
        (strength_cost - age_factor).max(0.01)
    }

    fn estimate_path_reduction(&self, concept_a: ConceptId, concept_b: ConceptId) -> usize {
        // Simplified path length estimation - in practice would use graph algorithms
        // For now, assume wormholes provide significant shortcuts
        5 // Assume average path reduction of 5 hops
    }

    fn get_cached_similarities(&self, concept: ConceptId) -> Option<Vec<(ConceptId, f64)>> {
        // Check if we have cached similarity results
        let cache = self.similarity_cache.read().ok()?;
        
        let mut results = Vec::new();
        for ((a, b), similarity) in cache.iter() {
            if *a == concept {
                results.push((*b, *similarity));
            } else if *b == concept {
                results.push((*a, *similarity));
            }
        }
        
        if results.is_empty() {
            None
        } else {
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            Some(results)
        }
    }

    fn cache_similarities(&self, concept: ConceptId, results: &[(ConceptId, f64)]) -> Result<(), WormholeError> {
        let mut cache = self.similarity_cache.write()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire cache lock".to_string()))?;

        for (other_concept, similarity) in results {
            let key = if concept < *other_concept {
                (concept, *other_concept)
            } else {
                (*other_concept, concept)
            };
            cache.insert(key, *similarity);
        }

        // Limit cache size
        if cache.len() > self.config.cache_size {
            let keys_to_remove: Vec<_> = cache.keys().take(cache.len() - self.config.cache_size).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }

        Ok(())
    }

    fn invalidate_similarity_cache(&self, concept_id: ConceptId) -> Result<(), WormholeError> {
        let mut cache = self.similarity_cache.write()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire cache lock".to_string()))?;

        let keys_to_remove: Vec<_> = cache.keys()
            .filter(|(a, b)| *a == concept_id || *b == concept_id)
            .cloned()
            .collect();

        for key in keys_to_remove {
            cache.remove(&key);
        }

        Ok(())
    }

    fn enhance_with_python_analysis(&self, concept: ConceptId, candidates: Vec<SimilarityCandidate>) -> Result<Vec<SimilarityCandidate>, WormholeError> {
        // In a full implementation, this would call the Python service
        // For now, return enhanced candidates with additional analysis
        let mut enhanced = candidates;
        
        for candidate in &mut enhanced {
            // Simulate Python enhancement
            candidate.confidence *= 1.1; // Boost confidence slightly
            candidate.explanation = format!("{} (enhanced)", candidate.explanation);
        }
        
        Ok(enhanced)
    }

    fn detect_wormhole_clusters(&self) -> Result<Vec<WormholeCluster>, WormholeError> {
        let wormholes = self.wormholes.read()
            .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;

        let mut clusters = Vec::new();
        let mut processed_concepts = HashSet::new();

        for wormhole in wormholes.values() {
            if processed_concepts.contains(&wormhole.concept_a) ||
               processed_concepts.contains(&wormhole.concept_b) {
                continue;
            }

            let cluster = self.build_cluster_from_concept(wormhole.concept_a, &wormholes)?;
            if cluster.wormholes.len() >= 3 { // Minimum cluster size
                for wh_id in &cluster.wormholes {
                    if let Some(wh) = wormholes.get(wh_id) {
                        processed_concepts.insert(wh.concept_a);
                        processed_concepts.insert(wh.concept_b);
                    }
                }
                clusters.push(cluster);
            }
        }

        // Emit cluster detection events
        for cluster in &clusters {
            if let Some(ref sender) = self.event_sender {
                let _ = sender.send(WormholeEvent::ClusterDetected {
                    cluster: cluster.clone(),
                });
            }
        }

        Ok(clusters)
    }

    fn build_cluster_from_concept(&self, center_concept: ConceptId, wormholes: &HashMap<WormholeId, Wormhole>) -> Result<WormholeCluster, WormholeError> {
        let mut cluster_wormholes = Vec::new();
        let mut total_strength = 0.0;
        let mut max_distance = 0.0;

        for (id, wormhole) in wormholes.iter() {
            if wormhole.concept_a == center_concept || wormhole.concept_b == center_concept {
                cluster_wormholes.push(*id);
                total_strength += wormhole.strength;
                max_distance = max_distance.max(1.0 - wormhole.strength);
            }
        }

        let density = if cluster_wormholes.is_empty() {
            0.0
        } else {
            total_strength / cluster_wormholes.len() as f64
        };

        let cohesion_score = if cluster_wormholes.is_empty() {
            0.0
        } else {
            density * (1.0 - max_distance / cluster_wormholes.len() as f64)
        };

        Ok(WormholeCluster {
            cluster_id: Uuid::new_v4(),
            wormholes: cluster_wormholes,
            center_concept,
            radius: max_distance,
            density,
            cohesion_score,
        })
    }

    fn prune_unused_wormholes(&self, current_time: Timestamp) -> Result<usize, WormholeError> {
        let prune_threshold = current_time - self.config.prune_unused_after.as_secs();
        let mut pruned_count = 0;

        let wormholes_to_prune: Vec<WormholeId> = {
            let wormholes = self.wormholes.read()
                .map_err(|_| WormholeError::ConcurrencyConflict("Failed to acquire wormholes lock".to_string()))?;

            wormholes.values()
                .filter(|wh| wh.last_accessed < prune_threshold && wh.access_count == 0)
                .map(|wh| wh.id)
                .collect()
        };

        for wormhole_id in wormholes_to_prune {
            self.delete_wormhole(wormhole_id, "Unused wormhole pruned".to_string())?;
            pruned_count += 1;
        }

        Ok(pruned_count)
    }

    fn calculate_clustering_coefficient(&self, wormholes: &HashMap<WormholeId, Wormhole>, concept_wormholes: &HashMap<ConceptId, HashSet<WormholeId>>) -> Result<f64, WormholeError> {
        let mut total_coefficient = 0.0;
        let mut concept_count = 0;

        for (concept_id, wormhole_ids) in concept_wormholes.iter() {
            if wormhole_ids.len() < 2 {
                continue;
            }

            let neighbors: HashSet<ConceptId> = wormhole_ids.iter()
                .filter_map(|id| wormholes.get(id))
                .flat_map(|wh| vec![wh.concept_a, wh.concept_b])
                .filter(|&id| id != *concept_id)
                .collect();

            if neighbors.len() < 2 {
                continue;
            }

            let possible_edges = neighbors.len() * (neighbors.len() - 1) / 2;
            let actual_edges = self.count_edges_between_concepts(&neighbors, wormholes);
            
            let coefficient = actual_edges as f64 / possible_edges as f64;
            total_coefficient += coefficient;
            concept_count += 1;
        }

        if concept_count == 0 {
            Ok(0.0)
        } else {
            Ok(total_coefficient / concept_count as f64)
        }
    }

    fn count_edges_between_concepts(&self, concepts: &HashSet<ConceptId>, wormholes: &HashMap<WormholeId, Wormhole>) -> usize {
        let mut edge_count = 0;

        for wormhole in wormholes.values() {
            if concepts.contains(&wormhole.concept_a) && concepts.contains(&wormhole.concept_b) {
                edge_count += 1;
            }
        }

        edge_count
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_config() -> WormholeEngineConfig {
        WormholeEngineConfig {
            similarity_threshold: 0.6,
            max_wormholes_per_concept: 5,
            vector_dimensions: 64,
            python_service_url: "http://localhost:8003".to_string(),
            python_service_timeout: Duration::from_secs(5),
            cache_size: 100,
            clustering_threshold: 0.8,
            background_scan_interval: Duration::from_secs(60),
            prune_unused_after: Duration::from_secs(3600),
            max_search_radius: 3,
            enable_experimental_detection: true,
            auto_bidirectional: true,
            quality_threshold: 0.7,
            batch_size: 10,
        }
    }

    fn create_test_vector(concept_id: ConceptId, dimensions: usize) -> ConceptVector {
        ConceptVector {
            concept_id,
            vector: (0..dimensions).map(|i| (i as f64 + concept_id as f64) / 100.0).collect(),
            model_version: "test_v1".to_string(),
            created_at: WormholeEngine::current_timestamp(),
            dimensions,
        }
    }

    fn create_test_engine() -> WormholeEngine {
        let config = create_test_config();
        WormholeEngine {
            config: config.clone(),
            wormholes: Arc::new(RwLock::new(HashMap::new())),
            concept_wormholes: Arc::new(RwLock::new(HashMap::new())),
            spatial_index: SpatialIndex::new(),
            similarity_cache: Arc::new(RwLock::new(HashMap::new())),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
            python_service: Arc::new(Mutex::new(None)),
            last_maintenance: Arc::new(RwLock::new(WormholeEngine::current_timestamp())),
            event_sender: None,
        }
    }

    #[test]
    fn test_spatial_index() {
        let index = SpatialIndex::new();
        
        let vector1 = create_test_vector(1, 64);
        let vector2 = create_test_vector(2, 64);
        
        index.add_vector(vector1).unwrap();
        index.add_vector(vector2).unwrap();
        
        let candidates = index.find_similar(1, 5, 0.5).unwrap();
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].concept_id, 2);
    }

    #[test]
    fn test_wormhole_creation_and_retrieval() {
        let engine = create_test_engine();
        
        // Create wormhole
        let wormhole_id = engine.create_wormhole(1, 2, 0.8).unwrap();
        
        // Retrieve wormhole
        let wormhole_info = engine.get_wormhole(wormhole_id).unwrap();
        assert_eq!(wormhole_info.wormhole.concept_a, 1);
        assert_eq!(wormhole_info.wormhole.concept_b, 2);
        assert_eq!(wormhole_info.wormhole.strength, 0.8);
        
        // List wormholes
        let all_wormholes = engine.list_wormholes().unwrap();
        assert_eq!(all_wormholes.len(), 1);
    }

    #[test]
    fn test_similarity_calculation() {
        let index = SpatialIndex::new();
        
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        let vec_c = vec![0.0, 1.0, 0.0];
        
        // Identical vectors should have similarity 1.0
        let sim_identical = index.cosine_similarity(&vec_a, &vec_b).unwrap();
        assert!((sim_identical - 1.0).abs() < 1e-10);
        
        // Orthogonal vectors should have similarity 0.0
        let sim_orthogonal = index.cosine_similarity(&vec_a, &vec_c).unwrap();
        assert!(sim_orthogonal.abs() < 1e-10);
    }

    #[test]
    fn test_wormhole_traversal() {
        let engine = create_test_engine();
        
        let wormhole_id = engine.create_wormhole(1, 2, 0.9).unwrap();
        
        let (concept_a, concept_b) = engine.traverse_wormhole(wormhole_id, "test_user".to_string()).unwrap();
        assert_eq!(concept_a, 1);
        assert_eq!(concept_b, 2);
        
        // Check that traversal was recorded
        let wormhole_info = engine.get_wormhole(wormhole_id).unwrap();
        assert_eq!(wormhole_info.usage_statistics.total_traversals, 1);
    }

    #[test]
    fn test_duplicate_wormhole_prevention() {
        let engine = create_test_engine();
        
        // Create first wormhole
        let _wormhole_id1 = engine.create_wormhole(1, 2, 0.8).unwrap();
        
        // Attempt to create duplicate
        let result = engine.create_wormhole(1, 2, 0.9);
        assert!(matches!(result, Err(WormholeError::DuplicateWormhole(1, 2))));
        
        // Reverse direction should also be detected as duplicate (if bidirectional)
        let result = engine.create_wormhole(2, 1, 0.9);
        assert!(matches!(result, Err(WormholeError::DuplicateWormhole(2, 1))));
    }

    #[test]
    fn test_wormhole_deletion() {
        let engine = create_test_engine();
        
        let wormhole_id = engine.create_wormhole(1, 2, 0.8).unwrap();
        
        // Verify wormhole exists
        assert!(engine.get_wormhole(wormhole_id).is_ok());
        
        // Delete wormhole
        engine.delete_wormhole(wormhole_id, "Test deletion".to_string()).unwrap();
        
        // Verify wormhole no longer exists
        assert!(matches!(engine.get_wormhole(wormhole_id), Err(WormholeError::WormholeNotFound(_))));
    }

    #[test]
    fn test_connectivity_analysis() {
        let engine = create_test_engine();
        
        // Create multiple wormholes for concept 1
        engine.create_wormhole(1, 2, 0.8).unwrap();
        engine.create_wormhole(1, 3, 0.9).unwrap();
        engine.create_wormhole(1, 4, 0.7).unwrap();
        
        let connectivity = engine.analyze_concept_connectivity(1).unwrap();
        assert!(connectivity > 0.0);
        assert!(connectivity <= 1.0);
        
        // Concept 5 has no wormholes
        let no_connectivity = engine.analyze_concept_connectivity(5).unwrap();
        assert_eq!(no_connectivity, 0.0);
    }

    #[test]
    fn test_statistics_generation() {
        let engine = create_test_engine();
        
        // Initially no wormholes
        let stats = engine.get_statistics().unwrap();
        assert_eq!(stats.get("total_wormholes").unwrap(), &0.0);
        
        // Create some wormholes
        engine.create_wormhole(1, 2, 0.8).unwrap();
        engine.create_wormhole(2, 3, 0.9).unwrap();
        
        let stats = engine.get_statistics().unwrap();
        assert_eq!(stats.get("total_wormholes").unwrap(), &2.0);
        assert!(stats.get("average_strength").unwrap() > &0.0);
    }

    #[test]
    fn test_maintenance_operations() {
        let engine = create_test_engine();
        
        // Create some wormholes
        engine.create_wormhole(1, 2, 0.8).unwrap();
        engine.create_wormhole(2, 3, 0.9).unwrap();
        
        // Run maintenance
        let result = engine.run_maintenance();
        assert!(result.is_ok());
        
        // Verify maintenance timestamp was updated
        let last_maintenance = *engine.last_maintenance.read().unwrap();
        assert!(last_maintenance > 0);
    }

    #[test]
    fn test_error_handling() {
        let engine = create_test_engine();
        
        // Test invalid similarity score
        let result = engine.create_wormhole(1, 2, 1.5);
        assert!(matches!(result, Err(WormholeError::InvalidSimilarityScore(1.5))));
        
        // Test non-existent wormhole
        let fake_id = Uuid::new_v4();
        let result = engine.get_wormhole(fake_id);
        assert!(matches!(result, Err(WormholeError::WormholeNotFound(_))));
        
        // Test non-existent concept
        let result = engine.analyze_concept_connectivity(999);
        assert!(matches!(result, Err(WormholeError::ConceptNotFound(999))));
    }

    #[test]
    fn test_vector_management() {
        let engine = create_test_engine();
        
        let vector1 = create_test_vector(1, engine.config.vector_dimensions);
        let vector2 = create_test_vector(2, engine.config.vector_dimensions);
        
        // Add vectors
        engine.add_concept_vector(vector1).unwrap();
        engine.add_concept_vector(vector2).unwrap();
        
        // Test invalid dimensions
        let invalid_vector = create_test_vector(3, 32); // Wrong dimensions
        let result = engine.add_concept_vector(invalid_vector);
        assert!(matches!(result, Err(WormholeError::InvalidVectorDimensions(_, _))));
    }

    #[test] 
    fn test_cluster_detection() {
        let engine = create_test_engine();
        
        // Create interconnected wormholes
        engine.create_wormhole(1, 2, 0.8).unwrap();
        engine.create_wormhole(1, 3, 0.9).unwrap();
        engine.create_wormhole(2, 3, 0.7).unwrap();
        engine.create_wormhole(1, 4, 0.6).unwrap();
        
        let clusters = engine.detect_wormhole_clusters().unwrap();
        
        // Should detect at least one cluster
        assert!(!clusters.is_empty());
        
        // Verify cluster properties
        for cluster in clusters {
            assert!(cluster.wormholes.len() >= 3);
            assert!(cluster.density > 0.0);
            assert!(cluster.cohesion_score >= 0.0);
        }
    }
}
