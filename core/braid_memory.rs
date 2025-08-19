// ===================================================================
// TORI BraidMemory - Core Implementation
// Mathematical Foundation: ∞-groupoid coherent memory threading with
// associative composition up to homotopy and temporal trace management
// ===================================================================

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

// ===================================================================
// CORE DATA TYPES AND STRUCTURES
// ===================================================================

/// Unique identifier for memory threads
pub type ThreadId = u64;

/// Unique identifier for braid points
pub type BraidId = u64;

/// Unique identifier for memory nodes within threads
pub type NodeId = u64;

/// Concept identifier (shared with MultiScaleHierarchy)
pub type ConceptId = u64;

/// Errors that can occur in braid memory operations
#[derive(Debug, Clone, PartialEq)]
pub enum BraidError {
    ThreadNotFound(ThreadId),
    BraidNotFound(BraidId),
    NodeNotFound(NodeId),
    ConceptNotFound(ConceptId),
    InvalidThread(String),
    CycleDetected(ThreadId, ThreadId),
    CoherenceViolation(String),
    ConcurrencyError(String),
    HomotopyError(String),
}

impl std::fmt::Display for BraidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BraidError::ThreadNotFound(id) => write!(f, "Thread {} not found", id),
            BraidError::BraidNotFound(id) => write!(f, "Braid {} not found", id),
            BraidError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            BraidError::ConceptNotFound(id) => write!(f, "Concept {} not found", id),
            BraidError::InvalidThread(msg) => write!(f, "Invalid thread: {}", msg),
            BraidError::CycleDetected(a, b) => write!(f, "Cycle detected between threads {} and {}", a, b),
            BraidError::CoherenceViolation(msg) => write!(f, "Coherence violation: {}", msg),
            BraidError::ConcurrencyError(msg) => write!(f, "Concurrency error: {}", msg),
            BraidError::HomotopyError(msg) => write!(f, "Homotopy error: {}", msg),
        }
    }
}

impl std::error::Error for BraidError {}

/// A memory node within a thread - represents a concept instance in temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Associated concept ID
    pub concept_id: ConceptId,
    /// Thread this node belongs to
    pub thread_id: ThreadId,
    /// Position within thread (sequence index)
    pub position: usize,
    /// Timestamp when this node was created
    pub timestamp: u64,
    /// Context or reasoning that led to this concept
    pub context: Option<String>,
    /// Strength or confidence of this memory
    pub strength: f64,
    /// Temporal causality links to other nodes
    pub causal_links: Vec<NodeId>,
    /// Metadata for cognitive processing
    pub metadata: HashMap<String, String>,
}

impl MemoryNode {
    pub fn new(
        id: NodeId,
        concept_id: ConceptId,
        thread_id: ThreadId,
        position: usize,
        context: Option<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            concept_id,
            thread_id,
            position,
            timestamp: now,
            context,
            strength: 1.0,
            causal_links: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a causal link to another memory node
    pub fn add_causal_link(&mut self, target_node: NodeId) {
        if !self.causal_links.contains(&target_node) {
            self.causal_links.push(target_node);
        }
    }

    /// Update strength based on access or reinforcement
    pub fn reinforce(&mut self, delta: f64) {
        self.strength = (self.strength + delta).max(0.0).min(10.0);
    }

    /// Check if this node is temporally before another
    pub fn is_before(&self, other: &MemoryNode) -> bool {
        if self.thread_id == other.thread_id {
            self.position < other.position
        } else {
            self.timestamp < other.timestamp
        }
    }
}

/// A memory thread - ordered sequence of concept instances representing a narrative or reasoning chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryThread {
    /// Unique thread identifier
    pub id: ThreadId,
    /// Human-readable title or description
    pub title: String,
    /// Ordered sequence of memory nodes
    pub nodes: Vec<NodeId>,
    /// Thread creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
    /// Thread activity level (0.0 to 1.0)
    pub activity_level: f64,
    /// Domain or context classification
    pub domain: Option<String>,
    /// Parent thread (for hierarchical threading)
    pub parent_thread: Option<ThreadId>,
    /// Child threads
    pub child_threads: HashSet<ThreadId>,
    /// Braid connections at specific positions
    pub braid_points: HashMap<usize, BraidId>,
    /// Thread-specific metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryThread {
    pub fn new(id: ThreadId, title: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            title,
            nodes: Vec::new(),
            created_at: now,
            modified_at: now,
            activity_level: 1.0,
            domain: None,
            parent_thread: None,
            child_threads: HashSet::new(),
            braid_points: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the end of the thread
    pub fn append_node(&mut self, node_id: NodeId) {
        self.nodes.push(node_id);
        self.touch();
    }

    /// Insert a node at a specific position
    pub fn insert_node(&mut self, position: usize, node_id: NodeId) -> Result<(), BraidError> {
        if position > self.nodes.len() {
            return Err(BraidError::InvalidThread(
                format!("Position {} exceeds thread length {}", position, self.nodes.len())
            ));
        }
        
        self.nodes.insert(position, node_id);
        self.touch();
        Ok(())
    }

    /// Remove a node from the thread
    pub fn remove_node(&mut self, node_id: NodeId) -> bool {
        if let Some(pos) = self.nodes.iter().position(|&id| id == node_id) {
            self.nodes.remove(pos);
            self.touch();
            true
        } else {
            false
        }
    }

    /// Get the length of the thread
    pub fn length(&self) -> usize {
        self.nodes.len()
    }

    /// Check if thread is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Update modification timestamp
    pub fn touch(&mut self) {
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Add a braid point at a specific position
    pub fn add_braid_point(&mut self, position: usize, braid_id: BraidId) {
        self.braid_points.insert(position, braid_id);
        self.touch();
    }

    /// Remove a braid point
    pub fn remove_braid_point(&mut self, position: usize) -> Option<BraidId> {
        let result = self.braid_points.remove(&position);
        if result.is_some() {
            self.touch();
        }
        result
    }
}

/// A braid connection - represents a point where multiple threads intersect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidConnection {
    /// Unique braid identifier
    pub id: BraidId,
    /// Threads involved in this braid
    pub threads: Vec<ThreadId>,
    /// The concept that forms the intersection
    pub concept_id: ConceptId,
    /// Positions within each thread where the braid occurs
    pub positions: HashMap<ThreadId, usize>,
    /// Braid creation timestamp
    pub created_at: u64,
    /// Strength of the braid connection
    pub strength: f64,
    /// Type of braid (semantic, temporal, causal, etc.)
    pub braid_type: BraidType,
    /// Homotopy equivalence class (for ∞-groupoid coherence)
    pub homotopy_class: Option<String>,
    /// Metadata for this braid
    pub metadata: HashMap<String, String>,
}

impl BraidConnection {
    pub fn new(
        id: BraidId,
        threads: Vec<ThreadId>,
        concept_id: ConceptId,
        positions: HashMap<ThreadId, usize>,
        braid_type: BraidType,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            threads,
            concept_id,
            positions,
            created_at: now,
            strength: 1.0,
            braid_type,
            homotopy_class: None,
            metadata: HashMap::new(),
        }
    }

    /// Check if this braid involves a specific thread
    pub fn involves_thread(&self, thread_id: ThreadId) -> bool {
        self.threads.contains(&thread_id)
    }

    /// Get the position for a specific thread
    pub fn get_position(&self, thread_id: ThreadId) -> Option<usize> {
        self.positions.get(&thread_id).copied()
    }

    /// Reinforce the braid connection
    pub fn reinforce(&mut self, delta: f64) {
        self.strength = (self.strength + delta).max(0.0).min(10.0);
    }
}

/// Types of braid connections
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BraidType {
    /// Semantic similarity between concepts
    Semantic,
    /// Temporal co-occurrence
    Temporal,
    /// Causal relationship
    Causal,
    /// Logical implication
    Logical,
    /// User-defined association
    UserDefined,
    /// System-detected pattern
    SystemDetected,
}

/// Homotopy relationship between braids (for ∞-groupoid coherence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomotopyRelation {
    /// Source braid
    pub source_braid: BraidId,
    /// Target braid
    pub target_braid: BraidId,
    /// Homotopy transformation description
    pub transformation: String,
    /// Confidence in this homotopy
    pub confidence: f64,
    /// Creation timestamp
    pub created_at: u64,
}

/// Statistics about the braid memory system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BraidMemoryStats {
    pub total_threads: usize,
    pub total_nodes: usize,
    pub total_braids: usize,
    pub total_homotopies: usize,
    pub active_threads: usize,
    pub average_thread_length: f64,
    pub max_thread_length: usize,
    pub braid_density: f64,
    pub coherence_score: f64,
    pub last_updated: u64,
}

// ===================================================================
// MAIN BRAID MEMORY IMPLEMENTATION
// ===================================================================

/// The main BraidMemory structure managing all temporal memory operations
pub struct BraidMemory {
    /// All memory threads (thread-safe)
    threads: Arc<RwLock<HashMap<ThreadId, MemoryThread>>>,
    /// All memory nodes (thread-safe)
    nodes: Arc<RwLock<HashMap<NodeId, MemoryNode>>>,
    /// All braid connections (thread-safe)
    braids: Arc<RwLock<HashMap<BraidId, BraidConnection>>>,
    /// Homotopy relations between braids
    homotopies: Arc<RwLock<Vec<HomotopyRelation>>>,
    /// Next available IDs
    next_thread_id: Arc<RwLock<ThreadId>>,
    next_node_id: Arc<RwLock<NodeId>>,
    next_braid_id: Arc<RwLock<BraidId>>,
    /// System statistics
    stats: Arc<RwLock<BraidMemoryStats>>,
}

impl BraidMemory {
    /// Create a new empty BraidMemory system
    pub fn new() -> Self {
        Self {
            threads: Arc::new(RwLock::new(HashMap::new())),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            braids: Arc::new(RwLock::new(HashMap::new())),
            homotopies: Arc::new(RwLock::new(Vec::new())),
            next_thread_id: Arc::new(RwLock::new(1)),
            next_node_id: Arc::new(RwLock::new(1)),
            next_braid_id: Arc::new(RwLock::new(1)),
            stats: Arc::new(RwLock::new(BraidMemoryStats::default())),
        }
    }

    /// Generate a new unique thread ID
    fn generate_thread_id(&self) -> Result<ThreadId, BraidError> {
        let mut next_id = self.next_thread_id.write()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    /// Generate a new unique node ID
    fn generate_node_id(&self) -> Result<NodeId, BraidError> {
        let mut next_id = self.next_node_id.write()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    /// Generate a new unique braid ID
    fn generate_braid_id(&self) -> Result<BraidId, BraidError> {
        let mut next_id = self.next_braid_id.write()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    /// Start a new memory thread
    pub fn start_thread(&mut self, title: String) -> Result<ThreadId, BraidError> {
        let thread_id = self.generate_thread_id()?;
        let thread = MemoryThread::new(thread_id, title);

        {
            let mut threads = self.threads.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            threads.insert(thread_id, thread);
        }

        self.update_stats()?;
        Ok(thread_id)
    }

    /// Append a concept to a memory thread
    pub fn append_to_thread(
        &mut self,
        thread_id: ThreadId,
        concept_id: ConceptId,
        context: Option<String>,
    ) -> Result<NodeId, BraidError> {
        // Generate new node ID
        let node_id = self.generate_node_id()?;

        // Get thread position
        let position = {
            let threads = self.threads.read()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            let thread = threads.get(&thread_id)
                .ok_or(BraidError::ThreadNotFound(thread_id))?;
            thread.length()
        };

        // Create memory node
        let node = MemoryNode::new(node_id, concept_id, thread_id, position, context);

        // Add node to storage
        {
            let mut nodes = self.nodes.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            nodes.insert(node_id, node);
        }

        // Add node to thread
        {
            let mut threads = self.threads.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            let thread = threads.get_mut(&thread_id)
                .ok_or(BraidError::ThreadNotFound(thread_id))?;
            thread.append_node(node_id);
        }

        self.update_stats()?;
        Ok(node_id)
    }

    /// Braid multiple threads together at a shared concept
    pub fn braid_threads(
        &mut self,
        thread_ids: &[ThreadId],
        via_concept: ConceptId,
    ) -> Result<BraidId, BraidError> {
        if thread_ids.len() < 2 {
            return Err(BraidError::InvalidThread(
                "Need at least 2 threads for braiding".to_string()
            ));
        }

        // Validate all threads exist and find positions with the concept
        let mut positions = HashMap::new();
        {
            let threads = self.threads.read()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            let nodes = self.nodes.read()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

            for &thread_id in thread_ids {
                let thread = threads.get(&thread_id)
                    .ok_or(BraidError::ThreadNotFound(thread_id))?;

                // Find a node in this thread with the target concept
                let mut found_position = None;
                for (pos, &node_id) in thread.nodes.iter().enumerate() {
                    if let Some(node) = nodes.get(&node_id) {
                        if node.concept_id == via_concept {
                            found_position = Some(pos);
                            break;
                        }
                    }
                }

                if let Some(pos) = found_position {
                    positions.insert(thread_id, pos);
                } else {
                    return Err(BraidError::ConceptNotFound(via_concept));
                }
            }
        }

        // Create the braid connection
        let braid_id = self.generate_braid_id()?;
        let braid = BraidConnection::new(
            braid_id,
            thread_ids.to_vec(),
            via_concept,
            positions.clone(),
            BraidType::Semantic,
        );

        // Store the braid
        {
            let mut braids = self.braids.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            braids.insert(braid_id, braid);
        }

        // Update threads with braid points
        {
            let mut threads = self.threads.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;

            for (&thread_id, &position) in &positions {
                if let Some(thread) = threads.get_mut(&thread_id) {
                    thread.add_braid_point(position, braid_id);
                }
            }
        }

        // Check for homotopy equivalences with existing braids
        self.detect_homotopy_equivalences(braid_id)?;

        self.update_stats()?;
        Ok(braid_id)
    }

    /// Get the contents of a memory thread
    pub fn get_thread(&self, thread_id: ThreadId) -> Result<Vec<ConceptId>, BraidError> {
        let threads = self.threads.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let nodes = self.nodes.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let thread = threads.get(&thread_id)
            .ok_or(BraidError::ThreadNotFound(thread_id))?;

        let concept_ids: Vec<ConceptId> = thread.nodes
            .iter()
            .filter_map(|&node_id| {
                nodes.get(&node_id).map(|node| node.concept_id)
            })
            .collect();

        Ok(concept_ids)
    }

    /// Get detailed thread information
    pub fn get_thread_info(&self, thread_id: ThreadId) -> Result<MemoryThread, BraidError> {
        let threads = self.threads.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        threads.get(&thread_id)
            .cloned()
            .ok_or(BraidError::ThreadNotFound(thread_id))
    }

    /// Get all memory nodes in a thread with full details
    pub fn get_thread_nodes(&self, thread_id: ThreadId) -> Result<Vec<MemoryNode>, BraidError> {
        let threads = self.threads.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let nodes = self.nodes.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let thread = threads.get(&thread_id)
            .ok_or(BraidError::ThreadNotFound(thread_id))?;

        let thread_nodes: Vec<MemoryNode> = thread.nodes
            .iter()
            .filter_map(|&node_id| nodes.get(&node_id).cloned())
            .collect();

        Ok(thread_nodes)
    }

    /// Get information about a specific braid
    pub fn get_braid(&self, braid_id: BraidId) -> Result<BraidConnection, BraidError> {
        let braids = self.braids.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        braids.get(&braid_id)
            .cloned()
            .ok_or(BraidError::BraidNotFound(braid_id))
    }

    /// Find all braids that involve a specific thread
    pub fn get_braids_for_thread(&self, thread_id: ThreadId) -> Result<Vec<BraidConnection>, BraidError> {
        let braids = self.braids.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let thread_braids: Vec<BraidConnection> = braids
            .values()
            .filter(|braid| braid.involves_thread(thread_id))
            .cloned()
            .collect();

        Ok(thread_braids)
    }

    /// Find all threads that share braids with a given thread
    pub fn get_connected_threads(&self, thread_id: ThreadId) -> Result<Vec<ThreadId>, BraidError> {
        let braids = self.braids.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let mut connected_threads = HashSet::new();

        for braid in braids.values() {
            if braid.involves_thread(thread_id) {
                for &other_thread_id in &braid.threads {
                    if other_thread_id != thread_id {
                        connected_threads.insert(other_thread_id);
                    }
                }
            }
        }

        Ok(connected_threads.into_iter().collect())
    }

    /// Remove a memory thread and all its nodes
    pub fn remove_thread(&mut self, thread_id: ThreadId) -> Result<(), BraidError> {
        // Get thread node IDs before removal
        let node_ids = {
            let threads = self.threads.read()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            let thread = threads.get(&thread_id)
                .ok_or(BraidError::ThreadNotFound(thread_id))?;
            thread.nodes.clone()
        };

        // Remove all nodes
        {
            let mut nodes = self.nodes.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            for node_id in node_ids {
                nodes.remove(&node_id);
            }
        }

        // Remove braids that involve this thread
        let braids_to_remove: Vec<BraidId> = {
            let braids = self.braids.read()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            braids
                .iter()
                .filter(|(_, braid)| braid.involves_thread(thread_id))
                .map(|(&braid_id, _)| braid_id)
                .collect()
        };

        {
            let mut braids = self.braids.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            for braid_id in braids_to_remove {
                braids.remove(&braid_id);
            }
        }

        // Remove the thread
        {
            let mut threads = self.threads.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            threads.remove(&thread_id);
        }

        self.update_stats()?;
        Ok(())
    }

    /// Clear all memory (reset the system)
    pub fn clear_all(&mut self) -> Result<(), BraidError> {
        {
            let mut threads = self.threads.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            threads.clear();
        }

        {
            let mut nodes = self.nodes.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            nodes.clear();
        }

        {
            let mut braids = self.braids.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            braids.clear();
        }

        {
            let mut homotopies = self.homotopies.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            homotopies.clear();
        }

        // Reset ID counters
        {
            let mut next_thread_id = self.next_thread_id.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            *next_thread_id = 1;
        }

        {
            let mut next_node_id = self.next_node_id.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            *next_node_id = 1;
        }

        {
            let mut next_braid_id = self.next_braid_id.write()
                .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            *next_braid_id = 1;
        }

        self.update_stats()?;
        Ok(())
    }

    /// Get current system statistics
    pub fn get_stats(&self) -> Result<BraidMemoryStats, BraidError> {
        let stats = self.stats.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(stats.clone())
    }

    /// Detect homotopy equivalences between braids (∞-groupoid coherence)
    fn detect_homotopy_equivalences(&mut self, new_braid_id: BraidId) -> Result<(), BraidError> {
        let braids = self.braids.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let new_braid = braids.get(&new_braid_id)
            .ok_or(BraidError::BraidNotFound(new_braid_id))?;

        // Look for existing braids that might be homotopy equivalent
        for (&existing_braid_id, existing_braid) in braids.iter() {
            if existing_braid_id == new_braid_id {
                continue;
            }

            // Check for homotopy equivalence criteria:
            // 1. Same concept
            // 2. Same or overlapping threads
            // 3. Similar temporal context
            if existing_braid.concept_id == new_braid.concept_id {
                let thread_overlap: HashSet<_> = existing_braid.threads.iter()
                    .filter(|&t| new_braid.threads.contains(t))
                    .collect();

                if !thread_overlap.is_empty() {
                    // Found a potential homotopy equivalence
                    let confidence = thread_overlap.len() as f64 / 
                        (existing_braid.threads.len().max(new_braid.threads.len()) as f64);

                    if confidence > 0.5 {
                        let homotopy = HomotopyRelation {
                            source_braid: existing_braid_id,
                            target_braid: new_braid_id,
                            transformation: format!(
                                "Concept {} equivalence with {} thread overlap",
                                new_braid.concept_id,
                                thread_overlap.len()
                            ),
                            confidence,
                            created_at: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };

                        let mut homotopies = self.homotopies.write()
                            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
                        homotopies.push(homotopy);
                    }
                }
            }
        }

        Ok(())
    }

    /// Update system statistics
    fn update_stats(&self) -> Result<(), BraidError> {
        let mut stats = self.stats.write()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;

        let threads = self.threads.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let nodes = self.nodes.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let braids = self.braids.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let homotopies = self.homotopies.read()
            .map_err(|e| BraidError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        stats.total_threads = threads.len();
        stats.total_nodes = nodes.len();
        stats.total_braids = braids.len();
        stats.total_homotopies = homotopies.len();

        // Calculate active threads (modified in last 24 hours)
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let day_ago = now.saturating_sub(24 * 60 * 60);
        stats.active_threads = threads.values()
            .filter(|thread| thread.modified_at > day_ago)
            .count();

        // Calculate thread length statistics
        if !threads.is_empty() {
            let total_length: usize = threads.values().map(|t| t.length()).sum();
            stats.average_thread_length = total_length as f64 / threads.len() as f64;
            stats.max_thread_length = threads.values().map(|t| t.length()).max().unwrap_or(0);
        } else {
            stats.average_thread_length = 0.0;
            stats.max_thread_length = 0;
        }

        // Calculate braid density (braids per thread pair)
        if threads.len() > 1 {
            let possible_connections = threads.len() * (threads.len() - 1) / 2;
            stats.braid_density = braids.len() as f64 / possible_connections as f64;
        } else {
            stats.braid_density = 0.0;
        }

        // Calculate coherence score based on homotopy equivalences
        if braids.len() > 0 {
            stats.coherence_score = 1.0 - (homotopies.len() as f64 / braids.len() as f64).min(1.0);
        } else {
            stats.coherence_score = 1.0;
        }

        stats.last_updated = now;

        Ok(())
    }

    /// Internal sanity check for testing and debugging
    #[cfg(test)]
    pub fn internal_sanity_check(&self) -> Result<(), String> {
        let threads = self.threads.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        let nodes = self.nodes.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        let braids = self.braids.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        // Check that all thread nodes exist in the nodes collection
        for (thread_id, thread) in threads.iter() {
            for (pos, &node_id) in thread.nodes.iter().enumerate() {
                if let Some(node) = nodes.get(&node_id) {
                    if node.thread_id != *thread_id {
                        return Err(format!("Node {} claims thread {} but is in thread {}", 
                                         node_id, node.thread_id, thread_id));
                    }
                    if node.position != pos {
                        return Err(format!("Node {} position mismatch: expected {}, got {}", 
                                         node_id, pos, node.position));
                    }
                } else {
                    return Err(format!("Thread {} references non-existent node {}", 
                                     thread_id, node_id));
                }
            }
        }

        // Check that all braids reference valid threads
        for (braid_id, braid) in braids.iter() {
            for &thread_id in &braid.threads {
                if !threads.contains_key(&thread_id) {
                    return Err(format!("Braid {} references non-existent thread {}", 
                                     braid_id, thread_id));
                }
            }

            // Check that braid positions are valid
            for (&thread_id, &position) in &braid.positions {
                if let Some(thread) = threads.get(&thread_id) {
                    if position >= thread.length() {
                        return Err(format!("Braid {} has invalid position {} for thread {} (length {})", 
                                         braid_id, position, thread_id, thread.length()));
                    }
                }
            }
        }

        // Check that causal links in nodes reference valid nodes
        for (node_id, node) in nodes.iter() {
            for &linked_node_id in &node.causal_links {
                if !nodes.contains_key(&linked_node_id) {
                    return Err(format!("Node {} has causal link to non-existent node {}", 
                                     node_id, linked_node_id));
                }
            }
        }

        Ok(())
    }
}

impl Default for BraidMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// UTILITY FUNCTIONS
// ===================================================================

/// Create a simple memory thread for testing
pub fn create_simple_thread(title: &str) -> MemoryThread {
    MemoryThread::new(1, title.to_string())
}

/// Create a simple memory node for testing
pub fn create_simple_node(concept_id: ConceptId, thread_id: ThreadId, position: usize) -> MemoryNode {
    MemoryNode::new(1, concept_id, thread_id, position, None)
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braid_memory_creation() {
        let memory = BraidMemory::new();
        let stats = memory.get_stats().unwrap();
        assert_eq!(stats.total_threads, 0);
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_braids, 0);
    }

    #[test]
    fn test_thread_creation() {
        let mut memory = BraidMemory::new();
        let thread_id = memory.start_thread("Test Thread".to_string()).unwrap();
        assert_eq!(thread_id, 1);

        let thread_info = memory.get_thread_info(thread_id).unwrap();
        assert_eq!(thread_info.title, "Test Thread");
        assert_eq!(thread_info.length(), 0);
    }

    #[test]
    fn test_append_to_thread() {
        let mut memory = BraidMemory::new();
        let thread_id = memory.start_thread("Test Thread".to_string()).unwrap();
        
        let node_id = memory.append_to_thread(thread_id, 42, Some("Test context".to_string())).unwrap();
        assert_eq!(node_id, 1);

        let concepts = memory.get_thread(thread_id).unwrap();
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0], 42);
    }

    #[test]
    fn test_braid_threads() {
        let mut memory = BraidMemory::new();
        
        // Create two threads
        let thread1 = memory.start_thread("Thread 1".to_string()).unwrap();
        let thread2 = memory.start_thread("Thread 2".to_string()).unwrap();
        
        // Add same concept to both threads
        memory.append_to_thread(thread1, 100, None).unwrap();
        memory.append_to_thread(thread2, 100, None).unwrap();
        
        // Braid them
        let braid_id = memory.braid_threads(&[thread1, thread2], 100).unwrap();
        assert_eq!(braid_id, 1);

        let braid = memory.get_braid(braid_id).unwrap();
        assert_eq!(braid.concept_id, 100);
        assert_eq!(braid.threads.len(), 2);
        assert!(braid.threads.contains(&thread1));
        assert!(braid.threads.contains(&thread2));
    }

    #[test]
    fn test_connected_threads() {
        let mut memory = BraidMemory::new();
        
        let thread1 = memory.start_thread("Thread 1".to_string()).unwrap();
        let thread2 = memory.start_thread("Thread 2".to_string()).unwrap();
        let thread3 = memory.start_thread("Thread 3".to_string()).unwrap();
        
        // Add concept to all threads
        memory.append_to_thread(thread1, 200, None).unwrap();
        memory.append_to_thread(thread2, 200, None).unwrap();
        memory.append_to_thread(thread3, 300, None).unwrap(); // Different concept
        
        // Braid first two threads
        memory.braid_threads(&[thread1, thread2], 200).unwrap();
        
        // Check connections
        let connected = memory.get_connected_threads(thread1).unwrap();
        assert_eq!(connected.len(), 1);
        assert!(connected.contains(&thread2));
        
        let connected3 = memory.get_connected_threads(thread3).unwrap();
        assert_eq!(connected3.len(), 0);
    }

    #[test]
    fn test_remove_thread() {
        let mut memory = BraidMemory::new();
        
        let thread_id = memory.start_thread("Test Thread".to_string()).unwrap();
        memory.append_to_thread(thread_id, 42, None).unwrap();
        
        assert!(memory.get_thread_info(thread_id).is_ok());
        
        memory.remove_thread(thread_id).unwrap();
        
        assert!(memory.get_thread_info(thread_id).is_err());
        
        let stats = memory.get_stats().unwrap();
        assert_eq!(stats.total_threads, 0);
        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn test_sanity_check() {
        let mut memory = BraidMemory::new();
        
        let thread1 = memory.start_thread("Thread 1".to_string()).unwrap();
        let thread2 = memory.start_thread("Thread 2".to_string()).unwrap();
        
        memory.append_to_thread(thread1, 100, None).unwrap();
        memory.append_to_thread(thread2, 100, None).unwrap();
        memory.braid_threads(&[thread1, thread2], 100).unwrap();
        
        assert!(memory.internal_sanity_check().is_ok());
    }

    #[test]
    fn test_thread_statistics() {
        let mut memory = BraidMemory::new();
        
        let thread1 = memory.start_thread("Thread 1".to_string()).unwrap();
        let thread2 = memory.start_thread("Thread 2".to_string()).unwrap();
        
        memory.append_to_thread(thread1, 1, None).unwrap();
        memory.append_to_thread(thread1, 2, None).unwrap();
        memory.append_to_thread(thread2, 3, None).unwrap();
        
        let stats = memory.get_stats().unwrap();
        assert_eq!(stats.total_threads, 2);
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.average_thread_length, 1.5);
        assert_eq!(stats.max_thread_length, 2);
    }
}
