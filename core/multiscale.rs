// ===================================================================
// TORI MultiScaleHierarchy - Core Implementation
// Mathematical Foundation: Hierarchical concept organization with
// dynamic scale adjustment and thread-safe concurrent operations
// ===================================================================

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

// ===================================================================
// CORE DATA TYPES AND STRUCTURES
// ===================================================================

/// Unique identifier for concepts in the hierarchy
pub type ConceptId = u64;

/// Scale level in the hierarchy (0 = most detailed, higher = more abstract)
pub type ScaleLevel = u8;

/// Unique identifier for hierarchy nodes
pub type NodeId = u64;

/// Errors that can occur in hierarchy operations
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchyError {
    ConceptNotFound(ConceptId),
    NodeNotFound(NodeId),
    InvalidParent(ConceptId),
    CycleDetected(ConceptId, ConceptId),
    ScaleViolation(ScaleLevel, ScaleLevel),
    ConcurrencyError(String),
    InvalidOperation(String),
}

impl std::fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HierarchyError::ConceptNotFound(id) => write!(f, "Concept {} not found", id),
            HierarchyError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            HierarchyError::InvalidParent(id) => write!(f, "Invalid parent concept {}", id),
            HierarchyError::CycleDetected(a, b) => write!(f, "Cycle detected between {} and {}", a, b),
            HierarchyError::ScaleViolation(expected, actual) => {
                write!(f, "Scale violation: expected {}, got {}", expected, actual)
            }
            HierarchyError::ConcurrencyError(msg) => write!(f, "Concurrency error: {}", msg),
            HierarchyError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for HierarchyError {}

/// Content data associated with a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptData {
    /// Human-readable name
    pub name: String,
    /// Optional description or summary
    pub description: Option<String>,
    /// Semantic embedding vector (for ML operations)
    pub embedding: Option<Vec<f64>>,
    /// Domain classification
    pub domain: Option<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ConceptData {
    pub fn new(name: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            name,
            description: None,
            embedding: None,
            domain: None,
            created_at: now,
            modified_at: now,
            metadata: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self
    }

    pub fn with_domain(mut self, domain: String) -> Self {
        self.domain = Some(domain);
        self.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self
    }
}

/// A node in the multi-scale hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Associated concept identifier
    pub concept_id: ConceptId,
    /// Scale level (0 = most detailed)
    pub scale: ScaleLevel,
    /// Parent node (None for root nodes)
    pub parent: Option<NodeId>,
    /// Child nodes
    pub children: HashSet<NodeId>,
    /// Node creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Node-specific metadata
    pub node_metadata: HashMap<String, String>,
}

impl HierarchyNode {
    pub fn new(id: NodeId, concept_id: ConceptId, scale: ScaleLevel, parent: Option<NodeId>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            concept_id,
            scale,
            parent,
            children: HashSet::new(),
            created_at: now,
            last_accessed: now,
            node_metadata: HashMap::new(),
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, child_id: NodeId) {
        self.children.insert(child_id);
    }

    /// Remove a child node
    pub fn remove_child(&mut self, child_id: NodeId) -> bool {
        self.children.remove(&child_id)
    }

    /// Update last accessed timestamp
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// View of a hierarchy subtree for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyView {
    /// Root node of this view
    pub root: HierarchyNode,
    /// All nodes in the subtree
    pub nodes: HashMap<NodeId, HierarchyNode>,
    /// Concept data for all concepts in the view
    pub concepts: HashMap<ConceptId, ConceptData>,
    /// Maximum depth included in this view
    pub max_depth: Option<usize>,
    /// Total number of nodes
    pub node_count: usize,
}

impl HierarchyView {
    pub fn new(root: HierarchyNode) -> Self {
        let node_count = 1;
        let mut nodes = HashMap::new();
        nodes.insert(root.id, root.clone());

        Self {
            root,
            nodes,
            concepts: HashMap::new(),
            max_depth: Some(0),
            node_count,
        }
    }

    /// Add a node to the view
    pub fn add_node(&mut self, node: HierarchyNode, concept: Option<ConceptData>) {
        self.nodes.insert(node.id, node.clone());
        if let Some(concept_data) = concept {
            self.concepts.insert(node.concept_id, concept_data);
        }
        self.node_count = self.nodes.len();
    }

    /// Get all leaf nodes in the view
    pub fn get_leaves(&self) -> Vec<&HierarchyNode> {
        self.nodes
            .values()
            .filter(|node| node.children.is_empty())
            .collect()
    }

    /// Get nodes at a specific scale level
    pub fn get_nodes_at_scale(&self, scale: ScaleLevel) -> Vec<&HierarchyNode> {
        self.nodes
            .values()
            .filter(|node| node.scale == scale)
            .collect()
    }
}

// ===================================================================
// MULTI-SCALE HIERARCHY IMPLEMENTATION
// ===================================================================

/// The main MultiScaleHierarchy structure
pub struct MultiScaleHierarchy {
    /// All nodes in the hierarchy (thread-safe)
    nodes: Arc<RwLock<HashMap<NodeId, HierarchyNode>>>,
    /// All concept data (thread-safe)
    concepts: Arc<RwLock<HashMap<ConceptId, ConceptData>>>,
    /// Mapping from concept ID to node IDs (a concept can exist at multiple scales)
    concept_to_nodes: Arc<RwLock<HashMap<ConceptId, HashSet<NodeId>>>>,
    /// Root nodes at each scale level
    scale_roots: Arc<RwLock<HashMap<ScaleLevel, HashSet<NodeId>>>>,
    /// Next available node ID
    next_node_id: Arc<RwLock<NodeId>>,
    /// Hierarchy statistics
    stats: Arc<RwLock<HierarchyStats>>,
}

/// Statistics about the hierarchy
#[derive(Debug, Clone, Default)]
pub struct HierarchyStats {
    pub total_nodes: usize,
    pub total_concepts: usize,
    pub max_scale: ScaleLevel,
    pub max_depth: usize,
    pub operations_count: u64,
    pub last_modified: u64,
}

impl MultiScaleHierarchy {
    /// Create a new empty hierarchy
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            concepts: Arc::new(RwLock::new(HashMap::new())),
            concept_to_nodes: Arc::new(RwLock::new(HashMap::new())),
            scale_roots: Arc::new(RwLock::new(HashMap::new())),
            next_node_id: Arc::new(RwLock::new(1)),
            stats: Arc::new(RwLock::new(HierarchyStats::default())),
        }
    }

    /// Generate a new unique node ID
    fn generate_node_id(&self) -> Result<NodeId, HierarchyError> {
        let mut next_id = self.next_node_id.write()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock for node ID: {}", e)))?;
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    /// Add a new concept to the hierarchy
    pub fn add_concept(
        &mut self,
        concept_id: ConceptId,
        content: ConceptData,
        parent: Option<ConceptId>,
        scale: ScaleLevel,
    ) -> Result<ConceptId, HierarchyError> {
        // Validate parent exists if specified
        if let Some(parent_id) = parent {
            let concepts = self.concepts.read()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            if !concepts.contains_key(&parent_id) {
                return Err(HierarchyError::ConceptNotFound(parent_id));
            }
        }

        // Check for scale violations (child must be at lower or equal scale than parent)
        if let Some(parent_id) = parent {
            let parent_scale = self.get_concept_primary_scale(parent_id)?;
            if scale > parent_scale {
                return Err(HierarchyError::ScaleViolation(parent_scale, scale));
            }
        }

        // Add concept data
        {
            let mut concepts = self.concepts.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            concepts.insert(concept_id, content);
        }

        // Create hierarchy node
        let node_id = self.generate_node_id()?;
        let parent_node_id = if let Some(parent_concept_id) = parent {
            Some(self.get_primary_node_for_concept(parent_concept_id)?)
        } else {
            None
        };

        let mut node = HierarchyNode::new(node_id, concept_id, scale, parent_node_id);

        // Add to parent's children if parent exists
        if let Some(parent_node_id) = parent_node_id {
            let mut nodes = self.nodes.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            if let Some(parent_node) = nodes.get_mut(&parent_node_id) {
                parent_node.add_child(node_id);
            }
        }

        // Add node to hierarchy
        {
            let mut nodes = self.nodes.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            nodes.insert(node_id, node);
        }

        // Update concept-to-nodes mapping
        {
            let mut concept_to_nodes = self.concept_to_nodes.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            concept_to_nodes.entry(concept_id).or_insert_with(HashSet::new).insert(node_id);
        }

        // Update scale roots if this is a root node
        if parent.is_none() {
            let mut scale_roots = self.scale_roots.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
            scale_roots.entry(scale).or_insert_with(HashSet::new).insert(node_id);
        }

        // Update statistics
        self.update_stats()?;

        Ok(concept_id)
    }

    /// Link two existing concepts (create parent-child relationship)
    pub fn link_concepts(&mut self, parent: ConceptId, child: ConceptId) -> Result<(), HierarchyError> {
        // Validate both concepts exist
        {
            let concepts = self.concepts.read()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            if !concepts.contains_key(&parent) {
                return Err(HierarchyError::ConceptNotFound(parent));
            }
            if !concepts.contains_key(&child) {
                return Err(HierarchyError::ConceptNotFound(child));
            }
        }

        // Check for cycles (basic check - child cannot be ancestor of parent)
        if self.is_ancestor(child, parent)? {
            return Err(HierarchyError::CycleDetected(parent, child));
        }

        // Get primary nodes for both concepts
        let parent_node_id = self.get_primary_node_for_concept(parent)?;
        let child_node_id = self.get_primary_node_for_concept(child)?;

        // Validate scale relationship
        let parent_scale = {
            let nodes = self.nodes.read()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            nodes.get(&parent_node_id)
                .ok_or(HierarchyError::NodeNotFound(parent_node_id))?
                .scale
        };

        let child_scale = {
            let nodes = self.nodes.read()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
            nodes.get(&child_node_id)
                .ok_or(HierarchyError::NodeNotFound(child_node_id))?
                .scale
        };

        if child_scale > parent_scale {
            return Err(HierarchyError::ScaleViolation(parent_scale, child_scale));
        }

        // Create the link
        {
            let mut nodes = self.nodes.write()
                .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;

            // Add child to parent's children
            if let Some(parent_node) = nodes.get_mut(&parent_node_id) {
                parent_node.add_child(child_node_id);
            }

            // Set parent for child
            if let Some(child_node) = nodes.get_mut(&child_node_id) {
                child_node.parent = Some(parent_node_id);
            }
        }

        // Update statistics
        self.update_stats()?;

        Ok(())
    }

    /// Get a subhierarchy rooted at a specific concept
    pub fn get_subhierarchy(&self, root: ConceptId) -> Result<HierarchyView, HierarchyError> {
        let root_node_id = self.get_primary_node_for_concept(root)?;
        self.get_subhierarchy_by_node(root_node_id)
    }

    /// Get a subhierarchy rooted at a specific node
    pub fn get_subhierarchy_by_node(&self, root_node_id: NodeId) -> Result<HierarchyView, HierarchyError> {
        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let concepts = self.concepts.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let root_node = nodes.get(&root_node_id)
            .ok_or(HierarchyError::NodeNotFound(root_node_id))?
            .clone();

        let mut view = HierarchyView::new(root_node.clone());

        // Add root concept data
        if let Some(concept_data) = concepts.get(&root_node.concept_id) {
            view.concepts.insert(root_node.concept_id, concept_data.clone());
        }

        // Traverse children recursively
        let mut queue = VecDeque::new();
        queue.push_back((root_node_id, 0));
        let mut max_depth = 0;

        while let Some((node_id, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);

            if let Some(node) = nodes.get(&node_id) {
                if node_id != root_node_id {
                    view.add_node(node.clone(), concepts.get(&node.concept_id).cloned());
                }

                // Add children to queue
                for &child_id in &node.children {
                    queue.push_back((child_id, depth + 1));
                }
            }
        }

        view.max_depth = Some(max_depth);
        Ok(view)
    }

    /// Get all children of a concept
    pub fn get_children(&self, concept_id: ConceptId) -> Result<Vec<ConceptId>, HierarchyError> {
        let node_id = self.get_primary_node_for_concept(concept_id)?;
        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let node = nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))?;

        let mut children = Vec::new();
        for &child_node_id in &node.children {
            if let Some(child_node) = nodes.get(&child_node_id) {
                children.push(child_node.concept_id);
            }
        }

        Ok(children)
    }

    /// Get the parent of a concept
    pub fn get_parent(&self, concept_id: ConceptId) -> Result<Option<ConceptId>, HierarchyError> {
        let node_id = self.get_primary_node_for_concept(concept_id)?;
        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let node = nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))?;

        if let Some(parent_node_id) = node.parent {
            if let Some(parent_node) = nodes.get(&parent_node_id) {
                return Ok(Some(parent_node.concept_id));
            }
        }

        Ok(None)
    }

    /// Get all concepts at a specific scale level
    pub fn get_concepts_at_scale(&self, scale: ScaleLevel) -> Result<Vec<ConceptId>, HierarchyError> {
        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let concepts: Vec<ConceptId> = nodes
            .values()
            .filter(|node| node.scale == scale)
            .map(|node| node.concept_id)
            .collect();

        Ok(concepts)
    }

    /// Check if concept A is an ancestor of concept B
    pub fn is_ancestor(&self, ancestor: ConceptId, descendant: ConceptId) -> Result<bool, HierarchyError> {
        let ancestor_node_id = self.get_primary_node_for_concept(ancestor)?;
        let descendant_node_id = self.get_primary_node_for_concept(descendant)?;

        self.is_ancestor_by_node(ancestor_node_id, descendant_node_id)
    }

    /// Check if node A is an ancestor of node B
    fn is_ancestor_by_node(&self, ancestor_node_id: NodeId, descendant_node_id: NodeId) -> Result<bool, HierarchyError> {
        if ancestor_node_id == descendant_node_id {
            return Ok(false); // A node is not its own ancestor
        }

        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let mut current = descendant_node_id;
        let mut visited = HashSet::new();

        while let Some(node) = nodes.get(&current) {
            if visited.contains(&current) {
                // Cycle detected - should not happen in a well-formed hierarchy
                return Err(HierarchyError::CycleDetected(ancestor_node_id, descendant_node_id));
            }
            visited.insert(current);

            if let Some(parent_id) = node.parent {
                if parent_id == ancestor_node_id {
                    return Ok(true);
                }
                current = parent_id;
            } else {
                break;
            }
        }

        Ok(false)
    }

    /// Get primary node for a concept (first node found, typically at lowest scale)
    fn get_primary_node_for_concept(&self, concept_id: ConceptId) -> Result<NodeId, HierarchyError> {
        let concept_to_nodes = self.concept_to_nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let node_ids = concept_to_nodes.get(&concept_id)
            .ok_or(HierarchyError::ConceptNotFound(concept_id))?;

        // Return the first node (could be improved to return the node at the most appropriate scale)
        node_ids.iter().next()
            .copied()
            .ok_or(HierarchyError::ConceptNotFound(concept_id))
    }

    /// Get primary scale for a concept
    fn get_concept_primary_scale(&self, concept_id: ConceptId) -> Result<ScaleLevel, HierarchyError> {
        let node_id = self.get_primary_node_for_concept(concept_id)?;
        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        let node = nodes.get(&node_id)
            .ok_or(HierarchyError::NodeNotFound(node_id))?;

        Ok(node.scale)
    }

    /// Update hierarchy statistics
    fn update_stats(&self) -> Result<(), HierarchyError> {
        let mut stats = self.stats.write()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;

        let nodes = self.nodes.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        let concepts = self.concepts.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;

        stats.total_nodes = nodes.len();
        stats.total_concepts = concepts.len();
        stats.max_scale = nodes.values().map(|n| n.scale).max().unwrap_or(0);
        stats.operations_count += 1;
        stats.last_modified = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate max depth (expensive operation, might want to cache)
        let mut max_depth = 0;
        for node in nodes.values() {
            if node.parent.is_none() {
                let depth = self.calculate_subtree_depth_by_node(node.id, &nodes)?;
                max_depth = max_depth.max(depth);
            }
        }
        stats.max_depth = max_depth;

        Ok(())
    }

    /// Calculate the depth of a subtree rooted at a specific node
    fn calculate_subtree_depth_by_node(
        &self,
        root_node_id: NodeId,
        nodes: &HashMap<NodeId, HierarchyNode>,
    ) -> Result<usize, HierarchyError> {
        let mut max_depth = 0;
        let mut stack = vec![(root_node_id, 0)];

        while let Some((node_id, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);

            if let Some(node) = nodes.get(&node_id) {
                for &child_id in &node.children {
                    stack.push((child_id, depth + 1));
                }
            }
        }

        Ok(max_depth)
    }

    /// Get current hierarchy statistics
    pub fn get_stats(&self) -> Result<HierarchyStats, HierarchyError> {
        let stats = self.stats.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        Ok(stats.clone())
    }

    /// Internal sanity check for testing and debugging
    #[cfg(test)]
    pub fn internal_sanity_check(&self) -> Result<(), String> {
        let nodes = self.nodes.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        let concepts = self.concepts.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        let concept_to_nodes = self.concept_to_nodes.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        // Check that all nodes reference valid concepts
        for (node_id, node) in nodes.iter() {
            if !concepts.contains_key(&node.concept_id) {
                return Err(format!("Node {} references non-existent concept {}", node_id, node.concept_id));
            }

            // Check that parent-child relationships are symmetric
            if let Some(parent_id) = node.parent {
                if let Some(parent_node) = nodes.get(&parent_id) {
                    if !parent_node.children.contains(node_id) {
                        return Err(format!("Node {} has parent {} but parent doesn't list it as child", node_id, parent_id));
                    }
                } else {
                    return Err(format!("Node {} has non-existent parent {}", node_id, parent_id));
                }
            }

            // Check that children exist and reference this node as parent
            for &child_id in &node.children {
                if let Some(child_node) = nodes.get(&child_id) {
                    if child_node.parent != Some(*node_id) {
                        return Err(format!("Node {} lists {} as child but child has different parent", node_id, child_id));
                    }
                } else {
                    return Err(format!("Node {} has non-existent child {}", node_id, child_id));
                }
            }
        }

        // Check concept-to-nodes mapping consistency
        for (concept_id, node_ids) in concept_to_nodes.iter() {
            for &node_id in node_ids {
                if let Some(node) = nodes.get(&node_id) {
                    if node.concept_id != *concept_id {
                        return Err(format!("Concept-to-nodes mapping inconsistent: concept {} maps to node {} which has concept {}", concept_id, node_id, node.concept_id));
                    }
                } else {
                    return Err(format!("Concept {} maps to non-existent node {}", concept_id, node_id));
                }
            }
        }

        Ok(())
    }
}

impl Default for MultiScaleHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// UTILITY FUNCTIONS AND HELPERS
// ===================================================================

/// Generate a concept ID from a string (for testing and simple cases)
pub fn concept_id_from_string(s: &str) -> ConceptId {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Create a concept data from just a name (convenience function)
pub fn simple_concept(name: &str) -> ConceptData {
    ConceptData::new(name.to_string())
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_creation() {
        let hierarchy = MultiScaleHierarchy::new();
        let stats = hierarchy.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_concepts, 0);
    }

    #[test]
    fn test_add_concept() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        let concept_id = 1;
        let content = simple_concept("Test Concept");
        
        let result = hierarchy.add_concept(concept_id, content, None, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), concept_id);

        let stats = hierarchy.get_stats().unwrap();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_concepts, 1);
    }

    #[test]
    fn test_add_concept_with_parent() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        // Add parent concept
        let parent_id = 1;
        let parent_content = simple_concept("Parent");
        hierarchy.add_concept(parent_id, parent_content, None, 1).unwrap();

        // Add child concept
        let child_id = 2;
        let child_content = simple_concept("Child");
        let result = hierarchy.add_concept(child_id, child_content, Some(parent_id), 0);
        
        assert!(result.is_ok());

        // Check parent-child relationship
        let children = hierarchy.get_children(parent_id).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], child_id);

        let parent = hierarchy.get_parent(child_id).unwrap();
        assert_eq!(parent, Some(parent_id));
    }

    #[test]
    fn test_scale_violation() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        // Add parent at scale 0
        let parent_id = 1;
        let parent_content = simple_concept("Parent");
        hierarchy.add_concept(parent_id, parent_content, None, 0).unwrap();

        // Try to add child at higher scale (should fail)
        let child_id = 2;
        let child_content = simple_concept("Child");
        let result = hierarchy.add_concept(child_id, child_content, Some(parent_id), 1);
        
        assert!(matches!(result, Err(HierarchyError::ScaleViolation(0, 1))));
    }

    #[test]
    fn test_cycle_detection() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        // Add two concepts
        let concept_a = 1;
        let concept_b = 2;
        hierarchy.add_concept(concept_a, simple_concept("A"), None, 0).unwrap();
        hierarchy.add_concept(concept_b, simple_concept("B"), None, 0).unwrap();

        // Link A -> B
        hierarchy.link_concepts(concept_a, concept_b).unwrap();

        // Try to link B -> A (should create cycle and fail)
        let result = hierarchy.link_concepts(concept_b, concept_a);
        assert!(matches!(result, Err(HierarchyError::CycleDetected(_, _))));
    }

    #[test]
    fn test_subhierarchy() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        // Create a small hierarchy
        let root_id = 1;
        let child1_id = 2;
        let child2_id = 3;
        let grandchild_id = 4;

        hierarchy.add_concept(root_id, simple_concept("Root"), None, 2).unwrap();
        hierarchy.add_concept(child1_id, simple_concept("Child1"), Some(root_id), 1).unwrap();
        hierarchy.add_concept(child2_id, simple_concept("Child2"), Some(root_id), 1).unwrap();
        hierarchy.add_concept(grandchild_id, simple_concept("Grandchild"), Some(child1_id), 0).unwrap();

        let view = hierarchy.get_subhierarchy(root_id).unwrap();
        assert_eq!(view.node_count, 4);
        assert_eq!(view.max_depth, Some(2));
        assert_eq!(view.root.concept_id, root_id);
    }

    #[test]
    fn test_sanity_check() {
        let mut hierarchy = MultiScaleHierarchy::new();
        
        // Add some concepts
        hierarchy.add_concept(1, simple_concept("A"), None, 1).unwrap();
        hierarchy.add_concept(2, simple_concept("B"), Some(1), 0).unwrap();
        hierarchy.add_concept(3, simple_concept("C"), Some(1), 0).unwrap();

        // Sanity check should pass
        assert!(hierarchy.internal_sanity_check().is_ok());
    }
}

/// Thread-safe hierarchy operations for multi-threaded environments
impl MultiScaleHierarchy {
    /// Mark a node as accessed (for LRU or access-based operations)
    pub fn mark_accessed(&self, concept_id: ConceptId) -> Result<(), HierarchyError> {
        let node_id = self.get_primary_node_for_concept(concept_id)?;
        let mut nodes = self.nodes.write()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
        
        if let Some(node) = nodes.get_mut(&node_id) {
            node.touch();
        }
        
        Ok(())
    }

    /// Get concept data
    pub fn get_concept_data(&self, concept_id: ConceptId) -> Result<ConceptData, HierarchyError> {
        let concepts = self.concepts.read()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire read lock: {}", e)))?;
        
        concepts.get(&concept_id)
            .cloned()
            .ok_or(HierarchyError::ConceptNotFound(concept_id))
    }

    /// Update concept data
    pub fn update_concept_data(&mut self, concept_id: ConceptId, mut new_data: ConceptData) -> Result<(), HierarchyError> {
        new_data.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut concepts = self.concepts.write()
            .map_err(|e| HierarchyError::ConcurrencyError(format!("Failed to acquire write lock: {}", e)))?;
        
        if concepts.contains_key(&concept_id) {
            concepts.insert(concept_id, new_data);
            Ok(())
        } else {
            Err(HierarchyError::ConceptNotFound(concept_id))
        }
    }
}

pub use self::{
    MultiScaleHierarchy, HierarchyNode, HierarchyView, ConceptData, HierarchyError,
    ConceptId, ScaleLevel, NodeId, HierarchyStats,
    concept_id_from_string, simple_concept,
};
