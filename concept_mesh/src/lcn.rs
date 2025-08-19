//! Large Concept Network (LCN)
//!
//! The LCN represents the knowledge graph that stores concepts and their relationships.
//! It provides a phase-aligned storage system that replaces traditional embedding databases
//! with a concept-first approach.

use crate::diff::{ConceptDiff, ConceptDiffRef, NodeId, Op};
use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Phase vector representing a concept embedding
pub type PhaseVector = Array1<f32>;

/// A node in the concept graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Unique identifier for this concept
    pub id: NodeId,

    /// Type of this concept
    pub node_type: String,

    /// Properties associated with this concept
    pub properties: HashMap<String, serde_json::Value>,

    /// Phase vector for this concept (used for cognitive alignment)
    #[serde(skip)]
    pub phase_vector: Option<PhaseVector>,

    /// Creation timestamp for this concept
    pub created_at: u64,

    /// Last modification timestamp for this concept
    pub updated_at: u64,
}

impl Concept {
    /// Create a new concept
    pub fn new(id: impl Into<NodeId>, node_type: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: id.into(),
            node_type: node_type.into(),
            properties: HashMap::new(),
            phase_vector: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the phase vector for this concept
    pub fn with_phase_vector(mut self, phase_vector: PhaseVector) -> Self {
        self.phase_vector = Some(phase_vector);
        self
    }

    /// Add a property to this concept
    pub fn with_property(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.properties.insert(key.into(), value);
        }
        self
    }

    /// Update a property of this concept
    pub fn update_property(&mut self, key: &str, value: impl Serialize) -> bool {
        if let Ok(value) = serde_json::to_value(value) {
            self.properties.insert(key.to_string(), value);
            self.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            true
        } else {
            false
        }
    }

    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Get all property keys
    pub fn property_keys(&self) -> Vec<String> {
        self.properties.keys().cloned().collect()
    }
}

/// A relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelationship {
    /// Type of relationship
    pub rel_type: String,

    /// Properties associated with this relationship
    pub properties: HashMap<String, serde_json::Value>,

    /// Creation timestamp for this relationship
    pub created_at: u64,

    /// Last modification timestamp for this relationship
    pub updated_at: u64,
}

impl ConceptRelationship {
    /// Create a new relationship
    pub fn new(rel_type: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            rel_type: rel_type.into(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a property to this relationship
    pub fn with_property(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.properties.insert(key.into(), value);
        }
        self
    }

    /// Update a property of this relationship
    pub fn update_property(&mut self, key: &str, value: impl Serialize) -> bool {
        if let Ok(value) = serde_json::to_value(value) {
            self.properties.insert(key.to_string(), value);
            self.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            true
        } else {
            false
        }
    }
}

/// The LargeConceptNetwork stores the concept graph
pub struct LargeConceptNetwork {
    /// The graph data structure
    graph: RwLock<DiGraph<Concept, ConceptRelationship>>,

    /// Map from node IDs to graph indices
    node_indices: RwLock<HashMap<NodeId, NodeIndex>>,

    /// GENESIS timestamp
    genesis_timestamp: u64,

    /// Whether GENESIS has occurred
    genesis_complete: RwLock<bool>,
}

impl LargeConceptNetwork {
    /// Create a new empty LCN
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            node_indices: RwLock::new(HashMap::new()),
            genesis_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            genesis_complete: RwLock::new(false),
        }
    }

    /// Check if GENESIS has occurred
    pub fn is_genesis_complete(&self) -> bool {
        *self.genesis_complete.read().unwrap()
    }

    /// Apply a ConceptDiff to the graph
    pub fn apply_diff(&self, diff: &ConceptDiff) -> Result<(), String> {
        // Check for GENESIS frame
        if diff.frame_id == crate::GENESIS_FRAME_ID && !self.is_genesis_complete() {
            // Mark GENESIS as complete and record timestamp
            *self.genesis_complete.write().unwrap() = true;
        }

        // Apply each operation in order
        for op in &diff.ops {
            match op {
                Op::Create {
                    node,
                    node_type,
                    properties,
                } => {
                    self.create_node(node, node_type, properties)?;
                }
                Op::Delete { node } => {
                    self.delete_node(node)?;
                }
                Op::Update { node, properties } => {
                    self.update_node(node, properties)?;
                }
                Op::Link {
                    source,
                    target,
                    rel_type,
                    properties,
                } => {
                    self.create_relationship(source, target, rel_type, properties)?;
                }
                Op::Unlink {
                    source,
                    target,
                    rel_type,
                } => {
                    self.delete_relationship(source, target, rel_type.as_deref())?;
                }
                Op::Bind {
                    parent,
                    node,
                    bind_type,
                } => {
                    self.create_relationship(parent, node, bind_type, &HashMap::new())?;
                }
                Op::Signal { .. } => {
                    // Signals don't directly affect the graph
                    continue;
                }
                Op::Execute { .. } => {
                    // Executions don't directly affect the graph
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Create a new node in the graph
    fn create_node(
        &self,
        id: &NodeId,
        node_type: &str,
        properties: &HashMap<String, serde_json::Value>,
    ) -> Result<NodeIndex, String> {
        let mut node_indices = self.node_indices.write().unwrap();
        let mut graph = self.graph.write().unwrap();

        if node_indices.contains_key(id) {
            return Err(format!("Node {} already exists", id));
        }

        let mut concept = Concept::new(id, node_type);
        concept.properties = properties.clone();

        let node_idx = graph.add_node(concept);
        node_indices.insert(id.clone(), node_idx);

        Ok(node_idx)
    }

    /// Delete a node from the graph
    fn delete_node(&self, id: &NodeId) -> Result<(), String> {
        let mut node_indices = self.node_indices.write().unwrap();
        let mut graph = self.graph.write().unwrap();

        let node_idx = node_indices
            .get(id)
            .ok_or_else(|| format!("Node {} not found", id))?;

        // Remove the node from the graph
        graph.remove_node(*node_idx);
        node_indices.remove(id);

        Ok(())
    }

    /// Update a node in the graph
    fn update_node(
        &self,
        id: &NodeId,
        properties: &HashMap<String, serde_json::Value>,
    ) -> Result<(), String> {
        let node_indices = self.node_indices.read().unwrap();
        let mut graph = self.graph.write().unwrap();

        let node_idx = node_indices
            .get(id)
            .ok_or_else(|| format!("Node {} not found", id))?;

        let node = graph
            .node_weight_mut(*node_idx)
            .ok_or_else(|| format!("Node {} has invalid index", id))?;

        // Update properties
        for (key, value) in properties {
            node.properties.insert(key.clone(), value.clone());
        }

        // Update timestamp
        node.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(())
    }

    /// Create a relationship between nodes
    fn create_relationship(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
        rel_type: &str,
        properties: &HashMap<String, serde_json::Value>,
    ) -> Result<(), String> {
        let node_indices = self.node_indices.read().unwrap();
        let mut graph = self.graph.write().unwrap();

        let source_idx = node_indices
            .get(source_id)
            .ok_or_else(|| format!("Source node {} not found", source_id))?;

        let target_idx = node_indices
            .get(target_id)
            .ok_or_else(|| format!("Target node {} not found", target_id))?;

        // Check if relationship already exists
        for edge in graph.edges_directed(*source_idx, Direction::Outgoing) {
            if edge.target() == *target_idx {
                let rel = graph.edge_weight_mut(edge.id()).unwrap();
                if rel.rel_type == rel_type {
                    // Update existing relationship
                    for (key, value) in properties {
                        rel.properties.insert(key.clone(), value.clone());
                    }
                    return Ok(());
                }
            }
        }

        // Create new relationship
        let mut rel = ConceptRelationship::new(rel_type);
        rel.properties = properties.clone();

        graph.add_edge(*source_idx, *target_idx, rel);

        Ok(())
    }

    /// Delete a relationship between nodes
    fn delete_relationship(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
        rel_type: Option<&str>,
    ) -> Result<(), String> {
        let node_indices = self.node_indices.read().unwrap();
        let mut graph = self.graph.write().unwrap();

        let source_idx = node_indices
            .get(source_id)
            .ok_or_else(|| format!("Source node {} not found", source_id))?;

        let target_idx = node_indices
            .get(target_id)
            .ok_or_else(|| format!("Target node {} not found", target_id))?;

        // Collect edges to remove
        let mut edges_to_remove = Vec::new();

        for edge in graph.edges_directed(*source_idx, Direction::Outgoing) {
            if edge.target() == *target_idx {
                if let Some(rel_type) = rel_type {
                    let rel = graph.edge_weight(edge.id()).unwrap();
                    if rel.rel_type == rel_type {
                        edges_to_remove.push(edge.id());
                    }
                } else {
                    edges_to_remove.push(edge.id());
                }
            }
        }

        // Remove edges
        for edge_id in edges_to_remove {
            graph.remove_edge(edge_id);
        }

        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &NodeId) -> Option<Concept> {
        let node_indices = self.node_indices.read().unwrap();
        let graph = self.graph.read().unwrap();

        let node_idx = node_indices.get(id)?;
        let node = graph.node_weight(*node_idx)?;

        Some(node.clone())
    }

    /// Get all nodes
    pub fn get_all_nodes(&self) -> Vec<Concept> {
        let graph = self.graph.read().unwrap();
        graph.node_weights().cloned().collect()
    }

    /// Get nodes by type
    pub fn get_nodes_by_type(&self, node_type: &str) -> Vec<Concept> {
        let graph = self.graph.read().unwrap();
        graph
            .node_weights()
            .filter(|node| node.node_type == node_type)
            .cloned()
            .collect()
    }

    /// Get relationships between two nodes
    pub fn get_relationships(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
    ) -> Vec<ConceptRelationship> {
        let node_indices = self.node_indices.read().unwrap();
        let graph = self.graph.read().unwrap();

        if let (Some(source_idx), Some(target_idx)) =
            (node_indices.get(source_id), node_indices.get(target_id))
        {
            let mut relationships = Vec::new();

            for edge in graph.edges_directed(*source_idx, Direction::Outgoing) {
                if edge.target() == *target_idx {
                    if let Some(rel) = graph.edge_weight(edge.id()) {
                        relationships.push(rel.clone());
                    }
                }
            }

            relationships
        } else {
            Vec::new()
        }
    }

    /// Get all nodes connected to a source node
    pub fn get_connected_nodes(&self, source_id: &NodeId) -> Vec<(NodeId, ConceptRelationship)> {
        let node_indices = self.node_indices.read().unwrap();
        let graph = self.graph.read().unwrap();

        if let Some(source_idx) = node_indices.get(source_id) {
            let mut connected = Vec::new();

            for edge in graph.edges_directed(*source_idx, Direction::Outgoing) {
                let target_idx = edge.target();
                if let Some(target_node) = graph.node_weight(target_idx) {
                    if let Some(rel) = graph.edge_weight(edge.id()) {
                        connected.push((target_node.id.clone(), rel.clone()));
                    }
                }
            }

            connected
        } else {
            Vec::new()
        }
    }

    /// Check if the TIMELESS_ROOT concept exists
    pub fn has_timeless_root(&self) -> bool {
        self.get_node(&"TIMELESS_ROOT".to_string()).is_some()
    }
}

impl Default for LargeConceptNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::Op;

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new("test", "TestType")
            .with_property("name", "Test Concept")
            .with_property("value", 42);

        assert_eq!(concept.id, "test");
        assert_eq!(concept.node_type, "TestType");
        assert_eq!(concept.properties.len(), 2);
        assert_eq!(
            concept.get_property("name").unwrap().as_str().unwrap(),
            "Test Concept"
        );
        assert_eq!(concept.get_property("value").unwrap().as_i64().unwrap(), 42);
    }

    #[test]
    fn test_lcn_create_node() {
        let lcn = LargeConceptNetwork::new();

        // Create a node
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), serde_json::json!("Test Node"));

        let result = lcn.create_node(&"test".to_string(), "TestType", &properties);
        assert!(result.is_ok());

        // Verify node exists
        let node = lcn.get_node(&"test".to_string());
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.id, "test");
        assert_eq!(node.node_type, "TestType");
        assert_eq!(
            node.get_property("name").unwrap().as_str().unwrap(),
            "Test Node"
        );
    }

    #[test]
    fn test_lcn_create_relationship() {
        let lcn = LargeConceptNetwork::new();

        // Create nodes
        lcn.create_node(&"source".to_string(), "TestType", &HashMap::new())
            .unwrap();
        lcn.create_node(&"target".to_string(), "TestType", &HashMap::new())
            .unwrap();

        // Create relationship
        let mut properties = HashMap::new();
        properties.insert("weight".to_string(), serde_json::json!(0.5));

        let result = lcn.create_relationship(
            &"source".to_string(),
            &"target".to_string(),
            "RELATES_TO",
            &properties,
        );
        assert!(result.is_ok());

        // Verify relationship exists
        let relationships = lcn.get_relationships(&"source".to_string(), &"target".to_string());
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].rel_type, "RELATES_TO");
        assert_eq!(
            relationships[0]
                .properties
                .get("weight")
                .unwrap()
                .as_f64()
                .unwrap(),
            0.5
        );
    }

    #[test]
    fn test_lcn_apply_diff() {
        let lcn = LargeConceptNetwork::new();

        // Create a diff
        let mut diff = ConceptDiff::new(1);

        // Add operations
        diff.ops.push(Op::Create {
            node: "node1".to_string(),
            node_type: "TestType".to_string(),
            properties: HashMap::new(),
        });

        diff.ops.push(Op::Create {
            node: "node2".to_string(),
            node_type: "TestType".to_string(),
            properties: HashMap::new(),
        });

        diff.ops.push(Op::Link {
            source: "node1".to_string(),
            target: "node2".to_string(),
            rel_type: "CONNECTS_TO".to_string(),
            properties: HashMap::new(),
        });

        // Apply diff
        let result = lcn.apply_diff(&diff);
        assert!(result.is_ok());

        // Verify graph state
        assert!(lcn.get_node(&"node1".to_string()).is_some());
        assert!(lcn.get_node(&"node2".to_string()).is_some());
        assert_eq!(
            lcn.get_relationships(&"node1".to_string(), &"node2".to_string())
                .len(),
            1
        );
        assert_eq!(
            lcn.get_relationships(&"node1".to_string(), &"node2".to_string())[0].rel_type,
            "CONNECTS_TO"
        );
    }

    #[test]
    fn test_lcn_genesis() {
        let lcn = LargeConceptNetwork::new();
        assert!(!lcn.is_genesis_complete());

        // Create and apply GENESIS diff
        let diff = crate::diff::create_genesis_diff("TestCorpus");
        lcn.apply_diff(&diff).unwrap();

        // Check GENESIS state
        assert!(lcn.is_genesis_complete());
        assert!(lcn.has_timeless_root());

        // Check relationships
        let rels = lcn.get_relationships(&"TIMELESS_ROOT".to_string(), &"TestCorpus".to_string());
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].rel_type, "CONTAINS");
    }
}
