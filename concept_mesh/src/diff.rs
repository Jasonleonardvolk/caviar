//! ConceptDiff Module
//!
//! Defines the ConceptDiff data structure, which represents a graph change operation
//! and is the primary communication mechanism in the Concept Mesh.

pub mod user_context;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a node in the concept graph
pub type NodeId = String;

/// Frame ID for a ConceptDiff transaction
pub type FrameId = u64;

/// ConceptDiff represents a set of operations to be applied to the concept graph
///
/// It is the primary message format for communication between agents in the mesh.
/// Each ConceptDiff contains one or more operations (Ops) that describe changes
/// to make to the concept graph or signals to other agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptDiff {
    /// Unique identifier for this frame of operations
    pub frame_id: FrameId,

    /// Source node/agent that generated this diff
    pub source: Option<String>,

    /// Target nodes/agents that should process this diff
    pub targets: Option<Vec<String>>,

    /// Timestamp in milliseconds when this diff was created
    pub timestamp_ms: u64,

    /// Operations to be applied
    pub ops: Vec<Op>,

    /// Additional metadata for this diff
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A single operation in a ConceptDiff
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Op {
    /// Create a new node in the graph
    Create {
        /// ID of the node to create
        node: NodeId,
        /// Type of node to create
        node_type: String,
        /// Properties for the new node
        #[serde(default)]
        properties: HashMap<String, serde_json::Value>,
    },

    /// Delete a node from the graph
    Delete {
        /// ID of the node to delete
        node: NodeId,
    },

    /// Update a node's properties
    Update {
        /// ID of the node to update
        node: NodeId,
        /// Properties to update (only specified properties are modified)
        properties: HashMap<String, serde_json::Value>,
    },

    /// Create an edge between two nodes
    Link {
        /// Source node ID
        source: NodeId,
        /// Target node ID
        target: NodeId,
        /// Type of relationship
        rel_type: String,
        /// Properties for the relationship
        #[serde(default)]
        properties: HashMap<String, serde_json::Value>,
    },

    /// Remove an edge between two nodes
    Unlink {
        /// Source node ID
        source: NodeId,
        /// Target node ID
        target: NodeId,
        /// Type of relationship (if None, removes all edges between source and target)
        rel_type: Option<String>,
    },

    /// Bind a node to a parent (specialized Link operation for hierarchical relationships)
    Bind {
        /// Parent node ID
        parent: NodeId,
        /// Child node ID
        node: NodeId,
        /// Optional binding type (defaults to "CONTAINS")
        #[serde(default = "default_bind_type")]
        bind_type: String,
    },

    /// Signal a special event to listeners
    Signal {
        /// Event type
        event: String,
        /// Target for the signal (if None, broadcast to all)
        target: Option<String>,
        /// Event payload
        #[serde(default)]
        payload: HashMap<String, serde_json::Value>,
    },

    /// Execute a domain-specific operation (extensible)
    Execute {
        /// Operation name
        operation: String,
        /// Parameters for the operation
        #[serde(default)]
        params: HashMap<String, serde_json::Value>,
    },
}

/// Default binding type for Bind operations
fn default_bind_type() -> String {
    "CONTAINS".to_string()
}

impl ConceptDiff {
    /// Create a new ConceptDiff with the given frame ID
    pub fn new(frame_id: FrameId) -> Self {
        Self {
            frame_id,
            source: None,
            targets: None,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ops: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new GENESIS diff
    pub fn genesis(corpus_id: &str) -> Self {
        let mut diff = Self::new(crate::GENESIS_FRAME_ID);
        diff.ops.push(Op::Create {
            node: "TIMELESS_ROOT".to_string(),
            node_type: "Concept".to_string(),
            properties: HashMap::new(),
        });
        diff.ops.push(Op::Bind {
            parent: "TIMELESS_ROOT".to_string(),
            node: corpus_id.to_string(),
            bind_type: "CONTAINS".to_string(),
        });
        diff.ops.push(Op::Signal {
            event: "GENESIS_ACTIVATE".to_string(),
            target: Some("UI".to_string()),
            payload: HashMap::new(),
        });
        diff
    }

    /// Add an operation to this diff
    pub fn add_op(&mut self, op: Op) -> &mut Self {
        self.ops.push(op);
        self
    }

    /// Set the source of this diff
    pub fn from_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the targets for this diff
    pub fn to_targets(mut self, targets: Vec<String>) -> Self {
        self.targets = Some(targets);
        self
    }

    /// Add metadata to this diff
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), value);
        }
        self
    }
}

/// Immutable reference to a ConceptDiff
pub type ConceptDiffRef = Arc<ConceptDiff>;

/// Builder for constructing ConceptDiffs in a fluent style
pub struct ConceptDiffBuilder {
    inner: ConceptDiff,
}

// Re-export user context extension trait
pub use user_context::UserContextExt;

impl ConceptDiffBuilder {
    /// Create a new ConceptDiffBuilder with the given frame ID
    pub fn new(frame_id: FrameId) -> Self {
        Self {
            inner: ConceptDiff::new(frame_id),
        }
    }

    /// Add a Create operation
    pub fn create(
        mut self,
        node: impl Into<String>,
        node_type: impl Into<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.inner.ops.push(Op::Create {
            node: node.into(),
            node_type: node_type.into(),
            properties,
        });
        self
    }

    /// Add a Bind operation
    pub fn bind(
        mut self,
        parent: impl Into<String>,
        node: impl Into<String>,
        bind_type: Option<String>,
    ) -> Self {
        self.inner.ops.push(Op::Bind {
            parent: parent.into(),
            node: node.into(),
            bind_type: bind_type.unwrap_or_else(default_bind_type),
        });
        self
    }

    /// Add a Signal operation
    pub fn signal(
        mut self,
        event: impl Into<String>,
        target: Option<String>,
        payload: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.inner.ops.push(Op::Signal {
            event: event.into(),
            target,
            payload,
        });
        self
    }

    /// Set the source of this diff
    pub fn from_source(mut self, source: impl Into<String>) -> Self {
        self.inner.source = Some(source.into());
        self
    }

    /// Set the targets for this diff
    pub fn to_targets(mut self, targets: Vec<String>) -> Self {
        self.inner.targets = Some(targets);
        self
    }

    /// Add metadata to this diff
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.inner.metadata.insert(key.into(), value);
        }
        self
    }

    /// Build the ConceptDiff
    pub fn build(self) -> ConceptDiff {
        self.inner
    }

    /// Build and wrap in an Arc
    pub fn build_ref(self) -> ConceptDiffRef {
        Arc::new(self.inner)
    }
}

/// Helper function to create a GENESIS ConceptDiff
pub fn create_genesis_diff(corpus_id: &str) -> ConceptDiff {
    ConceptDiff::genesis(corpus_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conceptdiff_creation() {
        let diff = ConceptDiff::new(1);
        assert_eq!(diff.frame_id, 1);
        assert!(diff.ops.is_empty());
    }

    #[test]
    fn test_genesis_diff() {
        let diff = create_genesis_diff("TestCorpus");
        assert_eq!(diff.frame_id, crate::GENESIS_FRAME_ID);
        assert_eq!(diff.ops.len(), 3);

        // Verify first op is Create TIMELESS_ROOT
        if let Op::Create {
            node, node_type, ..
        } = &diff.ops[0]
        {
            assert_eq!(node, "TIMELESS_ROOT");
            assert_eq!(node_type, "Concept");
        } else {
            panic!("First op should be Create");
        }

        // Verify second op is Bind corpus to TIMELESS_ROOT
        if let Op::Bind { parent, node, .. } = &diff.ops[1] {
            assert_eq!(parent, "TIMELESS_ROOT");
            assert_eq!(node, "TestCorpus");
        } else {
            panic!("Second op should be Bind");
        }

        // Verify third op is Signal GENESIS_ACTIVATE
        if let Op::Signal { event, target, .. } = &diff.ops[2] {
            assert_eq!(event, "GENESIS_ACTIVATE");
            assert_eq!(target, &Some("UI".to_string()));
        } else {
            panic!("Third op should be Signal");
        }
    }

    #[test]
    fn test_builder_pattern() {
        let diff = ConceptDiffBuilder::new(42)
            .create("node1", "TestNode", HashMap::new())
            .bind("parent1", "node1", None)
            .from_source("TestAgent")
            .build();

        assert_eq!(diff.frame_id, 42);
        assert_eq!(diff.source, Some("TestAgent".to_string()));
        assert_eq!(diff.ops.len(), 2);
    }
}
