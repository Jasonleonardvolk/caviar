use crate::mesh::MeshNode;
use crate::diff::*;
use crate::mesh::phase_event_bus::create_phase_event_bus;
//! Mesh module for the concept mesh
//!
//! This module provides the core communication infrastructure for the concept mesh,
//! enabling agents and components to exchange messages and maintain phase coherence.

use async_trait::async_trait;
use std::sync::Arc;

pub mod phase_event_bus;

pub use phase_event_bus::{
    PhaseAgent, PhaseEvent, PhaseEventBus, PhaseEventData, PhaseEventType, SubscriptionPattern,
};

/// Mesh node trait representing a component in the concept mesh
#[async_trait]
pub trait MeshNode: Send + Sync {
    /// Get the node's ID
    fn id(&self) -> &str;

    /// Initialize the node
    async fn initialize(&mut self) -> Result<(), String>;

    /// Start the node
    async fn start(&mut self) -> Result<(), String>;

    /// Stop the node
    async fn stop(&mut self) -> Result<(), String>;

    /// Check if the node is running
    fn is_running(&self) -> bool;
}

/// In-memory mesh implementation
pub struct InMemoryMesh {
    /// Nodes in the mesh
    nodes: Vec<Arc<dyn MeshNode>>,

    /// Event bus
    event_bus: Option<Arc<PhaseEventBus>>,
}

impl InMemoryMesh {
    /// Create a new in-memory mesh
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            event_bus: None,
        }
    }

    /// Initialize the mesh with an event bus
    pub async fn initialize(&mut self, bus_id: &str) -> Result<(), String> {
        // Create the event bus if it doesn't exist
        if self.event_bus.is_none() {
            let bus = Arc::new(create_phase_event_bus(bus_id));
            self.event_bus = Some(bus);
        }

        Ok(())
    }

    /// Add a node to the mesh
    pub fn add_node(&mut self, node: Arc<dyn MeshNode>) {
        self.nodes.push(node);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<Arc<dyn MeshNode>> {
        self.nodes.iter().find(|n| n.id() == id).cloned()
    }

    /// Get all nodes
    pub fn get_all_nodes(&self) -> Vec<Arc<dyn MeshNode>> {
        self.nodes.clone()
    }

    /// Start all nodes
    pub async fn start_all(&mut self) -> Result<(), String> {
        for node in &mut self.nodes {
            // Clone the node Arc for moving into the async block
            let mut node_clone = Arc::clone(node);

            // Start the node
            // We use a block here to allow mutable access to the node
            let result = {
                let node = Arc::get_mut(&mut node_clone).ok_or_else(|| {
                    format!("Failed to get mutable reference to node {}", node.id())
                })?;

                node.start().await
            };

            // Handle errors
            if let Err(e) = result {
                return Err(format!("Failed to start node {}: {}", node.id(), e));
            }
        }

        Ok(())
    }

    /// Stop all nodes
    pub async fn stop_all(&mut self) -> Result<(), String> {
        for node in &mut self.nodes {
            // Clone the node Arc for moving into the async block
            let mut node_clone = Arc::clone(node);

            // Stop the node
            // We use a block here to allow mutable access to the node
            let result = {
                let node = Arc::get_mut(&mut node_clone).ok_or_else(|| {
                    format!("Failed to get mutable reference to node {}", node.id())
                })?;

                node.stop().await
            };

            // Handle errors
            if let Err(e) = result {
                return Err(format!("Failed to stop node {}: {}", node.id(), e));
            }
        }

        // Also stop the event bus if it exists
        if let Some(bus) = &self.event_bus {
            bus.stop().await?;
        }

        Ok(())
    }

    /// Get the event bus
    pub fn get_event_bus(&self) -> Option<Arc<PhaseEventBus>> {
        self.event_bus.clone()
    }

    /// Check if the mesh has genesis completed
    pub fn is_genesis_complete(&self) -> bool {
        // In an actual implementation, this would check if the GENESIS event has been processed
        // and if the required concepts (like TIMELESS_ROOT) exist

        // For now, we'll just return true if we have at least one node
        !self.nodes.is_empty()
    }

    /// Check if the mesh has a timeless root concept
    pub fn has_timeless_root(&self) -> bool {
        // In an actual implementation, this would check if the TIMELESS_ROOT concept exists
        // in the concept graph

        // For now, we'll just return true if we have at least one node
        !self.nodes.is_empty()
    }
}

impl Default for InMemoryMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Mock MeshNode for testing
    struct MockNode {
        id: String,
        running: AtomicBool,
    }

    impl MockNode {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                running: AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl MeshNode for MockNode {
        fn id(&self) -> &str {
            &self.id
        }

        async fn initialize(&mut self) -> Result<(), String> {
            Ok(())
        }

        async fn start(&mut self) -> Result<(), String> {
            self.running.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn stop(&mut self) -> Result<(), String> {
            self.running.store(false, Ordering::SeqCst);
            Ok(())
        }

        fn is_running(&self) -> bool {
            self.running.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn test_in_memory_mesh() {
        // Create mesh
        let mut mesh = InMemoryMesh::new();

        // Add some nodes
        let node1 = Arc::new(MockNode::new("node1"));
        let node2 = Arc::new(MockNode::new("node2"));

        mesh.add_node(Arc::clone(&node1));
        mesh.add_node(Arc::clone(&node2));

        // Initialize mesh with event bus
        mesh.initialize("test_bus").await.unwrap();

        // Verify nodes were added
        assert_eq!(mesh.get_all_nodes().len(), 2);

        // Get node by ID
        let found_node = mesh.get_node("node1").unwrap();
        assert_eq!(found_node.id(), "node1");

        // Start all nodes
        mesh.start_all().await.unwrap();

        // Verify nodes are running
        assert!(node1.is_running());
        assert!(node2.is_running());

        // Stop all nodes
        mesh.stop_all().await.unwrap();

        // Verify nodes are stopped
        assert!(!node1.is_running());
        assert!(!node2.is_running());
    }
}

// Stub implementations for missing types
pub struct Mesh {
    pub nodes: Vec<Box<dyn MeshNode>>,
}

pub struct Pattern {
    pub name: String,
}

pub trait MeshTrait: Send + Sync {
    fn node_count(&self) -> usize;
}

impl MeshTrait for Mesh {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}



