//! Concept Mesh Communication Layer
//!
//! This module provides the core communication infrastructure for the concept-agent mesh,
//! implementing a pub/sub system that allows agents to communicate via ConceptDiffs.

use crate::diff::{ConceptDiff, ConceptDiffRef};
use async_trait::async_trait;
use futures::stream::Stream;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinHandle;

/// Mesh node identifier
pub type NodeId = String;

/// A subscription pattern for filtering ConceptDiffs
///
/// Patterns can include wildcards, e.g., "MATH.*" to subscribe to all math-related concepts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pattern(pub String);

impl Pattern {
    /// Create a new subscription pattern
    pub fn new(pattern: impl Into<String>) -> Self {
        Self(pattern.into())
    }
    
    /// Check if this pattern matches the given concept
    pub fn matches(&self, concept: &str) -> bool {
        if self.0 == "*" {
            return true; // Wildcard pattern matches everything
        }
        
        // Simple glob pattern matching
        if self.0.ends_with(".*") {
            let prefix = &self.0[..self.0.len() - 2];
            return concept.starts_with(prefix);
        }
        
        // Exact match
        self.0 == concept
    }
    
    /// Check if this pattern matches any of the concepts in the given diff
    pub fn matches_diff(&self, diff: &ConceptDiff) -> bool {
        // 1. Check if we want all diffs
        if self.0 == "*" {
            return true;
        }
        
        // 2. Check if we match the source
        if let Some(ref source) = diff.source {
            if self.matches(source) {
                return true;
            }
        }
        
        // 3. Check if we're in the diff's targets
        if let Some(ref targets) = diff.targets {
            for target in targets {
                if self.matches(target) {
                    return true;
                }
            }
        }
        
        // 4. Check for operations involving matching concepts
        for op in &diff.ops {
            match op {
                crate::diff::Op::Create { node, .. } |
                crate::diff::Op::Delete { node, .. } |
                crate::diff::Op::Update { node, .. } => {
                    if self.matches(node) {
                        return true;
                    }
                },
                crate::diff::Op::Link { source, target, .. } |
                crate::diff::Op::Unlink { source, target, .. } => {
                    if self.matches(source) || self.matches(target) {
                        return true;
                    }
                },
                crate::diff::Op::Bind { parent, node, .. } => {
                    if self.matches(parent) || self.matches(node) {
                        return true;
                    }
                },
                crate::diff::Op::Signal { event, target, .. } => {
                    if self.matches(event) || target.as_ref().map_or(false, |t| self.matches(t)) {
                        return true;
                    }
                },
                crate::diff::Op::Execute { operation, .. } => {
                    if self.matches(operation) {
                        return true;
                    }
                },
            }
        }
        
        // 5. Check for special patterns in the metadata
        for (key, value) in &diff.metadata {
            if self.matches(key) {
                return true;
            }
            
            if let Some(value_str) = value.as_str() {
                if self.matches(value_str) {
                    return true;
                }
            }
        }
        
        false
    }
}

/// Subscription ID
pub type SubscriptionId = usize;

/// A subscription to ConceptDiffs in the mesh
#[derive(Debug)]
pub struct Subscription {
    /// Unique ID for this subscription
    id: SubscriptionId,
    
    /// Node that owns this subscription
    node_id: NodeId,
    
    /// Patterns that this subscription is interested in
    patterns: Vec<Pattern>,
    
    /// Sender to push matching diffs to
    sender: Sender<ConceptDiffRef>,
}

impl Subscription {
    /// Create a new subscription
    fn new(
        id: SubscriptionId,
        node_id: NodeId,
        patterns: Vec<Pattern>,
        sender: Sender<ConceptDiffRef>,
    ) -> Self {
        Self {
            id,
            node_id,
            patterns,
            sender,
        }
    }
    
    /// Check if this subscription is interested in the given diff
    fn is_interested(&self, diff: &ConceptDiff) -> bool {
        self.patterns.iter().any(|pattern| pattern.matches_diff(diff))
    }
    
    /// Send a diff to the subscriber
    async fn send(&self, diff: ConceptDiffRef) -> Result<(), tokio::sync::mpsc::error::SendError<ConceptDiffRef>> {
        self.sender.send(diff).await
    }
}

/// In-memory implementation of the Mesh
pub struct InMemoryMesh {
    /// Next subscription ID to assign
    next_sub_id: Arc<Mutex<SubscriptionId>>,
    
    /// Active subscriptions
    subscriptions: Arc<RwLock<Vec<Subscription>>>,
    
    /// Nodes currently connected to the mesh
    nodes: Arc<RwLock<HashSet<NodeId>>>,
    
    /// Handle for the mesh's background task
    _background_task: Option<JoinHandle<()>>,
}

impl InMemoryMesh {
    /// Create a new in-memory mesh
    pub fn new() -> Self {
        Self {
            next_sub_id: Arc::new(Mutex::new(1)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            nodes: Arc::new(RwLock::new(HashSet::new())),
            _background_task: None,
        }
    }
    
    /// Get the next subscription ID
    fn next_sub_id(&self) -> SubscriptionId {
        let mut id = self.next_sub_id.lock().unwrap();
        let result = *id;
        *id += 1;
        result
    }
}

/// The Mesh trait defines the interface for concept-agent mesh communication
#[async_trait]
pub trait Mesh: Send + Sync + 'static {
    /// Register a node with the mesh
    async fn register_node(&self, node_id: impl Into<NodeId> + Send) -> Result<(), String>;
    
    /// Unregister a node from the mesh
    async fn unregister_node(&self, node_id: &str) -> Result<(), String>;
    
    /// Subscribe to ConceptDiffs matching the given patterns
    async fn subscribe(
        &self,
        node_id: impl Into<NodeId> + Send,
        patterns: Vec<Pattern>,
    ) -> Result<(SubscriptionId, impl Stream<Item = ConceptDiffRef>), String>;
    
    /// Unsubscribe from the mesh
    async fn unsubscribe(&self, sub_id: SubscriptionId) -> Result<(), String>;
    
    /// Publish a ConceptDiff to the mesh
    async fn publish(&self, diff: ConceptDiff) -> Result<(), String>;
    
    /// Check if a node is registered with the mesh
    async fn is_registered(&self, node_id: &str) -> bool;
    
    /// Get all registered nodes
    async fn get_nodes(&self) -> Vec<NodeId>;
    
    /// Get all active subscriptions for a node
    async fn get_subscriptions(&self, node_id: &str) -> Vec<SubscriptionId>;
}

#[async_trait]
impl Mesh for InMemoryMesh {
    async fn register_node(&self, node_id: impl Into<NodeId> + Send) -> Result<(), String> {
        let node_id = node_id.into();
        let mut nodes = self.nodes.write().unwrap();
        
        if nodes.contains(&node_id) {
            return Err(format!("Node {} already registered", node_id));
        }
        
        nodes.insert(node_id);
        Ok(())
    }
    
    async fn unregister_node(&self, node_id: &str) -> Result<(), String> {
        let mut nodes = self.nodes.write().unwrap();
        
        if !nodes.remove(node_id) {
            return Err(format!("Node {} not registered", node_id));
        }
        
        // Remove all subscriptions for this node
        let mut subs = self.subscriptions.write().unwrap();
        subs.retain(|sub| sub.node_id != node_id);
        
        Ok(())
    }
    
    async fn subscribe(
        &self,
        node_id: impl Into<NodeId> + Send,
        patterns: Vec<Pattern>,
    ) -> Result<(SubscriptionId, impl Stream<Item = ConceptDiffRef>), String> {
        let node_id = node_id.into();
        
        // Check if node is registered
        let nodes = self.nodes.read().unwrap();
        if !nodes.contains(&node_id) {
            return Err(format!("Node {} not registered", node_id));
        }
        drop(nodes);
        
        // Create channel for sending diffs to the subscriber
        let (tx, rx) = mpsc::channel(100); // Buffer up to 100 diffs
        
        // Create subscription
        let sub_id = self.next_sub_id();
        let sub = Subscription::new(sub_id, node_id, patterns, tx);
        
        // Add to subscriptions
        self.subscriptions.write().unwrap().push(sub);
        
        Ok((sub_id, rx))
    }
    
    async fn unsubscribe(&self, sub_id: SubscriptionId) -> Result<(), String> {
        let mut subs = self.subscriptions.write().unwrap();
        let idx = subs.iter().position(|sub| sub.id == sub_id);
        
        match idx {
            Some(idx) => {
                subs.remove(idx);
                Ok(())
            }
            None => Err(format!("Subscription {} not found", sub_id)),
        }
    }
    
    async fn publish(&self, diff: ConceptDiff) -> Result<(), String> {
        let diff_ref = Arc::new(diff);
        let subs = self.subscriptions.read().unwrap();
        
        // Collect interested subscriptions
        let interested_subs: Vec<_> = subs
            .iter()
            .filter(|sub| sub.is_interested(&diff_ref))
            .collect();
        
        // Send to all interested subscribers
        for sub in interested_subs {
            if let Err(e) = sub.send(Arc::clone(&diff_ref)).await {
                eprintln!("Failed to send diff to subscription {}: {:?}", sub.id, e);
            }
        }
        
        Ok(())
    }
    
    async fn is_registered(&self, node_id: &str) -> bool {
        let nodes = self.nodes.read().unwrap();
        nodes.contains(node_id)
    }
    
    async fn get_nodes(&self) -> Vec<NodeId> {
        let nodes = self.nodes.read().unwrap();
        nodes.iter().cloned().collect()
    }
    
    async fn get_subscriptions(&self, node_id: &str) -> Vec<SubscriptionId> {
        let subs = self.subscriptions.read().unwrap();
        subs.iter()
            .filter(|sub| sub.node_id == node_id)
            .map(|sub| sub.id)
            .collect()
    }
}

/// A node in the concept-agent mesh
pub struct MeshNode {
    /// Unique identifier for this node
    id: NodeId,
    
    /// Reference to the mesh this node is connected to
    mesh: Arc<dyn Mesh>,
    
    /// Active subscriptions
    subscriptions: Mutex<HashMap<SubscriptionId, JoinHandle<()>>>,
}

impl MeshNode {
    /// Create a new mesh node
    pub async fn new(id: impl Into<NodeId>, mesh: Arc<dyn Mesh>) -> Result<Self, String> {
        let id = id.into();
        mesh.register_node(id.clone()).await?;
        
        Ok(Self {
            id,
            mesh,
            subscriptions: Mutex::new(HashMap::new()),
        })
    }
    
    /// Get the node's ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Subscribe to ConceptDiffs matching the given patterns
    pub async fn subscribe<F, Fut>(
        &self,
        patterns: Vec<Pattern>,
        handler: F,
    ) -> Result<SubscriptionId, String>
    where
        F: FnMut(ConceptDiffRef) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (sub_id, mut stream) = self.mesh.subscribe(self.id.clone(), patterns).await?;
        
        // Spawn a task to handle incoming diffs
        let handle = tokio::spawn(async move {
            let mut handler = handler;
            while let Some(diff) = stream.recv().await {
                handler(diff).await;
            }
        });
        
        // Store the subscription
        self.subscriptions.lock().unwrap().insert(sub_id, handle);
        
        Ok(sub_id)
    }
    
    /// Unsubscribe from the mesh
    pub async fn unsubscribe(&self, sub_id: SubscriptionId) -> Result<(), String> {
        // Remove from mesh
        self.mesh.unsubscribe(sub_id).await?;
        
        // Cancel the task
        let mut subs = self.subscriptions.lock().unwrap();
        if let Some(handle) = subs.remove(&sub_id) {
            handle.abort();
        }
        
        Ok(())
    }
    
    /// Publish a ConceptDiff to the mesh
    pub async fn publish(&self, mut diff: ConceptDiff) -> Result<(), String> {
        // Set source if not already set
        if diff.source.is_none() {
            diff.source = Some(self.id.clone());
        }
        
        self.mesh.publish(diff).await
    }
    
    /// Disconnect from the mesh
    pub async fn disconnect(&self) -> Result<(), String> {
        // Cancel all subscription handlers
        {
            let mut subs = self.subscriptions.lock().unwrap();
            for (_, handle) in subs.drain() {
                handle.abort();
            }
        }
        
        // Unregister from mesh
        self.mesh.unregister_node(&self.id).await
    }
}

impl Drop for MeshNode {
    fn drop(&mut self) {
        // Cancel all subscription handlers
        let mut subs = self.subscriptions.lock().unwrap();
        for (_, handle) in subs.drain() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::Op;
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    
    #[test]
    fn test_pattern_matching() {
        let pattern = Pattern::new("MATH.*");
        assert!(pattern.matches("MATH.ALGEBRA"));
        assert!(pattern.matches("MATH.CALCULUS"));
        assert!(!pattern.matches("PHYSICS"));
        
        let wildcard = Pattern::new("*");
        assert!(wildcard.matches("ANYTHING"));
        
        let exact = Pattern::new("EXACT");
        assert!(exact.matches("EXACT"));
        assert!(!exact.matches("NOT_EXACT"));
    }
    
    #[test]
    fn test_pattern_matches_diff() {
        let mut diff = ConceptDiff::new(1);
        diff.source = Some("TestSource".to_string());
        diff.ops.push(Op::Create {
            node: "MATH.ALGEBRA".to_string(),
            node_type: "Concept".to_string(),
            properties: HashMap::new(),
        });
        
        // Pattern matching source
        let source_pattern = Pattern::new("TestSource");
        assert!(source_pattern.matches_diff(&diff));
        
        // Pattern matching concept in op
        let concept_pattern = Pattern::new("MATH.*");
        assert!(concept_pattern.matches_diff(&diff));
        
        // Pattern not matching anything
        let non_matching = Pattern::new("PHYSICS");
        assert!(!non_matching.matches_diff(&diff));
    }
    
    #[test]
    fn test_in_memory_mesh() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create mesh
            let mesh = Arc::new(InMemoryMesh::new());
            
            // Register nodes
            mesh.register_node("node1").await.unwrap();
            mesh.register_node("node2").await.unwrap();
            
            // Subscribe node1 to MATH.*
            let (sub_id, mut stream) = mesh.subscribe("node1", vec![Pattern::new("MATH.*")]).await.unwrap();
            
            // Create a diff that matches the subscription
            let mut diff = ConceptDiff::new(1);
            diff.ops.push(Op::Create {
                node: "MATH.ALGEBRA".to_string(),
                node_type: "Concept".to_string(),
                properties: HashMap::new(),
            });
            
            // Publish diff
            mesh.publish(diff).await.unwrap();
            
            // Check that node1 received the diff
            let received = tokio::time::timeout(std::time::Duration::from_millis(100), stream.recv()).await.unwrap();
            assert!(received.is_some());
            let received = received.unwrap();
            assert_eq!(received.frame_id, 1);
            
            // Unsubscribe
            mesh.unsubscribe(sub_id).await.unwrap();
            
            // Unregister nodes
            mesh.unregister_node("node1").await.unwrap();
            mesh.unregister_node("node2").await.unwrap();
        });
    }
    
    #[test]
    fn test_mesh_node() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create mesh
            let mesh = Arc::new(InMemoryMesh::new());
            
            // Create nodes
            let node1 = MeshNode::new("node1", Arc::clone(&mesh)).await.unwrap();
            let node2 = MeshNode::new("node2", Arc::clone(&mesh)).await.unwrap();
            
            // Subscribe node1 to MATH.*
            let received = Arc::new(Mutex::new(Vec::new()));
            let received_clone = Arc::clone(&received);
            let sub_id = node1.subscribe(
                vec![Pattern::new("MATH.*")],
                move |diff| {
                    let received = Arc::clone(&received_clone);
                    async move {
                        received.lock().unwrap().push(diff);
                    }
                },
            ).await.unwrap();
            
            // Create a diff that matches the subscription
            let mut diff = ConceptDiff::new(1);
            diff.ops.push(Op::Create {
                node: "MATH.ALGEBRA".to_string(),
                node_type: "Concept".to_string(),
                properties: HashMap::new(),
            });
            
            // Publish diff from node2
            node2.publish(diff).await.unwrap();
            
            // Give some time for the diff to be processed
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            
            // Check that node1 received the diff
            let received = received.lock().unwrap();
            assert_eq!(received.len(), 1);
            assert_eq!(received[0].frame_id, 1);
            
            // Unsubscribe
            node1.unsubscribe(sub_id).await.unwrap();
            
            // Disconnect nodes
            node1.disconnect().await.unwrap();
            node2.disconnect().await.unwrap();
        });
    }
}
