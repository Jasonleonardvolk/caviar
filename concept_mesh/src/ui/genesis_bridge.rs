use crate::mesh::MeshNode;
use crate::diff::*;
use std::sync::Arc;
//! GENESIS UI Bridge
//!
//! This module provides the bridge between the concept mesh and the UI
//! for handling GENESIS events and triggering the oscillator bloom animation.

use crate::diff::{ConceptDiff, ConceptDiffRef};
use crate::mesh::{Mesh, MeshNode, Pattern};

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// GENESIS event handler function type
pub type GenesisEventHandler = Box<dyn Fn() + Send + Sync + 'static>;

/// Message type for the UI bridge
#[derive(Debug)]
enum BridgeMessage {
    /// GENESIS event detected
    Genesis,

    /// Stop the bridge
    Stop,
}

/// The GENESIS UI bridge connects to the concept mesh and listens for GENESIS events
pub struct GenesisBridge {
    /// Mesh node for communicating with the concept mesh
    mesh_node: Arc<dyn MeshNode>,

    /// Command sender
    cmd_tx: Sender<BridgeMessage>,

    /// GENESIS event handler
    handler: Arc<Mutex<Option<GenesisEventHandler>>>,

    /// Background task handle
    background_task: Option<JoinHandle<()>>,

    /// Subscription IDs
    subscriptions: Arc<Mutex<Vec<usize>>>,
}

impl GenesisBridge {
    /// Create a new GENESIS UI bridge with the given mesh
    pub async fn new(id: impl Into<String>, mesh: Arc<Mesh>) -> Result<Self, String> {
        let id = format!("ui-genesis-{}", id.into());

        // Create mesh node
        let mesh_node = Arc::new(MeshNode::new(id, mesh).await?);

        // Create command channel
        let (cmd_tx, cmd_rx) = mpsc::channel(100);

        // Create handler
        let handler = Arc::new(Mutex::new(None));

        // Create subscriptions list
        let subscriptions = Arc::new(Mutex::new(Vec::new()));

        let bridge = Self {
            mesh_node,
            cmd_tx,
            handler,
            background_task: None,
            subscriptions,
        };

        // Start background task
        let background_task = bridge.start_background_task(cmd_rx);

        Ok(Self {
            background_task: Some(background_task),
            ..bridge
        })
    }

    /// Set the GENESIS event handler
    pub fn set_handler<F>(&self, handler: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        let mut handler_lock = self.handler.lock().unwrap();
        *handler_lock = Some(Box::new(handler));
    }

    /// Start the background task
    fn start_background_task(&self, mut cmd_rx: Receiver<BridgeMessage>) -> JoinHandle<()> {
        let mesh_node = Arc::clone(&self.mesh_node);
        let handler = Arc::clone(&self.handler);
        let subscriptions = Arc::clone(&self.subscriptions);
        let cmd_tx = self.cmd_tx.clone();

        tokio::spawn(async move {
            // Subscribe to GENESIS events
            let patterns = vec![
                Pattern::new("GENESIS_ACTIVATE"),
                Pattern::new("!Genesis"),
                Pattern::new("TIMELESS_ROOT"),
            ];

            if let Ok(sub_id) = mesh_node
                .subscribe(patterns, move |diff| {
                    Self::handle_genesis_diff(diff, cmd_tx.clone());
                })
                .await
            {
                subscriptions.lock().unwrap().push(sub_id);
            } else {
                error!("Failed to subscribe to GENESIS events");
            }

            info!("GENESIS UI bridge started");

            // Main loop
            let mut running = true;
            while running {
                if let Some(msg) = cmd_rx.recv().await {
                    match msg {
                        BridgeMessage::Genesis => {
                            info!("GENESIS event detected");
                            let handler_lock = handler.lock().unwrap();
                            if let Some(handler) = &*handler_lock {
                                handler();
                            }
                        }
                        BridgeMessage::Stop => {
                            info!("Stopping GENESIS UI bridge");
                            running = false;
                        }
                    }
                }
            }

            // Unsubscribe from mesh
            for sub_id in subscriptions.lock().unwrap().iter() {
                let _ = mesh_node.unsubscribe(*sub_id).await;
            }

            info!("GENESIS UI bridge stopped");
        })
    }

    /// Handle a ConceptDiff that might contain a GENESIS event
    async fn handle_genesis_diff(diff: ConceptDiffRef, cmd_tx: Sender<BridgeMessage>) {
        // Check if this is a GENESIS diff
        let is_genesis = diff.frame_id == crate::GENESIS_FRAME_ID;

        // Check for GENESIS_ACTIVATE signal
        let has_genesis_signal = diff.ops.iter().any(|op| {
            if let crate::diff::Op::Signal { event, .. } = op {
                event == "GENESIS_ACTIVATE"
            } else {
                false
            }
        });

        // Check for TIMELESS_ROOT creation
        let has_timeless_root = diff.ops.iter().any(|op| {
            if let crate::diff::Op::Create { node, .. } = op {
                node == "TIMELESS_ROOT"
            } else {
                false
            }
        });

        if is_genesis || has_genesis_signal || has_timeless_root {
            // Send Genesis message
            let _ = cmd_tx.send(BridgeMessage::Genesis).await;
        }
    }

    /// Stop the bridge
    pub async fn stop(&self) -> Result<(), String> {
        let _ = self.cmd_tx.send(BridgeMessage::Stop).await;
        Ok(())
    }
}

impl Drop for GenesisBridge {
    fn drop(&mut self) {
        // Cancel the background task if it's still running
        if let Some(task) = self.background_task.take() {
            task.abort();
        }
    }
}

/// The OscillatorBloom handles the animation for the GENESIS event
pub struct OscillatorBloom {
    /// Whether the bloom has been triggered
    triggered: bool,

    /// Timestamp when the bloom was triggered
    trigger_time: Option<std::time::Instant>,

    /// Animation duration in milliseconds
    duration_ms: u64,

    /// Current animation frame
    frame: usize,

    /// Total number of animation frames
    total_frames: usize,
}

impl OscillatorBloom {
    /// Create a new OscillatorBloom
    pub fn new(duration_ms: u64, total_frames: usize) -> Self {
        Self {
            triggered: false,
            trigger_time: None,
            duration_ms,
            frame: 0,
            total_frames,
        }
    }

    /// Trigger the bloom animation
    pub fn trigger(&mut self) {
        self.triggered = true;
        self.trigger_time = Some(std::time::Instant::now());
        self.frame = 0;
    }

    /// Check if the bloom is active
    pub fn is_active(&self) -> bool {
        if !self.triggered || self.trigger_time.is_none() {
            return false;
        }

        let elapsed = self.trigger_time.unwrap().elapsed();
        elapsed.as_millis() < self.duration_ms as u128
    }

    /// Reset the bloom
    pub fn reset(&mut self) {
        self.triggered = false;
        self.trigger_time = None;
        self.frame = 0;
    }

    /// Update the animation state
    pub fn update(&mut self) -> Option<f32> {
        if !self.is_active() {
            return None;
        }

        let elapsed = self.trigger_time.unwrap().elapsed();
        let progress = (elapsed.as_millis() as f32) / (self.duration_ms as f32);
        let progress = progress.clamp(0.0, 1.0);

        // Calculate current frame
        let frame = (progress * (self.total_frames as f32)) as usize;
        self.frame = frame.min(self.total_frames - 1);

        Some(progress)
    }

    /// Get the current animation frame
    pub fn current_frame(&self) -> usize {
        self.frame
    }

    /// Get the animation progress (0.0 - 1.0)
    pub fn progress(&self) -> Option<f32> {
        if !self.is_active() {
            return None;
        }

        let elapsed = self.trigger_time.unwrap().elapsed();
        let progress = (elapsed.as_millis() as f32) / (self.duration_ms as f32);
        Some(progress.clamp(0.0, 1.0))
    }
}

/// Create a new GENESIS UI bridge and connect it to an OscillatorBloom
pub async fn create_genesis_ui_bridge(
    id: impl Into<String>,
    mesh: Arc<Mesh>,
    bloom: Arc<Mutex<OscillatorBloom>>,
) -> Result<Arc<GenesisBridge>, String> {
    let bridge = GenesisBridge::new(id, mesh).await?;

    // Set handler to trigger the bloom
    let bloom_clone = Arc::clone(&bloom);
    bridge.set_handler(move || {
        let mut bloom = bloom_clone.lock().unwrap();
        bloom.trigger();
    });

    Ok(Arc::new(bridge))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{ConceptDiffBuilder, Op};
    use crate::mesh::InMemoryMesh;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn test_genesis_bridge() {
        // Create mesh
        let mesh = Arc::new(InMemoryMesh::new());

        // Create flag to track if handler was called
        let handler_called = Arc::new(AtomicBool::new(false));

        // Create bridge
        let bridge = GenesisBridge::new("test", Arc::clone(&mesh)).await.unwrap();

        // Set handler
        let handler_called_clone = Arc::clone(&handler_called);
        bridge.set_handler(move || {
            handler_called_clone.store(true, Ordering::SeqCst);
        });

        // Create a GENESIS diff
        let genesis_diff = crate::diff::create_genesis_diff("TestCorpus");

        // Register a test node
        let node = MeshNode::new("test_node", Arc::clone(&mesh)).await.unwrap();

        // Publish the diff
        node.publish(genesis_diff).await.unwrap();

        // Wait a bit for the handler to be called
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check if handler was called
        assert!(handler_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_oscillator_bloom() {
        // Create bloom
        let mut bloom = OscillatorBloom::new(1000, 60);

        // Initially not active
        assert!(!bloom.is_active());
        assert_eq!(bloom.progress(), None);

        // Trigger
        bloom.trigger();

        // Now active
        assert!(bloom.is_active());
        assert!(bloom.progress().is_some());

        // Reset
        bloom.reset();

        // Back to inactive
        assert!(!bloom.is_active());
        assert_eq!(bloom.progress(), None);
    }
}

