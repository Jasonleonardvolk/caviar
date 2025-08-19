use crate::mesh::MeshNode;
use crate::diff::*;
//! CLI Adapters for Concept Mesh
//!
//! This module provides CLI adapters and shims to bridge the gap between
//! traditional command-line tools and the concept-agent mesh architecture.
//! It allows for backward compatibility with existing workflows and scripts.

use crate::agents::orchestrator::{IngestDocument, IngestStatus, Orchestrator};
use crate::diff::{ConceptDiff, ConceptDiffBuilder};
use crate::mesh::{Mesh, MeshNode, Pattern};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// The CLI adapter provides a command-line interface to the concept mesh
pub struct CliAdapter {
    /// Orchestrator for the mesh
    orchestrator: Arc<Orchestrator>,

    /// Mesh node for direct mesh communication
    mesh_node: Arc<dyn MeshNode>,

    /// Background task handle
    background_task: Option<JoinHandle<()>>,

    /// Command sender
    cmd_tx: Sender<CliCommand>,

    /// Whether the adapter is running
    running: bool,
}

/// Command for the CLI adapter
#[derive(Debug)]
enum CliCommand {
    /// Queue a document for ingestion
    QueueDocument(PathBuf, Sender<Result<String, String>>),

    /// Get the status of an ingest job
    GetJobStatus(String, Sender<Result<String, String>>),

    /// Get metrics about the ingest system
    GetMetrics(Sender<Result<HashMap<String, f64>, String>>),

    /// Stop the file watcher
    StopWatcher,

    /// Stop the adapter
    Stop,
}

impl CliAdapter {
    /// Create a new CLI adapter
    pub async fn new(orchestrator: Arc<Orchestrator>, mesh: Arc<Mesh>) -> Result<Self, String> {
        // Create mesh node
        let mesh_node = Arc::new(MeshNode::new("cli-adapter", mesh).await?);

        // Create command channel
        let (cmd_tx, cmd_rx) = mpsc::channel(100);

        let adapter = Self {
            orchestrator,
            mesh_node,
            background_task: None,
            cmd_tx,
            running: false,
        };

        Ok(adapter)
    }

    /// Start the CLI adapter
    pub fn start(&mut self) -> Result<(), String> {
        if self.running {
            return Ok(());
        }

        let cmd_rx = mpsc::channel(100).1;
        let background_task = self.start_background_task(cmd_rx);
        self.background_task = Some(background_task);
        self.running = true;

        Ok(())
    }

    /// Start file watcher for auto-ingestion
    pub async fn start_file_watcher(
        &self,
        directory: impl AsRef<Path>,
        file_pattern: &str,
    ) -> Result<JoinHandle<()>, String> {
        let dir_path = PathBuf::from(directory.as_ref());
        let pattern = file_pattern.to_string();
        let cmd_tx = self.cmd_tx.clone();

        // Ensure directory exists
        if !dir_path.exists() || !dir_path.is_dir() {
            return Err(format!("Directory {} does not exist", dir_path.display()));
        }

        info!("Starting file watcher for {}", dir_path.display());

        // Start background task for watching files
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            let mut processed_files = std::collections::HashSet::new();

            loop {
                interval.tick().await;

                // Check for stop command
                // This is a simplification - in a real implementation we would use a channel
                // to signal the watcher to stop

                // Read directory
                match tokio::fs::read_dir(&dir_path).await {
                    Ok(mut entries) => {
                        while let Ok(Some(entry)) = entries.next_entry().await {
                            let path = entry.path();

                            // Check if this is a file and matches the pattern
                            if path.is_file() && path.to_string_lossy().contains(&pattern) {
                                let canonical_path = match path.canonicalize() {
                                    Ok(p) => p,
                                    Err(_) => continue,
                                };

                                // Skip if already processed
                                if processed_files.contains(&canonical_path) {
                                    continue;
                                }

                                // Process file
                                info!("Found new file: {}", path.display());
                                let (tx, rx) = mpsc::channel(1);
                                if let Err(e) = cmd_tx
                                    .send(CliCommand::QueueDocument(path.clone(), tx))
                                    .await
                                {
                                    error!("Failed to send QueueDocument command: {}", e);
                                    continue;
                                }

                                match rx.await {
                                    Ok(Ok(job_id)) => {
                                        info!("Queued file {} as job {}", path.display(), job_id);
                                        processed_files.insert(canonical_path);
                                    }
                                    Ok(Err(e)) => {
                                        error!("Failed to queue file {}: {}", path.display(), e);
                                    }
                                    Err(e) => {
                                        error!("Failed to receive response: {}", e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to read directory {}: {}", dir_path.display(), e);
                    }
                }
            }
        });

        Ok(task)
    }

    /// Queue a document for ingestion
    pub async fn queue_document(&self, path: impl AsRef<Path>) -> Result<String, String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(CliCommand::QueueDocument(path.as_ref().to_path_buf(), tx))
            .await
            .map_err(|_| "Failed to send command to CLI adapter".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from CLI adapter".to_string())?
    }

    /// Get the status of an ingest job
    pub async fn get_job_status(&self, job_id: &str) -> Result<String, String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(CliCommand::GetJobStatus(job_id.to_string(), tx))
            .await
            .map_err(|_| "Failed to send command to CLI adapter".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from CLI adapter".to_string())?
    }

    /// Get metrics about the ingest system
    pub async fn get_metrics(&self) -> Result<HashMap<String, f64>, String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(CliCommand::GetMetrics(tx))
            .await
            .map_err(|_| "Failed to send command to CLI adapter".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from CLI adapter".to_string())?
    }

    /// Stop the file watcher
    pub async fn stop_watcher(&self) -> Result<(), String> {
        self.cmd_tx
            .send(CliCommand::StopWatcher)
            .await
            .map_err(|_| "Failed to send command to CLI adapter".to_string())?;

        Ok(())
    }

    /// Stop the adapter
    pub async fn stop(&self) -> Result<(), String> {
        self.cmd_tx
            .send(CliCommand::Stop)
            .await
            .map_err(|_| "Failed to send command to CLI adapter".to_string())?;

        Ok(())
    }

    /// Start the background task
    fn start_background_task(&self, mut cmd_rx: Receiver<CliCommand>) -> JoinHandle<()> {
        let orchestrator = Arc::clone(&self.orchestrator);
        let mesh_node = Arc::clone(&self.mesh_node);

        tokio::spawn(async move {
            info!("CLI adapter started");

            // Main loop
            let mut running = true;
            while running {
                if let Some(cmd) = cmd_rx.recv().await {
                    match cmd {
                        CliCommand::QueueDocument(path, resp_tx) => {
                            // Create document from file
                            match Self::create_document_from_file(&path).await {
                                Ok(doc) => {
                                    // Queue document
                                    match orchestrator.queue_document(doc).await {
                                        Ok(job_id) => {
                                            let _ = resp_tx.send(Ok(job_id)).await;
                                        }
                                        Err(e) => {
                                            let _ = resp_tx.send(Err(e)).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = resp_tx
                                        .send(Err(format!("Failed to read file: {}", e)))
                                        .await;
                                }
                            }
                        }
                        CliCommand::GetJobStatus(job_id, resp_tx) => {
                            // Get job status
                            match orchestrator.get_job_status(&job_id).await {
                                Ok(job) => {
                                    let status = format!(
                                        "Job {} is {} ({}% complete)",
                                        job_id, job.status, job.percent_complete
                                    );
                                    let _ = resp_tx.send(Ok(status)).await;
                                }
                                Err(e) => {
                                    let _ = resp_tx.send(Err(e)).await;
                                }
                            }
                        }
                        CliCommand::GetMetrics(resp_tx) => {
                            // Simulate metrics for Day 1
                            let mut metrics = HashMap::new();
                            metrics.insert("queue_depth".to_string(), 0.0);
                            metrics.insert("active_jobs".to_string(), 0.0);
                            metrics.insert("processing_time_ms".to_string(), 0.0);

                            let _ = resp_tx.send(Ok(metrics)).await;
                        }
                        CliCommand::StopWatcher => {
                            // Not implemented yet
                        }
                        CliCommand::Stop => {
                            info!("Stopping CLI adapter");
                            running = false;
                        }
                    }
                }
            }

            info!("CLI adapter stopped");
        })
    }

    /// Create an IngestDocument from a file
    async fn create_document_from_file(path: &Path) -> io::Result<IngestDocument> {
        // Get file extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("txt");

        // Determine document type from extension
        let document_type = match ext.to_lowercase().as_str() {
            "pdf" => "pdf",
            "docx" | "doc" => "docx",
            "md" | "markdown" => "markdown",
            "txt" => "text",
            _ => "generic",
        };

        // Get file name as title
        let title = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Untitled")
            .to_string();

        // Create document
        let doc = IngestDocument::new(document_type)
            .with_title(title)
            .with_file_path(path);

        Ok(doc)
    }
}

impl Drop for CliAdapter {
    fn drop(&mut self) {
        // Cancel the background task if it's still running
        if let Some(task) = self.background_task.take() {
            task.abort();
        }
    }
}

/// Create a CLI command to ingest a file
pub async fn create_ingest_command(
    adapter: &CliAdapter,
    path: impl AsRef<Path>,
) -> Result<String, String> {
    adapter.queue_document(path).await
}

/// Create a CLI command to get job status
pub async fn create_status_command(adapter: &CliAdapter, job_id: &str) -> Result<String, String> {
    adapter.get_job_status(job_id).await
}

/// Create a CLI command to get metrics
pub async fn create_metrics_command(adapter: &CliAdapter) -> Result<HashMap<String, f64>, String> {
    adapter.get_metrics().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::orchestrator::{create_orchestrator, OrchestratorConfig};
    use crate::mesh::InMemoryMesh;
    use tempfile::NamedTempFile;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    async fn create_test_file(content: &str) -> io::Result<NamedTempFile> {
        let file = NamedTempFile::new()?;
        let mut file_handle = File::create(file.path()).await?;
        file_handle.write_all(content.as_bytes()).await?;
        Ok(file)
    }

    #[tokio::test]
    async fn test_cli_adapter_creation() {
        // Create orchestrator
        let orchestrator = create_orchestrator("test-orchestrator", None)
            .await
            .unwrap();

        // Create mesh
        let mesh = Arc::new(InMemoryMesh::new());

        // Create adapter
        let result = CliAdapter::new(orchestrator, mesh).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_queue_document() {
        // Create orchestrator
        let orchestrator = create_orchestrator("test-orchestrator", None)
            .await
            .unwrap();

        // Create mesh
        let mesh = Arc::new(InMemoryMesh::new());

        // Create adapter
        let mut adapter = CliAdapter::new(orchestrator, mesh).await.unwrap();
        adapter.start().unwrap();

        // Create test file
        let file = create_test_file("This is a test document").await.unwrap();

        // Queue document
        let result = adapter.queue_document(file.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_document_from_file() {
        // Create test file
        let file = create_test_file("This is a test document").await.unwrap();

        // Create document
        let doc = CliAdapter::create_document_from_file(file.path())
            .await
            .unwrap();

        // Check document
        assert_eq!(doc.document_type, "txt");
        assert!(doc.title.is_some());
        assert!(doc.file_path.is_some());
    }
}

