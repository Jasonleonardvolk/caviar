use crate::mesh::MeshNode;
use crate::diff::*;
use std::sync::Arc;
use crate::ConceptDiffRef;
//! Agentic Orchestrator
//!
//! The orchestrator is the central coordinator for the concept-agent mesh.
//! It receives ConceptPacks from the Concept Boundary Detector and orchestrates
//! their processing through the mesh via ConceptDiffs.

use crate::cbd::{ConceptPack, ConceptPackRef};
use crate::diff::{ConceptDiff, ConceptDiffBuilder, Op};
use crate::lcn::LargeConceptNetwork;
use crate::mesh::{Mesh, MeshNode, Pattern};
use crate::psiarc::{PsiarcLog, PsiarcManager, PsiarcOptions};

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, instrument, warn};

/// Status of an ingest job
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestStatus {
    /// Queued for processing
    Queued,
    /// Currently being processed
    Processing,
    /// Extraction phase
    Extracting,
    /// Chunking phase
    Chunking,
    /// Vectorizing phase
    Vectorizing,
    /// Concept mapping phase
    ConceptMapping,
    /// Storing phase
    Storing,
    /// Processing completed successfully
    Completed,
    /// Processing failed
    Failed,
}

impl std::fmt::Display for IngestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Processing => write!(f, "processing"),
            Self::Extracting => write!(f, "extracting"),
            Self::Chunking => write!(f, "chunking"),
            Self::Vectorizing => write!(f, "vectorizing"),
            Self::ConceptMapping => write!(f, "concept_mapping"),
            Self::Storing => write!(f, "storing"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// A document to ingest
#[derive(Debug, Clone)]
pub struct IngestDocument {
    /// Unique identifier for this document
    pub id: String,

    /// Type of document
    pub document_type: String,

    /// Title of the document
    pub title: Option<String>,

    /// Source URL or path
    pub source: Option<String>,

    /// Tags for this document
    pub tags: Vec<String>,

    /// Metadata for this document
    pub metadata: HashMap<String, serde_json::Value>,

    /// Path to the document content if it's a file
    pub file_path: Option<PathBuf>,

    /// Raw document content if available
    pub content: Option<String>,

    /// Priority of this ingest job (higher is more important)
    pub priority: u8,
}

impl IngestDocument {
    /// Create a new IngestDocument with a generated ID
    pub fn new(document_type: impl Into<String>) -> Self {
        let id = format!(
            "doc_{}_{}",
            document_type.into(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        Self::with_id(id, document_type)
    }

    /// Create a new IngestDocument with a specific ID
    pub fn with_id(id: impl Into<String>, document_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            document_type: document_type.into(),
            title: None,
            source: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
            file_path: None,
            content: None,
            priority: 1,
        }
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl serde::Serialize) -> Self {
        if let Ok(value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), value);
        }
        self
    }

    /// Set the file path
    pub fn with_file_path(mut self, path: impl AsRef<Path>) -> Self {
        self.file_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the content
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Read content from file path if available and not already loaded
    pub fn load_content(&mut self) -> std::io::Result<()> {
        if self.content.is_some() {
            return Ok(());
        }

        if let Some(path) = &self.file_path {
            let content = std::fs::read_to_string(path)?;
            self.content = Some(content);
        }

        Ok(())
    }
}

/// Status of an ingest job
#[derive(Debug)]
pub struct IngestJob {
    /// The document being ingested
    pub document: IngestDocument,

    /// Current status of the job
    pub status: IngestStatus,

    /// Timestamp when this job was created
    pub created_at: u64,

    /// Timestamp when this job was last updated
    pub updated_at: u64,

    /// Timestamp when this job was completed
    pub completed_at: Option<u64>,

    /// Percentage of completion (0-100)
    pub percent_complete: f32,

    /// Number of segments processed
    pub segments_processed: usize,

    /// Total number of segments
    pub segments_total: Option<usize>,

    /// Concept IDs mapped from this document
    pub concept_ids: HashSet<String>,

    /// Failure code if the job failed
    pub failure_code: Option<String>,

    /// Failure message if the job failed
    pub failure_message: Option<String>,
}

impl IngestJob {
    /// Create a new IngestJob
    pub fn new(document: IngestDocument) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            document,
            status: IngestStatus::Queued,
            created_at: now,
            updated_at: now,
            completed_at: None,
            percent_complete: 0.0,
            segments_processed: 0,
            segments_total: None,
            concept_ids: HashSet::new(),
            failure_code: None,
            failure_message: None,
        }
    }

    /// Update the status of this job
    pub fn update_status(&mut self, status: IngestStatus) {
        self.status = status;
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if status == IngestStatus::Completed || status == IngestStatus::Failed {
            self.completed_at = Some(self.updated_at);
            self.percent_complete = 100.0;
        }
    }

    /// Set this job as failed
    pub fn set_failed(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.status = IngestStatus::Failed;
        self.failure_code = Some(code.into());
        self.failure_message = Some(message.into());
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.completed_at = Some(self.updated_at);
    }

    /// Update the progress of this job
    pub fn update_progress(&mut self, percent_complete: f32) {
        self.percent_complete = percent_complete.clamp(0.0, 100.0);
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Add a processed segment
    pub fn add_segment(&mut self) {
        self.segments_processed += 1;

        if let Some(total) = self.segments_total {
            if total > 0 {
                self.percent_complete = (self.segments_processed as f32 / total as f32) * 100.0;
            }
        }

        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Set the total number of segments
    pub fn set_segments_total(&mut self, total: usize) {
        self.segments_total = Some(total);

        if total > 0 {
            self.percent_complete = (self.segments_processed as f32 / total as f32) * 100.0;
        }

        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Add a concept ID
    pub fn add_concept(&mut self, concept_id: impl Into<String>) {
        self.concept_ids.insert(concept_id.into());
    }
}

/// Configuration for the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Whether to emit the GENESIS event on startup
    pub genesis_on_startup: bool,

    /// Corpus ID to use for GENESIS
    pub corpus_id: String,

    /// Directory for logs
    pub log_directory: String,

    /// Name for psiarc logs
    pub log_name: String,

    /// Maximum number of concurrent ingest jobs
    pub max_concurrent_jobs: usize,

    /// Job timeout in seconds
    pub job_timeout_secs: u64,

    /// Max queue size
    pub max_queue_size: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            genesis_on_startup: true,
            corpus_id: "MainCorpus".to_string(),
            log_directory: "logs".to_string(),
            log_name: "orchestrator".to_string(),
            max_concurrent_jobs: 4,
            job_timeout_secs: 3600, // 1 hour
            max_queue_size: 1000,
        }
    }
}

/// Command to the orchestrator
#[derive(Debug)]
enum OrchestratorCommand {
    /// Queue a document for ingestion
    QueueDocument(IngestDocument, Sender<Result<String, String>>),

    /// Get the status of a job
    GetJobStatus(String, Sender<Result<Arc<IngestJob>, String>>),

    /// Process a ConceptPack
    ProcessConceptPack(ConceptPackRef),

    /// Explicitly trigger GENESIS
    TriggerGenesis(Sender<Result<(), String>>),

    /// Stop the orchestrator
    Stop,
}

/// The Agentic Orchestrator
pub struct Orchestrator {
    /// Node ID for this orchestrator
    id: String,

    /// Configuration
    config: OrchestratorConfig,

    /// Concept mesh node
    mesh_node: Arc<dyn MeshNode>,

    /// Large Concept Network
    lcn: Arc<LargeConceptNetwork>,

    /// Psiarc manager for logging
    psiarc_manager: Arc<PsiarcManager>,

    /// Queue of ingest jobs
    job_queue: Arc<Mutex<VecDeque<Arc<IngestJob>>>>,

    /// Active ingest jobs
    active_jobs: Arc<RwLock<HashMap<String, Arc<IngestJob>>>>,

    /// Command sender
    cmd_tx: Sender<OrchestratorCommand>,

    /// Concept subscriptions
    subscriptions: Arc<Mutex<Vec<usize>>>,

    /// Background task handle
    background_task: Option<JoinHandle<()>>,
}

impl Orchestrator {
    /// Create a new orchestrator with the given mesh and LCN
    pub async fn new(
        id: impl Into<String>,
        mesh: Arc<Mesh>,
        lcn: Arc<LargeConceptNetwork>,
        config: OrchestratorConfig,
    ) -> Result<Self, String> {
        let id = id.into();

        // Create mesh node
        let mesh_node = MeshNode::new(id.clone(), mesh).await?;

        // Create psiarc manager
        let psiarc_manager = Arc::new(PsiarcManager::new(&config.log_directory));

        // Create command channel
        let (cmd_tx, cmd_rx) = mpsc::channel(100);

        let orchestrator = Self {
            id,
            config,
            mesh_node: Arc::new(mesh_node),
            lcn,
            psiarc_manager,
            job_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            cmd_tx,
            subscriptions: Arc::new(Mutex::new(Vec::new())),
            background_task: None,
        };

        // Start background task
        let background_task = orchestrator.start_background_task(cmd_rx);

        Ok(Self {
            background_task: Some(background_task),
            ..orchestrator
        })
    }

    /// Get the command sender
    pub fn cmd_tx(&self) -> Sender<OrchestratorCommand> {
        self.cmd_tx.clone()
    }

    /// Queue a document for ingestion
    pub async fn queue_document(&self, document: IngestDocument) -> Result<String, String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(OrchestratorCommand::QueueDocument(document, tx))
            .await
            .map_err(|_| "Failed to send command to orchestrator".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from orchestrator".to_string())?
    }

    /// Get the status of a job
    pub async fn get_job_status(&self, job_id: &str) -> Result<Arc<IngestJob>, String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(OrchestratorCommand::GetJobStatus(job_id.to_string(), tx))
            .await
            .map_err(|_| "Failed to send command to orchestrator".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from orchestrator".to_string())?
    }

    /// Trigger GENESIS
    pub async fn trigger_genesis(&self) -> Result<(), String> {
        let (tx, rx) = mpsc::channel(1);
        self.cmd_tx
            .send(OrchestratorCommand::TriggerGenesis(tx))
            .await
            .map_err(|_| "Failed to send command to orchestrator".to_string())?;

        rx.await
            .map_err(|_| "Failed to receive response from orchestrator".to_string())?
    }

    /// Process a ConceptPack
    pub async fn process_concept_pack(&self, pack: ConceptPackRef) -> Result<(), String> {
        self.cmd_tx
            .send(OrchestratorCommand::ProcessConceptPack(pack))
            .await
            .map_err(|_| "Failed to send command to orchestrator".to_string())?;

        Ok(())
    }

    /// Stop the orchestrator
    pub async fn stop(&self) -> Result<(), String> {
        self.cmd_tx
            .send(OrchestratorCommand::Stop)
            .await
            .map_err(|_| "Failed to send command to orchestrator".to_string())?;

        Ok(())
    }

    /// Start the background task
    fn start_background_task(&self, mut cmd_rx: Receiver<OrchestratorCommand>) -> JoinHandle<()> {
        let id = self.id.clone();
        let mesh_node = Arc::clone(&self.mesh_node);
        let lcn = Arc::clone(&self.lcn);
        let psiarc_manager = Arc::clone(&self.psiarc_manager);
        let job_queue = Arc::clone(&self.job_queue);
        let active_jobs = Arc::clone(&self.active_jobs);
        let subscriptions = Arc::clone(&self.subscriptions);
        let config = self.config.clone();

        tokio::spawn(async move {
            // Create psiarc log
            if let Err(e) = psiarc_manager.create_log(&config.log_name) {
                error!("Failed to create psiarc log: {}", e);
            }

            // Subscribe to ingest requests
            let mut sub_id = 0;
            if let Ok(id) = mesh_node
                .subscribe(vec![Pattern::new("INGEST_REQUEST")], |diff| {
                    Self::handle_ingest_request_diff(diff, job_queue.clone(), self.cmd_tx.clone())
                })
                .await
            {
                subscriptions.lock().unwrap().push(id);
                sub_id = id;
            }

            // Emit GENESIS if configured
            if config.genesis_on_startup && !lcn.is_genesis_complete() {
                info!("Emitting GENESIS event");
                let genesis_diff = crate::diff::create_genesis_diff(&config.corpus_id);

                // Apply to LCN
                if let Err(e) = lcn.apply_diff(&genesis_diff) {
                    error!("Failed to apply GENESIS diff to LCN: {}", e);
                }

                // Record to psiarc
                if let Err(e) = psiarc_manager.record(&genesis_diff) {
                    error!("Failed to record GENESIS diff to psiarc: {}", e);
                }

                // Publish to mesh
                if let Err(e) = mesh_node.publish(genesis_diff).await {
                    error!("Failed to publish GENESIS diff to mesh: {}", e);
                }

                info!("GENESIS complete");
            }

            // Main loop
            let mut running = true;
            while running {
                tokio::select! {
                    cmd = cmd_rx.recv() => {
                        if let Some(cmd) = cmd {
                            match cmd {
                                OrchestratorCommand::QueueDocument(doc, resp_tx) => {
                                    // Create job
                                    let job = Arc::new(IngestJob::new(doc));
                                    let job_id = job.document.id.clone();

                                    // Add to queue if not full
                                    let mut queue = job_queue.lock().unwrap();
                                    if queue.len() < config.max_queue_size {
                                        queue.push_back(Arc::clone(&job));
                                        drop(queue);

                                        // Publish job queued diff
                                        let diff = Self::create_job_queued_diff(&job);
                                        if let Err(e) = mesh_node.publish(diff).await {
                                            error!("Failed to publish job queued diff: {}", e);
                                        }

                                        // Send response
                                        let _ = resp_tx.send(Ok(job_id)).await;
                                    } else {
                                        let _ = resp_tx.send(Err("Queue is full".to_string())).await;
                                    }
                                }
                                OrchestratorCommand::GetJobStatus(job_id, resp_tx) => {
                                    // Check active jobs first
                                    let active_jobs_guard = active_jobs.read().unwrap();
                                    if let Some(job) = active_jobs_guard.get(&job_id) {
                                        let _ = resp_tx.send(Ok(Arc::clone(job))).await;
                                        continue;
                                    }
                                    drop(active_jobs_guard);

                                    // Check queue
                                    let queue = job_queue.lock().unwrap();
                                    let job = queue.iter().find(|j| j.document.id == job_id);

                                    if let Some(job) = job {
                                        let _ = resp_tx.send(Ok(Arc::clone(job))).await;
                                    } else {
                                        let _ = resp_tx.send(Err(format!("Job {} not found", job_id))).await;
                                    }
                                }
                                OrchestratorCommand::ProcessConceptPack(pack) => {
                                    // Process the ConceptPack
                                    if let Err(e) = Self::process_concept_pack_internal(
                                        &pack,
                                        &mesh_node,
                                        &lcn,
                                        &psiarc_manager,
                                        &active_jobs,
                                    ).await {
                                        error!("Failed to process ConceptPack: {}", e);
                                    }
                                }
                                OrchestratorCommand::TriggerGenesis(resp_tx) => {
                                    info!("Triggering GENESIS event");
                                    let genesis_diff = crate::diff::create_genesis_diff(&config.corpus_id);

                                    // Apply to LCN
                                    if let Err(e) = lcn.apply_diff(&genesis_diff) {
                                        error!("Failed to apply GENESIS diff to LCN: {}", e);
                                        let _ = resp_tx.send(Err(format!("Failed to apply GENESIS diff: {}", e))).await;
                                        continue;
                                    }

                                    // Record to psiarc
                                    if let Err(e) = psiarc_manager.record(&genesis_diff) {
                                        error!("Failed to record GENESIS diff to psiarc: {}", e);
                                        let _ = resp_tx.send(Err(format!("Failed to record GENESIS diff: {}", e))).await;
                                        continue;
                                    }

                                    // Publish to mesh
                                    if let Err(e) = mesh_node.publish(genesis_diff).await {
                                        error!("Failed to publish GENESIS diff to mesh: {}", e);
                                        let _ = resp_tx.send(Err(format!("Failed to publish GENESIS diff: {}", e))).await;
                                        continue;
                                    }

                                    info!("GENESIS complete");
                                    let _ = resp_tx.send(Ok(())).await;
                                }
                                OrchestratorCommand::Stop => {
                                    info!("Stopping orchestrator");
                                    running = false;
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Process queued jobs if we have capacity
                        let active_count = active_jobs.read().unwrap().len();
                        if active_count < config.max_concurrent_jobs {
                            let slots_available = config.max_concurrent_jobs - active_count;
                            let mut jobs_to_start = Vec::new();

                            // Get jobs from queue
                            let mut queue = job_queue.lock().unwrap();
                            for _ in 0..slots_available {
                                if let Some(job) = queue.pop_front() {
                                    jobs_to_start.push(job);
                                } else {
                                    break;
                                }
                            }
                            drop(queue);

                            // Start jobs
                            for job in jobs_to_start {
                                // Update job status
                                {
                                    let mut job = Arc::get_mut(&mut job.clone()).unwrap();
                                    job.update_status(IngestStatus::Processing);
                                }

                                // Add to active jobs
                                active_jobs.write().unwrap().insert(job.document.id.clone(), Arc::clone(&job));

                                // Start processing
                                let job_id = job.document.id.clone();
                                info!("Starting ingest job {}", job_id);

                                // Publish job processing diff
                                let diff = Self::create_job_processing_diff(&job);
                                if let Err(e) = mesh_node.publish(diff).await {
                                    error!("Failed to publish job processing diff: {}", e);
                                }

                                // TODO: Start processing the job
                                // For Day 1, we'll just simulate processing

                                // For demo purposes, mark job as completed after a delay
                                let mesh_node_clone = Arc::clone(&mesh_node);
                                let active_jobs_clone = Arc::clone(&active_jobs);
                                let psiarc_manager_clone = Arc::clone(&psiarc_manager);
                                tokio::spawn(async move {
                                    tokio::time::sleep(Duration::from_secs(2)).await;

                                    // Mark job as completed
                                    let mut jobs = active_jobs_clone.write().unwrap();
                                    if let Some(job) = jobs.get_mut(&job_id) {
                                        let mut job = Arc::get_mut(job).unwrap();
                                        job.update_status(IngestStatus::Completed);

                                        // Publish job completed diff
                                        let diff = Self::create_job_completed_diff(job);
                                        let _ = mesh_node_clone.publish(diff).await;
                                    }

                                    // Remove from active jobs
                                    jobs.remove(&job_id);
                                });
                            }
                        }

                        // Check for timed out jobs
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;

                        let timeout_secs = config.job_timeout_secs;
                        let timeout_ms = timeout_secs * 1000;

                        let mut timed_out_jobs = Vec::new();

                        let mut active_jobs_guard = active_jobs.write().unwrap();
                        for (job_id, job) in active_jobs_guard.iter() {
                            if job.status != IngestStatus::Completed && job.status != IngestStatus::Failed {
                                if now - job.updated_at > timeout_ms {
                                    timed_out_jobs.push(job_id.clone());
                                }
                            }
                        }

                        // Mark timed out jobs as failed
                        for job_id in timed_out_jobs {
                            if let Some(job) = active_jobs_guard.get_mut(&job_id) {
                                let mut job = Arc::get_mut(job).unwrap();
                                job.set_failed("TIMEOUT", format!("Job timed out after {} seconds", timeout_secs));

                                // Publish job failed diff
                                let diff = Self::create_job_failed_diff(job);
                                let _ = mesh_node.publish(diff).await;
                            }
                        }
                    }
                }
            }

            // Unsubscribe from mesh
            for sub_id in subscriptions.lock().unwrap().iter() {
                let _ = mesh_node.unsubscribe(*sub_id).await;
            }

            // Disconnect from mesh
            let _ = mesh_node.disconnect().await;

            info!("Orchestrator stopped");
        })
    }

    /// Handle an ingest request ConceptDiff
    async fn handle_ingest_request_diff(
        diff: ConceptDiffRef,
        job_queue: Arc<Mutex<VecDeque<Arc<IngestJob>>>>,
        cmd_tx: Sender<OrchestratorCommand>,
    ) {
        // Extract ingest request details from the diff
        // TODO: Implement full diff parsing

        // For Day 1, just handle a simple struct
        if let Some(metadata) = diff.metadata.get("ingest_request") {
            if let Ok(doc) = serde_json::from_value::<IngestDocument>(metadata.clone()) {
                let _ = cmd_tx
                    .send(OrchestratorCommand::QueueDocument(doc, mpsc::channel(1).0))
                    .await;
            }
        }
    }

    /// Process a ConceptPack
    async fn process_concept_pack_internal(
        pack: &ConceptPack,
        mesh_node: &dyn MeshNode,
        lcn: &LargeConceptNetwork,
        psiarc_manager: &PsiarcManager,
        active_jobs: &RwLock<HashMap<String, Arc<IngestJob>>>,
    ) -> Result<(), String> {
        // Extract job ID from ConceptPack (assuming it's stored in metadata)
        let job_id = if let Some(job_id) = pack.metadata.get("job_id") {
            job_id.as_str().unwrap_or("unknown").to_string()
        } else {
            "unknown".to_string()
        };

        // Update job status
        let mut job_updated = false;
        {
            let mut jobs = active_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                let mut job = Arc::get_mut(job).unwrap();
                job.add_segment();
                job_updated = true;
            }
        }

        // Create a diff to update the LCN with this ConceptPack
        let mut diff_builder = ConceptDiffBuilder::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        // Create a node for this concept pack
        let pack_node_id = format!("pack_{}", pack.id);
        let mut properties = HashMap::new();

        // Add key metadata from the pack
        properties.insert("content".to_string(), serde_json::json!(pack.content));
        properties.insert(
            "boundary_reason".to_string(),
            serde_json::json!(format!("{:?}", pack.boundary_reason)),
        );

        if !pack.concepts.is_empty() {
            properties.insert("concepts".to_string(), serde_json::json!(pack.concepts));
        }

        // Add all metadata
        for (key, value) in &pack.metadata {
            properties.insert(format!("meta_{}", key), value.clone());
        }

        // Create node
        diff_builder = diff_builder.create(&pack_node_id, "ConceptPack", properties);

        // Link to job if we have a valid job ID
        if job_id != "unknown" {
            diff_builder = diff_builder.bind(&job_id, &pack_node_id, Some("CONTAINS".to_string()));
        }

        // For each concept in the pack, create or update it and link
        for (concept_id, strength) in &pack.concepts {
            // Check if concept exists
            if lcn.get_node(concept_id).is_none() {
                // Create concept
                let mut concept_props = HashMap::new();
                concept_props.insert("type".to_string(), serde_json::json!("Concept"));
                concept_props.insert("source".to_string(), serde_json::json!(pack.id));

                diff_builder = diff_builder.create(concept_id, "Concept", concept_props);
            }

            // Link pack to concept
            let mut link_props = HashMap::new();
            link_props.insert("strength".to_string(), serde_json::json!(strength));

            diff_builder = diff_builder.link(
                &pack_node_id,
                concept_id,
                "HAS_CONCEPT".to_string(),
                link_props,
            );

            // Update job to add this concept
            if job_updated {
                let mut jobs = active_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    let mut job = Arc::get_mut(job).unwrap();
                    job.add_concept(concept_id);
                }
            }
        }

        // Build the diff
        let diff = diff_builder.build();

        // Apply to LCN
        if let Err(e) = lcn.apply_diff(&diff) {
            return Err(format!("Failed to apply diff to LCN: {}", e));
        }

        // Record to psiarc
        if let Err(e) = psiarc_manager.record(&diff) {
            return Err(format!("Failed to record diff to psiarc: {}", e));
        }

        // Publish to mesh
        if let Err(e) = mesh_node.publish(diff).await {
            return Err(format!("Failed to publish diff to mesh: {}", e));
        }

        Ok(())
    }

    /// Create a ConceptDiff for a queued job
    fn create_job_queued_diff(job: &IngestJob) -> ConceptDiff {
        let mut diff_builder = ConceptDiffBuilder::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        // Create job node
        let mut properties = HashMap::new();
        properties.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        properties.insert(
            "document_type".to_string(),
            serde_json::json!(job.document.document_type.clone()),
        );
        properties.insert("created_at".to_string(), serde_json::json!(job.created_at));
        properties.insert("updated_at".to_string(), serde_json::json!(job.updated_at));

        if let Some(title) = &job.document.title {
            properties.insert("title".to_string(), serde_json::json!(title));
        }

        if let Some(source) = &job.document.source {
            properties.insert("source".to_string(), serde_json::json!(source));
        }

        if !job.document.tags.is_empty() {
            properties.insert(
                "tags".to_string(),
                serde_json::json!(job.document.tags.clone()),
            );
        }

        diff_builder = diff_builder.create(&job.document.id, "IngestJob", properties);

        // Add signal for job queued
        let mut payload = HashMap::new();
        payload.insert(
            "job_id".to_string(),
            serde_json::json!(job.document.id.clone()),
        );
        payload.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );

        diff_builder = diff_builder.signal("JOB_QUEUED", None, payload);

        diff_builder.build()
    }

    /// Create a ConceptDiff for a processing job
    fn create_job_processing_diff(job: &IngestJob) -> ConceptDiff {
        let mut diff_builder = ConceptDiffBuilder::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        // Update job node
        let mut properties = HashMap::new();
        properties.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        properties.insert("updated_at".to_string(), serde_json::json!(job.updated_at));
        properties.insert(
            "percent_complete".to_string(),
            serde_json::json!(job.percent_complete),
        );

        diff_builder = diff_builder.update(&job.document.id, properties);

        // Add signal for job processing
        let mut payload = HashMap::new();
        payload.insert(
            "job_id".to_string(),
            serde_json::json!(job.document.id.clone()),
        );
        payload.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        payload.insert(
            "percent_complete".to_string(),
            serde_json::json!(job.percent_complete),
        );

        diff_builder = diff_builder.signal("JOB_PROCESSING", None, payload);

        diff_builder.build()
    }

    /// Create a ConceptDiff for a completed job
    fn create_job_completed_diff(job: &IngestJob) -> ConceptDiff {
        let mut diff_builder = ConceptDiffBuilder::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        // Update job node
        let mut properties = HashMap::new();
        properties.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        properties.insert("updated_at".to_string(), serde_json::json!(job.updated_at));
        properties.insert(
            "completed_at".to_string(),
            serde_json::json!(job.completed_at),
        );
        properties.insert(
            "percent_complete".to_string(),
            serde_json::json!(job.percent_complete),
        );

        if !job.concept_ids.is_empty() {
            properties.insert(
                "concept_ids".to_string(),
                serde_json::json!(job.concept_ids.iter().cloned().collect::<Vec<_>>()),
            );
        }

        diff_builder = diff_builder.update(&job.document.id, properties);

        // Add signal for job completed
        let mut payload = HashMap::new();
        payload.insert(
            "job_id".to_string(),
            serde_json::json!(job.document.id.clone()),
        );
        payload.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        payload.insert(
            "concept_count".to_string(),
            serde_json::json!(job.concept_ids.len()),
        );

        diff_builder = diff_builder.signal("JOB_COMPLETED", None, payload);

        diff_builder.build()
    }

    /// Create a ConceptDiff for a failed job
    fn create_job_failed_diff(job: &IngestJob) -> ConceptDiff {
        let mut diff_builder = ConceptDiffBuilder::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );

        // Update job node
        let mut properties = HashMap::new();
        properties.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );
        properties.insert("updated_at".to_string(), serde_json::json!(job.updated_at));
        properties.insert(
            "completed_at".to_string(),
            serde_json::json!(job.completed_at),
        );

        if let Some(code) = &job.failure_code {
            properties.insert("failure_code".to_string(), serde_json::json!(code));
        }

        if let Some(message) = &job.failure_message {
            properties.insert("failure_message".to_string(), serde_json::json!(message));
        }

        diff_builder = diff_builder.update(&job.document.id, properties);

        // Add signal for job failed
        let mut payload = HashMap::new();
        payload.insert(
            "job_id".to_string(),
            serde_json::json!(job.document.id.clone()),
        );
        payload.insert(
            "status".to_string(),
            serde_json::json!(job.status.to_string()),
        );

        if let Some(code) = &job.failure_code {
            payload.insert("failure_code".to_string(), serde_json::json!(code));
        }

        if let Some(message) = &job.failure_message {
            payload.insert("failure_message".to_string(), serde_json::json!(message));
        }

        diff_builder = diff_builder.signal("JOB_FAILED", None, payload);

        diff_builder.build()
    }
}

impl Drop for Orchestrator {
    fn drop(&mut self) {
        // Cancel the background task if it's still running
        if let Some(task) = self.background_task.take() {
            task.abort();
        }
    }
}

/// Convenience function to create the Agentic Orchestrator and the required components.
/// This sets up the entire concept-mesh infrastructure required for TORI ingest.
pub async fn create_orchestrator(
    id: impl Into<String>,
    config: Option<OrchestratorConfig>,
) -> Result<Arc<Orchestrator>, String> {
    // Create in-memory mesh
    let mesh = Arc::new(crate::mesh::InMemoryMesh::new());

    // Create LCN
    let lcn = Arc::new(LargeConceptNetwork::new());

    // Use provided config or default
    let config = config.unwrap_or_default();

    // Create orchestrator
    let orchestrator = Orchestrator::new(id, mesh, lcn, config).await?;

    Ok(Arc::new(orchestrator))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::InMemoryMesh;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let mesh = Arc::new(InMemoryMesh::new());
        let lcn = Arc::new(LargeConceptNetwork::new());
        let result = Orchestrator::new(
            "test-orchestrator",
            mesh,
            lcn,
            OrchestratorConfig::default(),
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_queue_document() {
        let orchestrator = create_orchestrator("test-orchestrator", None)
            .await
            .unwrap();

        let doc = IngestDocument::new("test")
            .with_title("Test Document")
            .with_content("This is a test document");

        let result = orchestrator.queue_document(doc).await;
        assert!(result.is_ok());

        // Give the background task some time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check job status
        let job_id = result.unwrap();
        let status_result = orchestrator.get_job_status(&job_id).await;
        assert!(status_result.is_ok());

        let job = status_result.unwrap();
        assert_eq!(job.status, IngestStatus::Queued);
    }

    #[tokio::test]
    async fn test_genesis_event() {
        let mesh = Arc::new(InMemoryMesh::new());
        let lcn = Arc::new(LargeConceptNetwork::new());

        // Create config with genesis_on_startup = false to test manual triggering
        let config = OrchestratorConfig {
            genesis_on_startup: false,
            ..Default::default()
        };

        let orchestrator = Orchestrator::new(
            "test-orchestrator",
            Arc::clone(&mesh),
            Arc::clone(&lcn),
            config,
        )
        .await
        .unwrap();

        // Trigger GENESIS
        let result = orchestrator.trigger_genesis().await;
        assert!(result.is_ok());

        // Verify LCN state
        assert!(lcn.is_genesis_complete());
        assert!(lcn.has_timeless_root());
    }
}



